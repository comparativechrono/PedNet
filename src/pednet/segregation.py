"""How strongly each candidate variant co-segregates with the condition.

For each variant, PedNet compares two explanations of the family's phenotypes:

- the variant causes the condition: the genotypes at the disease locus are the variant's genotypes, so the people
  who were sequenced fix part of the picture, and the genotypes of the rest are summed over
- it doesn't: the condition comes from some other, unseen locus, which the proband carries (the family having come
  to notice through the proband), and whose genotypes in everyone else are summed over

The ratio of the two probabilities is the variant's co-segregation likelihood ratio (a Bayes factor). It is 2 to
the power of the number of informative meioses for a variant that follows the condition exactly with full
penetrance, the measure used for the co-segregation criteria of variant classification. Above 1, the variant
follows the condition through the family better than an unseen locus would be expected to; below 1, worse. Both
explanations average over the same penetrances, and over the modes of inheritance that suit the variant's
chromosome (autosomal dominant and recessive for an autosomal variant, X-linked recessive and dominant for one on
the X), each weighted by how well it explains the phenotypes.

Candidates are ranked by their odds against an unseen locus: the Bayes factor times the probability that the mode
is one that suits the candidate's chromosome, so that an X-linked variant ranks low in a family whose condition is
clearly autosomal, and the other way round.

Pairs of variants in the same gene are scored in the same way as compound heterozygotes, under the recessive modes
(see compound), and rank alongside the single variants.

The Bayes factor is the full-likelihood Bayes factor for co-segregation of Thompson et al. (2003), as segregatr
computes it, except that PedNet conditions on the proband having the disease genotype rather than on the proband's
genotype call: the same condition for every variant, so that their Bayes factors can be compared. Its strength as
evidence for the PP1 criterion of the ACMG/AMP guidelines (or against, BS4) is given by the Bayes factor thresholds
of Tavtigian et al. (2018), as Belman et al. (2020) apply them to co-segregation: 2.08 for supporting evidence, 4.33
moderate and 18.7 strong, and their reciprocals against.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .engine import Network
from .inheritance import (DEFAULT_PENETRANCE, DEFAULT_PHENOCOPY, LN10, _logsumexp, _modes, _prior, _weights,
                          mode_log_likelihoods)
from .models import MODES, X_LINKED, Model, call_codes, founder_prior, genotype_table, phenotype_factor
from .pedigree import MALE, Pedigree
from .compound import PAIR_MODES, PairLocus, has_disease_genotype, pair_log_likelihoods

DEFAULT_VARIANT_FREQUENCY = 0.001
# Rarer variants are treated as having this frequency. Below it, the frequency hardly changes how the variant can
# pass through the family, and a variant seen in the family is there, however rare it is in the population.
MIN_VARIANT_FREQUENCY = 0.001
DEFAULT_ERROR_RATE = 1e-4
_X_NAMES = {'x', 'chrx', '23', 'chr23'}

# The strengths of evidence for co-segregation (PP1) or against it (BS4), by the log10 Bayes factor: the thresholds of
# Tavtigian et al. (2018) for the ACMG/AMP criteria, 18.7 strong, 4.33 moderate and 2.08 supporting
EVIDENCE = ((np.log10(18.7), 'Strong'), (np.log10(4.33), 'Moderate'), (np.log10(2.08), 'Supporting'))


def evidence_strength(log10_bf: Optional[float]) -> str:
    """PP1_Strong, PP1_Moderate or PP1_Supporting for co-segregation, BS4_ likewise against it, or '' for neither"""
    if log10_bf is None:
        return ''
    for threshold, strength in EVIDENCE:
        if log10_bf >= threshold - 1e-12:
            return f'PP1_{strength}'
        if log10_bf <= -threshold + 1e-12:
            return f'BS4_{strength}'
    return ''


@dataclass
class Variant:
    """
    A candidate variant: its genotypes, as the number of copies of the alternate allele each person carries (0, 1 or 2;
    None, or no entry, if not genotyped), with where it is if known, its allele frequency in the population, and
    the gene it is in (which pairs variants up as compound heterozygotes).
    """
    id: str
    genotypes: Dict[str, Optional[int]]
    chrom: Optional[str] = None
    pos: Optional[int] = None
    ref: Optional[str] = None
    alt: Optional[str] = None
    allele_frequency: Optional[float] = None
    gene: Optional[str] = None

    @property
    def x_linked(self) -> Optional[bool]:
        """True on the X chromosome, False on another, None if the chromosome isn't known"""
        if self.chrom is None:
            return None
        return str(self.chrom).strip().lower() in _X_NAMES


@dataclass
class VariantScore:
    """
    One candidate's evidence: a variant, or with a partner, a pair of variants in one gene as a compound
    heterozygote.

    log10_lr is the log10 likelihood ratio for co-segregation under each mode that suits the candidate's chromosome
    on its own (only AR, or XLR, for a pair). log10_bf, the log10 co-segregation Bayes factor, combines them, each
    weighted by how probable the family's phenotypes make that mode: it is near 0 for a variant whose genotypes say
    nothing, and evidence is its strength as evidence for co-segregation (PP1) or against it (BS4). mode_probability
    is the probability, from the phenotypes, that the mode is one that suits the candidate (autosomal or X-linked,
    or recessive for a pair), and log10_odds = log10_bf + log10(mode_probability): the log10 odds that this
    candidate, rather than some other locus, causes the condition, with even odds beforehand. The candidates are
    ranked by it, so that an X-linked variant in a family whose condition is clearly autosomal doesn't rank highly.
    rank is 1 for the candidate with the most evidence, with ties sharing their ranks.

    The rest describes the genotypes. A carrier has the variant (for a pair, the genotype of a compound heterozygote:
    both variants, or two copies of either). concordance is the share of the genotyped people with a known phenotype
    whose genotype fits it under the best mode, as a plain check. mendelian_errors counts the genotyped children whose
    genotype their genotyped parents can't have given them, which suggests calling errors: the Bayes factor allows
    for errors, but such a variant is worth checking. de_novo lists the genotyped children who carry the variant
    though both their parents were genotyped without it (on the X, a son whose mother was): the Bayes factor counts
    such a variant against itself, since PedNet doesn't model new mutations, and it needs judging on its own.
    """
    variant: Variant
    log10_bf: Optional[float]
    log10_lr: Dict[str, float] = field(default_factory=dict)
    rank: Optional[float] = None
    log10_odds: Optional[float] = None
    mode_probability: Optional[float] = None
    genotyped: int = 0
    affected_carriers: int = 0
    affected_noncarriers: int = 0
    unaffected_carriers: int = 0
    unaffected_noncarriers: int = 0
    best_mode: Optional[str] = None
    concordance: Optional[float] = None
    mendelian_errors: int = 0
    note: str = ''
    partner: Optional[Variant] = None
    de_novo: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.variant.id if self.partner is None else f'{self.variant.id} + {self.partner.id}'

    @property
    def compound(self) -> bool:
        return self.partner is not None

    @property
    def evidence(self) -> str:
        return evidence_strength(self.log10_bf)

    @property
    def notes(self) -> str:
        """The note, with the Mendelian errors and possible new mutations"""
        parts = [self.note] if self.note else []
        if self.mendelian_errors:
            parts.append(f'{self.mendelian_errors} Mendelian error{"s" if self.mendelian_errors > 1 else ""}')
        if self.de_novo:
            parts.append(f'possibly de novo in {", ".join(self.de_novo)}')
        return '; '.join(parts)

    def row(self) -> dict:
        v, w = self.variant, self.partner

        def both(name):
            if w is None:
                return getattr(v, name)
            return ','.join('' if x is None else str(x) for x in (getattr(v, name), getattr(w, name)))

        row = {'rank': self.rank, 'candidate': self.id, 'kind': 'compound heterozygote' if w else 'variant',
               'gene': v.gene, 'chrom': v.chrom, 'pos': both('pos'), 'ref': both('ref'), 'alt': both('alt'),
               'allele_frequency': both('allele_frequency'), 'log10_bf': self.log10_bf, 'evidence': self.evidence,
               'log10_odds': self.log10_odds, 'mode_probability': self.mode_probability, 'best_mode': self.best_mode}
        for mode in MODES:
            row[f'log10_lr_{mode}'] = self.log10_lr.get(mode)
        row.update(genotyped=self.genotyped, affected_carriers=self.affected_carriers,
                   affected_noncarriers=self.affected_noncarriers, unaffected_carriers=self.unaffected_carriers,
                   unaffected_noncarriers=self.unaffected_noncarriers, concordance=self.concordance,
                   mendelian_errors=self.mendelian_errors, de_novo=','.join(self.de_novo), note=self.note)
        return row


def _fits(mode: str, copies: int, male: Optional[bool], affected: bool) -> bool:
    """Whether a genotype fits a phenotype under a mode with full penetrance and no phenocopies"""
    if mode == 'AD':
        disease = copies >= 1
    elif mode == 'AR':
        disease = copies == 2
    elif mode == 'XLD':
        disease = copies >= 1
    else:
        disease = copies >= 1 if male else copies == 2
    return disease == affected


def fair_ranks(values: Sequence[Optional[float]], decimals: int = 9) -> List[Optional[float]]:
    """
    Ranks, highest first, with tied values sharing the average of their ranks, and None for a missing value. Values
    equal to the given number of decimal places are tied, so that likelihood ratios that are equal in theory but differ
    in the last digits of the arithmetic share their rank.
    """
    rounded = [None if v is None else round(v, decimals) for v in values]
    count = Counter(v for v in rounded if v is not None)
    higher, above = {}, 0
    for v in sorted(count, reverse=True):
        higher[v] = above
        above += count[v]
    return [None if v is None else 1 + higher[v] + (count[v] - 1) / 2 for v in rounded]


def _frequency(v: Variant, default: float, lowest: float) -> float:
    q = v.allele_frequency if v.allele_frequency is not None else default
    return float(np.clip(q, max(lowest, 1e-8), 0.5))


def rank_variants(pedigree: Pedigree, variants: Iterable[Variant], modes: Sequence[str] = MODES,
                  penetrance=DEFAULT_PENETRANCE, penetrance_weights=None, phenocopy: float = DEFAULT_PHENOCOPY,
                  allele_frequency: Optional[Dict[str, float]] = None, carrier_penetrance: Optional[float] = None,
                  variant_frequency: float = DEFAULT_VARIANT_FREQUENCY, min_frequency: float = MIN_VARIANT_FREQUENCY,
                  error_rate: float = DEFAULT_ERROR_RATE, prior: Optional[Dict[str, float]] = None,
                  proband: Optional[str] = None, compound: bool = True, chunk: int = 2000) -> List[VariantScore]:
    """
    Scores candidate variants for co-segregation with the condition in one family, and ranks them, the strongest
    evidence first.

    variants: Variant objects (see read_vcf and read_genotype_table); people not in the pedigree are ignored
    modes: the modes of inheritance to allow (by default all four); give one if the mode is known
    penetrance, penetrance_weights, phenocopy, allele_frequency, carrier_penetrance, prior, proband: as for
        infer_mode; allele_frequency is that of the unseen locus in the explanation without the variant, and proband
        the person that locus is taken to be carried by (by default, each affected person in turn)
    variant_frequency: the allele frequency to use for a variant that doesn't give its own
    min_frequency: the lowest frequency used; rarer variants are treated as having this one. If it were far below
        error_rate, a genotype call in one person would be put down to an error rather than to the variant
    error_rate: the chance that a genotype call is one copy out (see models.genotype_factor)
    compound: whether to score pairs of variants in one gene (see Variant.gene) as compound heterozygotes too, when a
        recessive mode is allowed: the pairs that an affected person (the proband, if given) carries both of
    """
    modes = _modes(modes)
    priors = _prior(modes, prior)
    pens = (float(penetrance),) if np.isscalar(penetrance) else tuple(float(f) for f in penetrance)
    with np.errstate(divide='ignore'):
        log_w = np.log(_weights(pens, penetrance_weights))
    allele_frequency = allele_frequency or {}
    variants = list(variants)
    networks: Dict[bool, Network] = {}
    if proband is not None and proband not in pedigree:
        raise ValueError(f'{proband} is not in family {pedigree.family}')
    models = {m: Model(m, allele_frequency.get(m), pens[0], phenocopy, carrier_penetrance) for m in modes}

    # the explanation without the variant: P(phenotypes | the proband carries the disease genotype, mode), averaged
    # over the penetrances and the probands, as for the mode of inheritance
    log_without = mode_log_likelihoods(pedigree, modes, pens, penetrance_weights, phenocopy, allele_frequency,
                                       carrier_penetrance, [proband] if proband else None, networks)
    log_modes = {m: np.log(priors[m]) + log_without[m] for m in modes if priors[m] > 0}

    scores = [VariantScore(v, None) for v in variants]
    errors = [mendelian_errors(pedigree, v) for v in variants]
    for k, v in enumerate(variants):
        _describe(pedigree, v, scores[k])
        scores[k].mendelian_errors = errors[k]
        scores[k].de_novo = de_novo(pedigree, v)

    # the explanation with the variant as the cause, for each mode that suits its chromosome
    for x in (False, True):
        chosen = [k for k, v in enumerate(variants) if v.x_linked in (x, None) and scores[k].genotyped]
        suited = [m for m in modes if X_LINKED[m] == x and priors[m] > 0]
        if not chosen or not suited:
            continue
        if x not in networks:
            networks[x] = Network(pedigree, x)
        for start in range(0, len(chosen), chunk):
            batch = chosen[start:start + chunk]
            _score_batch(pedigree, networks[x], [variants[k] for k in batch], [scores[k] for k in batch], suited,
                         models, pens, log_w, variant_frequency, min_frequency, error_rate, log_without)

    # pairs of variants in one gene, as compound heterozygotes
    pairs = []
    if compound:
        allowed = {x: [m for m in PAIR_MODES if m in modes and priors[m] > 0 and x in (X_LINKED[m], None)]
                   for x in (False, True, None)}
        pairs = [(i, j) for i, j in candidate_pairs(pedigree, variants, proband) if allowed[variants[i].x_linked]]
    pair_scores = []
    for i, j in pairs:
        score = VariantScore(variants[i], None, partner=variants[j], mendelian_errors=errors[i] + errors[j],
                             de_novo=sorted(set(scores[i].de_novo) | set(scores[j].de_novo)))
        _describe_pair(pedigree, variants[i], variants[j], score)
        pair_scores.append(score)
    for x in (False, True):
        chosen = [k for k, (i, j) in enumerate(pairs) if variants[i].x_linked in (x, None)]
        suited = [m for m in PAIR_MODES if m in modes and X_LINKED[m] == x and priors[m] > 0]
        if not chosen or not suited:
            continue
        net = Network(pedigree, x, PairLocus())
        for start in range(0, len(chosen), chunk):
            batch = chosen[start:start + chunk]
            both = [(variants[pairs[k][0]], variants[pairs[k][1]]) for k in batch]
            frequencies = [(_frequency(a, variant_frequency, min_frequency), _frequency(b, variant_frequency, min_frequency))
                           for a, b in both]
            logs = pair_log_likelihoods(pedigree, net, both, frequencies, suited, models, pens, log_w, error_rate)
            for mode, values in logs.items():
                for k, value in zip(batch, values):
                    pair_scores[k].log10_lr[mode] = float((value - log_without[mode]) / LN10)

    groups: Dict[Tuple[str, ...], List[VariantScore]] = {}
    for s in scores + pair_scores:
        if not s.genotyped:
            s.note = 'no one in the family was genotyped for it'
        elif not s.log10_lr:
            s.note = 'none of the modes allowed suits its chromosome'
        else:
            groups.setdefault(tuple(s.log10_lr), []).append(s)
    for suited, group in groups.items():
        _combine_modes(group, suited, log_modes)
        for s in group:
            s.best_mode = max(suited, key=lambda m: s.log10_lr[m])
            if s.partner is None:
                s.concordance = _concordance(pedigree, s.variant, s.best_mode)
            else:
                s.concordance = _pair_concordance(pedigree, s.variant, s.partner)
    scores = scores + pair_scores
    for s, r in zip(scores, fair_ranks([s.log10_odds for s in scores])):
        s.rank = r
    scores.sort(key=lambda s: (s.rank is None, s.rank if s.rank is not None else 0))
    return scores


def _combine_modes(group: List['VariantScore'], suited: Sequence[str], log_modes: Dict[str, float]):
    """
    The Bayes factors and odds of candidates whose log10 likelihood ratios are for the modes suited, from the log of
    each mode's prior times the likelihood of the phenotypes under it (log_modes, for every mode allowed)
    """
    log_all = _logsumexp(list(log_modes.values()))
    log_suited = _logsumexp([log_modes[m] for m in suited])
    if not np.isfinite(log_suited) or not np.isfinite(log_all):
        for s in group:
            s.note = s.note or 'the phenotypes rule out every mode that suits its chromosome'
        return
    lr = np.array([[s.log10_lr[m] * LN10 for m in suited] for s in group])
    log_with = _logsumexp(lr + np.array([log_modes[m] for m in suited])[None, :], axis=1)
    for s, value in zip(group, np.atleast_1d(log_with)):
        s.log10_bf = float((value - log_suited) / LN10)
        s.mode_probability = float(np.exp(log_suited - log_all))
        s.log10_odds = float((value - log_all) / LN10)


def candidate_pairs(pedigree: Pedigree, variants: Sequence[Variant],
                    proband: Optional[str] = None) -> List[Tuple[int, int]]:
    """
    The pairs of variants in one gene, as pairs of positions in variants, that an affected person (the proband, if
    given) carries both of; on the X, an affected person who isn't a man, since a man has only one copy
    """
    who = [proband] if proband else pedigree.affected()
    by_gene: Dict[str, List[int]] = {}
    for k, v in enumerate(variants):
        if v.gene:
            by_gene.setdefault(v.gene, []).append(k)
    pairs = []
    for ks in by_gene.values():
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                v, w = variants[ks[a]], variants[ks[b]]
                if v.x_linked != w.x_linked:
                    continue
                for pid in who:
                    if pid not in pedigree or (v.x_linked and pedigree[pid].sex == MALE):
                        continue
                    if (v.genotypes.get(pid) or 0) >= 1 and (w.genotypes.get(pid) or 0) >= 1:
                        pairs.append((ks[a], ks[b]))
                        break
    return pairs


def _describe(pedigree: Pedigree, v: Variant, s: VariantScore):
    for pid, copies in v.genotypes.items():
        if copies is None or pid not in pedigree:
            continue
        person = pedigree[pid]
        s.genotyped += 1
        carrier = copies >= 1
        if person.affected:
            if carrier:
                s.affected_carriers += 1
            else:
                s.affected_noncarriers += 1
        elif person.unaffected:
            if carrier:
                s.unaffected_carriers += 1
            else:
                s.unaffected_noncarriers += 1


def _describe_pair(pedigree: Pedigree, v: Variant, w: Variant, s: VariantScore):
    for pid in pedigree.ids:
        a, b = v.genotypes.get(pid), w.genotypes.get(pid)
        if a is None and b is None:
            continue
        s.genotyped += 1
        if a is None or b is None:
            continue
        person = pedigree[pid]
        carrier = has_disease_genotype(a, b, v.x_linked and person.sex == MALE)
        if person.affected:
            if carrier:
                s.affected_carriers += 1
            else:
                s.affected_noncarriers += 1
        elif person.unaffected:
            if carrier:
                s.unaffected_carriers += 1
            else:
                s.unaffected_noncarriers += 1


def _pair_concordance(pedigree: Pedigree, v: Variant, w: Variant) -> Optional[float]:
    fits = total = 0
    for pid in pedigree.ids:
        a, b = v.genotypes.get(pid), w.genotypes.get(pid)
        person = pedigree[pid]
        if a is None or b is None or person.phenotype not in (1, 2):
            continue
        total += 1
        fits += has_disease_genotype(a, b, v.x_linked and person.sex == MALE) == person.affected
    return fits / total if total else None


def de_novo(pedigree: Pedigree, v: Variant) -> List[str]:
    """
    The genotyped children who carry the variant though both their parents were genotyped without it; on the X, a
    son who carries it though his mother was genotyped without it
    """
    found = []
    x = v.x_linked
    for p in pedigree.people:
        child = v.genotypes.get(p.id)
        if not child or p.father is None:
            continue
        father, mother = v.genotypes.get(p.father), v.genotypes.get(p.mother)
        if x and p.sex == MALE:
            if mother == 0:
                found.append(p.id)
        elif father == 0 and mother == 0:
            found.append(p.id)
    return found


def mendelian_errors(pedigree: Pedigree, v: Variant) -> int:
    """The genotyped children whose genotype can't have come from their genotyped parents"""
    x = v.x_linked
    errors = 0
    for p in pedigree.people:
        child = v.genotypes.get(p.id)
        if child is None or p.father is None:
            continue
        father, mother = v.genotypes.get(p.father), v.genotypes.get(p.mother)
        if x:
            # a son has only his mother's X; a daughter has her father's (all of it) and one of her mother's
            if p.sex == 1:
                if mother is not None and ((mother == 0 and child >= 1) or (mother == 2 and child == 0)):
                    errors += 1
                continue
            if p.sex != 2:
                continue
            low = (1 if father is not None and father >= 1 else 0) + (1 if mother == 2 else 0)
            high = (0 if father == 0 else 1) + (0 if mother == 0 else 1)
        else:
            low = (1 if father == 2 else 0) + (1 if mother == 2 else 0)
            high = (0 if father == 0 else 1) + (0 if mother == 0 else 1)
        if not low <= child <= high:
            errors += 1
    return errors


def _concordance(pedigree: Pedigree, v: Variant, mode: str) -> Optional[float]:
    fits = total = 0
    for pid, copies in v.genotypes.items():
        if copies is None or pid not in pedigree:
            continue
        person = pedigree[pid]
        if person.phenotype not in (1, 2):
            continue
        male = person.sex == 1 if person.sex in (1, 2) else None
        total += 1
        fits += _fits(mode, copies, male, person.affected)
    return fits / total if total else None


def _score_batch(pedigree, net, variants, scores, modes, models, pens, log_w, variant_frequency, min_frequency,
                 error_rate, log_without):
    V, F = len(variants), len(pens)
    q = np.array([_frequency(v, variant_frequency, min_frequency) for v in variants])
    evidence = []
    for i, (kind, person) in enumerate(zip(net.kinds, pedigree.people)):
        e = genotype_table(kind, error_rate)[call_codes(variants, person.id)]
        if net.parents[i] is None:
            e = e * founder_prior(kind, q)
        evidence.append(e)
    # P(genotypes), which doesn't depend on the mode except through the chromosome
    log_genotypes = net.log_likelihood(evidence)
    for mode in modes:
        model = models[mode]
        unary = []
        for i, (kind, person) in enumerate(zip(net.kinds, pedigree.people)):
            phen = np.stack([phenotype_factor(model.with_(penetrance=f), kind, person.phenotype) for f in pens])
            unary.append((phen[:, None, :] * evidence[i][None, :, :]).reshape(F * V, -1))
        joint = net.log_likelihood(unary).reshape(F, V)
        with_variant = _logsumexp(joint + log_w[:, None], axis=0) - log_genotypes
        for s, value in zip(scores, with_variant):
            s.log10_lr[mode] = float((value - log_without[mode]) / LN10)


@dataclass
class CombinedScore:
    """
    A candidate's evidence from several families with the same condition. log10_bf is the sum of its log10 Bayes
    factors over the families in which an affected person (the family's proband, if given) carries it, whose scores
    are in families. A family in which no affected person carries it is left out, since the condition may have
    another cause in that family. mode_probability is the probability, from all the families' phenotypes, that the
    mode is one that suits the candidate, and log10_odds is log10_bf + log10(mode_probability), as for VariantScore.
    """
    id: str
    log10_bf: float
    families: Dict[str, VariantScore]
    rank: Optional[float] = None
    log10_odds: Optional[float] = None
    mode_probability: Optional[float] = None

    @property
    def evidence(self) -> str:
        return evidence_strength(self.log10_bf)

    @property
    def compound(self) -> bool:
        return next(iter(self.families.values())).compound

    def row(self) -> dict:
        first = next(iter(self.families.values()))
        row = {'rank': self.rank, 'candidate': self.id, 'kind': 'compound heterozygote' if self.compound else 'variant',
               'gene': first.variant.gene, 'log10_bf': self.log10_bf, 'evidence': self.evidence,
               'log10_odds': self.log10_odds, 'mode_probability': self.mode_probability,
               'families': ','.join(self.families)}
        for name, s in self.families.items():
            row[f'log10_bf_{name}'] = s.log10_bf
        return row


def rank_variants_across(pedigrees, variants: Iterable[Variant], probands: Optional[Dict[str, str]] = None,
                         **options) -> List[CombinedScore]:
    """
    Ranks candidates over several families with the same condition, by the sum of their log10 Bayes factors over the
    families in which an affected person carries them (see CombinedScore).

    pedigrees: a dict of family to Pedigree, as read_ped gives, or a list of pedigrees
    variants: the candidates, whose genotypes can cover the people of every family
    probands: the proband of each family, if known (a dict of family to person)
    options: passed on to rank_variants
    """
    if not isinstance(pedigrees, dict):
        pedigrees = {p.family: p for p in pedigrees}
    variants = list(variants)
    probands = probands or {}
    modes = _modes(options.get('modes', MODES))
    priors = _prior(modes, options.get('prior'))
    # the phenotypes' likelihood under each mode, over all the families
    log_modes = {m: np.log(priors[m]) for m in modes if priors[m] > 0}
    found: Dict[str, Dict[str, VariantScore]] = {}
    for name, pedigree in pedigrees.items():
        proband = probands.get(name)
        logs = mode_log_likelihoods(pedigree, modes, options.get('penetrance', DEFAULT_PENETRANCE),
                                    options.get('penetrance_weights'), options.get('phenocopy', DEFAULT_PHENOCOPY),
                                    options.get('allele_frequency'), options.get('carrier_penetrance'),
                                    [proband] if proband else None)
        for m in log_modes:
            log_modes[m] += logs[m]
        who = [proband] if proband else pedigree.affected()
        for s in rank_variants(pedigree, variants, proband=proband, **options):
            if s.log10_bf is None:
                continue
            if s.partner is None and not any((s.variant.genotypes.get(p) or 0) >= 1 for p in who):
                continue
            found.setdefault(s.id, {})[name] = s
    combined = []
    log_all = _logsumexp(list(log_modes.values()))
    for key, by_family in found.items():
        c = CombinedScore(key, float(sum(s.log10_bf for s in by_family.values())), by_family)
        suited = [m for m in next(iter(by_family.values())).log10_lr if m in log_modes]
        log_suited = _logsumexp([log_modes[m] for m in suited])
        if np.isfinite(log_suited) and np.isfinite(log_all):
            c.mode_probability = float(np.exp(log_suited - log_all))
            c.log10_odds = c.log10_bf + float((log_suited - log_all) / LN10)
        combined.append(c)
    for c, r in zip(combined, fair_ranks([c.log10_odds for c in combined])):
        c.rank = r
    combined.sort(key=lambda c: (c.rank is None, c.rank if c.rank is not None else 0))
    return combined
