import numpy as np
import pytest

from pednet import Variant, evidence_strength, pedigree_from_rows, rank_variants, rank_variants_across
from pednet.engine import Network
from pednet.models import Model, founder_prior, genotype_factor, phenotype_factor
from pednet.segregation import candidate_pairs, de_novo, fair_ranks, mendelian_errors

# four affected brothers with parents of unknown phenotype: segregatr's documented example
BROTHERS = [('1', 0, 0, 1, 0), ('2', 0, 0, 2, 0)] + [(str(i), '1', '2', 1, 2) for i in range(3, 7)]
# an affected father, his unaffected wife and four children, two affected
NUCLEAR = [('1', 0, 0, 1, 2), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2), ('4', '1', '2', 2, 2), ('5', '1', '2', 1, 1),
           ('6', '1', '2', 2, 1)]
# three generations of a dominant condition
FAMILY = [('1', 0, 0, 1, 2), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2), ('4', '1', '2', 2, 1), ('5', 0, 0, 2, 1),
          ('6', '3', '5', 1, 2), ('7', '3', '5', 2, 1), ('8', '3', '5', 2, 2)]


def variant(name, calls, chrom='1', af=1e-4, gene=None):
    return Variant(name, dict(calls), chrom, allele_frequency=af, gene=gene)


def test_matches_segregatr():
    # segregatr: FLB(nuclearPed(4), carriers = 3:6, aff = 3:6, unknown = 1:2, freq = 0.0001,
    # penetrances = c(0.01, 0.9, 0.9), proband = 3) gives 7.732161, conditioning on the proband's genotype call
    ped = pedigree_from_rows(BROTHERS)
    net = Network(ped, False)
    model = Model('AD', 1e-4, 0.9, 0.01)
    calls = {str(i): 1 for i in range(3, 7)}

    def log_l(phenotypes, genotypes, proband_states=None):
        unary = []
        for i, (kind, person) in enumerate(zip(net.kinds, ped.people)):
            u = founder_prior(kind, 1e-4) if net.parents[i] is None else np.ones(3)
            if phenotypes:
                u = u * phenotype_factor(model, kind, person.phenotype)
            if genotypes:
                u = u * genotype_factor(kind, calls.get(person.id), 0.0)
            if proband_states is not None and person.id == '3':
                u = u * proband_states
            unary.append(u)
        return net.log_likelihood(unary)[0]

    heterozygous = np.array([0.0, 1.0, 0.0])
    flb = np.exp(log_l(True, True) - log_l(False, True) - log_l(True, False, heterozygous)
                 + log_l(False, False, heterozygous))
    assert flb == pytest.approx(7.732161, abs=5e-7)

    # PedNet conditions on the proband having the disease genotype, one copy or two, which differs very little
    score = rank_variants(ped, [variant('v', calls)], modes=('AD',), penetrance=0.9, phenocopy=0.01,
                          allele_frequency={'AD': 1e-4}, min_frequency=1e-4, error_rate=0.0, proband='3')[0]
    assert 10 ** score.log10_bf == pytest.approx(7.732161, rel=2e-4)


def test_informative_meioses():
    # with full penetrance and no phenocopies, a variant that follows the condition through four meioses (from the
    # father to each child, the proband's counted out) has a Bayes factor of 2 ** 4
    ped = pedigree_from_rows(NUCLEAR)
    v = variant('v', {'1': 1, '2': 0, '3': 1, '4': 1, '5': 0, '6': 0}, af=1e-7)
    score = rank_variants(ped, [v], modes=('AD',), penetrance=1.0, phenocopy=0.0, allele_frequency={'AD': 1e-7},
                          min_frequency=1e-7, error_rate=0.0, proband='3')[0]
    assert 10 ** score.log10_bf == pytest.approx(16, rel=1e-4)
    assert score.evidence == 'PP1_Moderate'


def test_ranking():
    ped = pedigree_from_rows(FAMILY)
    variants = [
        variant('everyone', {p[0]: 1 for p in FAMILY}),
        variant('causal', {'1': 1, '2': 0, '3': 1, '4': 0, '5': 0, '6': 1, '7': 0, '8': 1}),
        variant('mother', {'1': 0, '2': 1, '3': 1, '4': 1, '5': 0, '6': 1, '7': 0, '8': 1}),
        variant('proband only', {'6': 1}),
        variant('nobody typed', {}),
    ]
    scores = rank_variants(ped, variants)
    assert [s.id for s in scores] == ['causal', 'proband only', 'everyone', 'mother', 'nobody typed']
    # genotypes that say nothing give a Bayes factor near 1: here only the proband's, which could be a calling error
    assert -0.05 < scores[1].log10_bf < 0 and scores[1].evidence == ''
    exact = rank_variants(ped, [variants[3]], modes=('AD',), error_rate=0, min_frequency=1e-4, proband='6')[0]
    assert exact.log10_bf == pytest.approx(0, abs=1e-4)
    best = scores[0]
    assert best.rank == 1 and best.best_mode == 'AD' and best.log10_bf > 1
    assert best.affected_carriers == 4 and best.unaffected_carriers == 0 and best.concordance == 1
    assert set(best.log10_lr) == {'AD', 'AR'}
    last = scores[-1]
    assert last.log10_bf is None and last.rank is None and 'genotyped' in last.note
    row = best.row()
    assert row['candidate'] == 'causal' and row['log10_lr_XLR'] is None and row['evidence'] == best.evidence


def test_ties_share_ranks():
    ped = pedigree_from_rows(FAMILY)
    calls = {'1': 1, '2': 0, '3': 1, '6': 1}
    scores = rank_variants(ped, [variant('a', calls), variant('b', calls), variant('c', {'6': 1, '3': 0})])
    assert sorted(s.rank for s in scores) == [1.5, 1.5, 3]
    assert fair_ranks([3.0, 1.0, 3.0 + 1e-12, None]) == [1.5, 3.0, 1.5, None]


def test_chromosomes_choose_the_modes():
    ped = pedigree_from_rows(FAMILY)
    calls = {'1': 1, '3': 1, '6': 1}
    scores = {s.id: s for s in rank_variants(ped, [variant('auto', calls), variant('x', calls, chrom='chrX'),
                                                    variant('unknown', calls, chrom=None)])}
    assert set(scores['auto'].log10_lr) == {'AD', 'AR'}
    assert set(scores['x'].log10_lr) == {'XLR', 'XLD'}
    assert set(scores['unknown'].log10_lr) == {'AD', 'AR', 'XLR', 'XLD'}
    only = rank_variants(ped, [variant('x', calls, chrom='X')], modes=('AD',))[0]
    assert only.log10_bf is None and 'suits' in only.note


def test_bayes_factor_and_odds():
    ped = pedigree_from_rows(FAMILY)
    v = variant('causal', {'1': 1, '2': 0, '3': 1, '4': 0, '5': 0, '6': 1, '7': 0, '8': 1})
    unknown = rank_variants(ped, [v])[0]
    known = rank_variants(ped, [v], modes=('AD',))[0]
    # the family is clearly dominant, so the Bayes factor with the mode unknown is nearly that for AD
    assert unknown.log10_bf == pytest.approx(known.log10_bf, abs=0.01)
    assert unknown.log10_odds == pytest.approx(unknown.log10_bf + np.log10(unknown.mode_probability))
    assert known.mode_probability == 1 and known.log10_odds == known.log10_bf
    # on the X, the same genotypes break the rules of inheritance (father to son), and the odds are lower still
    x = rank_variants(ped, [variant('x', v.genotypes, chrom='X')])[0]
    assert x.log10_odds < x.log10_bf < unknown.log10_bf


def test_genotype_errors_are_allowed_for():
    # one miscalled relative weakens the evidence but doesn't rule the variant out
    ped = pedigree_from_rows(FAMILY)
    clean = {'1': 1, '2': 0, '3': 1, '4': 0, '5': 0, '6': 1, '7': 0, '8': 1}
    wrong = dict(clean, **{'8': 0})
    scores = {s.id: s for s in rank_variants(ped, [variant('clean', clean), variant('wrong', wrong)])}
    assert scores['clean'].log10_bf > scores['wrong'].log10_bf > -6
    assert np.isfinite(rank_variants(ped, [variant('wrong', wrong)], error_rate=0)[0].log10_bf)


def test_evidence_strength():
    assert evidence_strength(None) == ''
    assert evidence_strength(np.log10(2.08)) == 'PP1_Supporting'
    assert evidence_strength(np.log10(2.0)) == ''
    assert evidence_strength(np.log10(4.33)) == 'PP1_Moderate'
    assert evidence_strength(np.log10(18.7)) == 'PP1_Strong'
    assert evidence_strength(5.0) == 'PP1_Strong'
    assert evidence_strength(-np.log10(2.08)) == 'BS4_Supporting'
    assert evidence_strength(-np.log10(4.33)) == 'BS4_Moderate'
    assert evidence_strength(-3.0) == 'BS4_Strong'


def test_mendelian_errors_and_de_novo():
    ped = pedigree_from_rows(FAMILY)
    assert mendelian_errors(ped, variant('ok', {'1': 1, '2': 0, '3': 1})) == 0
    assert mendelian_errors(ped, variant('hom', {'1': 1, '2': 0, '3': 2})) == 1
    assert mendelian_errors(ped, variant('new', {'1': 0, '2': 0, '3': 1})) == 1
    assert de_novo(ped, variant('new', {'1': 0, '2': 0, '3': 1, '6': 1})) == ['3']
    assert de_novo(ped, variant('one parent typed', {'1': 0, '3': 1})) == []
    # on the X a son has his mother's X only, and a daughter her father's
    assert mendelian_errors(ped, variant('x son', {'1': 1, '2': 0, '3': 1}, chrom='X')) == 1
    assert mendelian_errors(ped, variant('x daughter', {'1': 1, '2': 0, '4': 0}, chrom='X')) == 1
    assert de_novo(ped, variant('x son', {'1': 1, '2': 0, '3': 1}, chrom='X')) == ['3']
    score = rank_variants(ped, [variant('new', {'1': 0, '2': 0, '3': 1, '6': 1})])[0]
    assert score.de_novo == ['3'] and 'possibly de novo in 3' in score.notes


# two affected children of unaffected parents, each parent carrying one variant of the same gene
SIBS = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2), ('4', '1', '2', 2, 2), ('5', '1', '2', 1, 1),
        ('6', '1', '2', 2, 1)]


def compound_variants(gene='G'):
    return [variant('a', {'1': 1, '2': 0, '3': 1, '4': 1, '5': 1, '6': 0}, gene=gene),
            variant('b', {'1': 0, '2': 1, '3': 1, '4': 1, '5': 0, '6': 1}, gene=gene),
            # c and d: both from the father, on one copy of the gene, and carried by an unaffected son as well
            variant('c', {'1': 1, '2': 0, '3': 1, '4': 1, '5': 1, '6': 0}, gene='H'),
            variant('d', {'1': 1, '2': 0, '3': 1, '4': 1, '5': 1, '6': 0}, gene='H')]


def test_compound_heterozygotes():
    ped = pedigree_from_rows(SIBS)
    scores = rank_variants(ped, compound_variants())
    by_id = {s.id: s for s in scores}
    pair = by_id['a + b']
    assert scores[0] is pair and pair.compound
    assert pair.best_mode == 'AR' and set(pair.log10_lr) == {'AR'}
    assert pair.affected_carriers == 2 and pair.unaffected_carriers == 0 and pair.concordance == 1
    assert pair.log10_bf > max(by_id['a'].log10_bf, by_id['b'].log10_bf) + 1
    # in cis, from one parent, the pair explains nothing
    assert by_id['c + d'].log10_bf < 0
    row = pair.row()
    assert row['kind'] == 'compound heterozygote' and row['gene'] == 'G'


def test_compound_options():
    ped = pedigree_from_rows(SIBS)
    assert not any(s.compound for s in rank_variants(ped, compound_variants(), compound=False))
    assert not any(s.compound for s in rank_variants(ped, compound_variants(), modes=('AD',)))
    assert not any(s.compound for s in rank_variants(ped, compound_variants(gene=None)[:2]))
    assert candidate_pairs(ped, compound_variants()) == [(0, 1), (2, 3)]
    # only pairs that the proband, if given, carries both of
    assert candidate_pairs(ped, compound_variants(), proband='6') == []


def test_compound_on_the_x():
    # an affected woman with a variant from each parent; on the X a man has one copy, so a pair needs a woman
    rows = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 2, 2), ('4', '1', '2', 1, 1)]
    ped = pedigree_from_rows(rows)
    a = variant('a', {'1': 1, '2': 0, '3': 1, '4': 0}, chrom='X', gene='G')
    b = variant('b', {'1': 0, '2': 1, '3': 1, '4': 0}, chrom='X', gene='G')
    pair = next(s for s in rank_variants(ped, [a, b]) if s.compound)
    assert set(pair.log10_lr) == {'XLR'}
    men = pedigree_from_rows([('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2)])
    assert candidate_pairs(men, [a, b]) == []


def test_ranking_across_families():
    one = pedigree_from_rows(FAMILY, 'one')
    two = pedigree_from_rows([(f'b{pid}', f'b{f}' if f else 0, f'b{m}' if m else 0, s, p) for pid, f, m, s, p in FAMILY],
                             'two')
    shared = {'1': 1, '2': 0, '3': 1, '4': 0, '5': 0, '6': 1, '7': 0, '8': 1}
    calls = dict(shared, **{f'b{k}': v for k, v in shared.items()})
    private = dict({k: 0 for k in calls}, **{'b1': 1, 'b3': 1, 'b6': 1, 'b8': 1})
    combined = {c.id: c for c in rank_variants_across({'one': one, 'two': two},
                                                      [variant('shared', calls), variant('private', private)])}
    alone = rank_variants(one, [variant('shared', calls)])[0]
    assert set(combined['shared'].families) == {'one', 'two'}
    # with the mode shared, the families' likelihood ratios multiply mode by mode
    assert combined['shared'].log10_bf == pytest.approx(2 * alone.log10_bf, abs=1e-3)
    known = rank_variants_across({'one': one, 'two': two}, [variant('shared', calls)], modes=('AD',))[0]
    assert known.log10_bf == pytest.approx(2 * rank_variants(one, [variant('shared', calls)], modes=('AD',))[0].log10_bf)
    # the private variant counts in the family where an affected person carries it, not in the other
    assert set(combined['private'].families) == {'two'}
    assert combined['shared'].rank == 1
    assert combined['shared'].row()['log10_bf_one'] == pytest.approx(alone.log10_bf)


def test_no_nan_when_a_mode_is_impossible():
    # with full penetrance and no phenocopies, a dominant mode can't explain an affected child of unaffected parents
    ped = pedigree_from_rows(SIBS)
    scores = rank_variants(ped, compound_variants(), penetrance=1.0, phenocopy=0.0)
    for s in scores:
        assert s.log10_bf is None or s.log10_bf == s.log10_bf
    assert scores[0].id == 'a + b'
    trio = pedigree_from_rows([('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2)])
    for s in rank_variants(trio, [variant('v', {'1': 1, '2': 0, '3': 1})], penetrance=1.0):
        assert s.log10_bf == s.log10_bf and s.rank == 1


def test_an_unaffected_proband_is_refused():
    with pytest.raises(ValueError, match='not recorded as affected'):
        rank_variants(pedigree_from_rows(SIBS), compound_variants(), proband='5')


def test_y_and_mitochondrial_variants_are_left_out():
    ped = pedigree_from_rows(FAMILY)
    calls = {'1': 1, '3': 1, '6': 1}
    for chrom in ('Y', 'chrY', 'MT', 'chrM'):
        s = rank_variants(ped, [variant('v', calls, chrom=chrom)])[0]
        assert s.log10_bf is None and 'mitochondrial' in s.note and s.rank is None


def test_evidence_withheld_for_new_variants_and_halves_of_pairs():
    trio = pedigree_from_rows([('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2)])
    new = rank_variants(trio, [variant('new', {'1': 0, '2': 0, '3': 1})], modes=('AD',))[0]
    assert new.evidence == '' and 'PS2' in new.notes and new.log10_bf < 0
    by_id = {s.id: s for s in rank_variants(pedigree_from_rows(SIBS), compound_variants(), modes=('AR',))}
    assert by_id['a + b'].evidence.startswith('PP1')
    assert by_id['a'].evidence == '' and 'a + b' in by_id['a'].notes
    assert by_id['c + d'].log10_bf < 0 and by_id['c'].evidence.startswith('BS4')


def test_families_share_the_mode():
    # family one is clearly dominant; two is a pair of affected brothers with unaffected parents, unclear alone
    one = pedigree_from_rows(FAMILY, 'one')
    two = pedigree_from_rows([('b1', 0, 0, 1, 1), ('b2', 0, 0, 2, 1), ('b3', 'b1', 'b2', 1, 2), ('b4', 'b1', 'b2', 1, 2)],
                             'two')
    calls = {'1': 1, '2': 0, '3': 1, '4': 0, '5': 0, '6': 1, '7': 0, '8': 1, 'b1': 0, 'b2': 1, 'b3': 1, 'b4': 1}
    together = rank_variants_across({'one': one, 'two': two}, [variant('v', calls)])[0]
    known = rank_variants_across({'one': one, 'two': two}, [variant('v', calls)], modes=('AD',))[0]
    # the families together are clearly dominant, so the mode unknown comes to nearly the same as AD
    assert together.log10_bf == pytest.approx(known.log10_bf, abs=0.01)


def test_pairs_left_out_when_their_locus_is_too_tangled(monkeypatch):
    from pednet import engine
    monkeypatch.setattr(engine, 'MAX_STATES', 500)
    with pytest.warns(UserWarning, match='Pairs of variants in one gene are left out'):
        scores = rank_variants(pedigree_from_rows(SIBS), compound_variants())
    assert scores and not any(s.compound for s in scores)
