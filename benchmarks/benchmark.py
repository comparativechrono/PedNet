"""
Compares PedNet's ways of reading a pedigree graph on simulated families: for the mode of inheritance, the original
PedNet's network metric classifier, the pattern classifier and the likelihood of the graph as a Bayesian network;
for ranking candidate variants, the original network score, a plain concordance score and the Bayes factor.

    python benchmarks/benchmark.py [--families 200] [--seed 1] [--write benchmarks/RESULTS.md]

Each family is simulated with pednet.simulate: three generations, or in a quarter of the families of each mode four,
with a first-cousin marriage; at least two affected people, a penetrance drawn evenly between 0.6 and 1, a phenocopy
rate of 0.001, 5% of phenotypes unknown, each relative sequenced with probability 0.5 (the proband always), and 20
rare background variants that the proband carries, passed down the pedigree like real variants. Consanguinity is as
common whatever the mode, so that no method can take it alone as a sign of recessive inheritance, though it is in
fact commoner in families with a recessive condition. The likelihood methods don't know the penetrance simulated:
they average over their default range, as they would for a real family. The two classifiers were trained on other
families from the same simulator (benchmarks/train_classifiers.py, with another seed and over a wider range of
families), which favours them here, so the benchmark also tests all three on families unlike those they were trained
on.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import pednet  # noqa: E402
from pednet import infer_mode, rank_variants  # noqa: E402
from pednet.features import classify_by_metrics, classify_by_patterns, segregation_network_score  # noqa: E402
from pednet.segregation import _concordance, _pair_concordance, fair_ranks  # noqa: E402
from pednet.simulate import simulate_family  # noqa: E402

MODES = ('AD', 'AR', 'XLR', 'XLD')


def families(mode, n, rng, consanguineous_share=0.0, background=20):
    return [simulate_family(mode, rng, generations=3, consanguineous=rng.random() < consanguineous_share,
                            penetrance=float(rng.uniform(0.6, 1.0)), phenocopy=0.001, unknown_phenotype=0.05,
                            sequenced=0.5, background=background) for _ in range(n)]


def unlike_families(mode, n, rng):
    """Families unlike the classifiers' training families, and outside the likelihood methods' defaults: a lower
    penetrance, twice the most phenocopies they were trained with, more unknown phenotypes and larger families"""
    return [simulate_family(mode, rng, generations=int(rng.integers(2, 5)), children=(2, 6), partner=0.8,
                            penetrance=float(rng.uniform(0.3, 0.6)), phenocopy=0.01, unknown_phenotype=0.25,
                            background=0) for _ in range(n)]


def rank_of(values, target_index):
    return fair_ranks(values)[target_index]


def strictly_top(values, target_index):
    v = values[target_index]
    return v is not None and all(w is None or w < v for k, w in enumerate(values) if k != target_index)


def ceiling(f):
    """Whether the causal variant's genotypes in the sequenced people differ from every background variant's"""
    seq = sorted(f.sequenced)
    causal = tuple(f.variants[0].genotypes.get(p) for p in seq)
    return all(tuple(v.genotypes.get(p) for p in seq) != causal for v in f.variants[1:])


def pair_ceiling(f):
    """Whether the causal pair's genotypes in the sequenced people differ from every background pair's"""
    seq = sorted(f.sequenced)

    def calls(a, b):
        return tuple((a.genotypes.get(p), b.genotypes.get(p)) for p in seq)

    causal = calls(f.variants[0], f.variants[1])
    genes = {}
    for v in f.variants[2:]:
        genes.setdefault(v.gene, []).append(v)
    return all(calls(*pair) != causal for pair in genes.values() if len(pair) == 2)


def compound_families(n, rng):
    return [simulate_family('AR', rng, generations=3, penetrance=float(rng.uniform(0.6, 1.0)), phenocopy=0.001,
                            unknown_phenotype=0.05, sequenced=0.5, background=20, compound=True, background_pairs=5)
            for _ in range(n)]


def summarise_ranking(results):
    rows = []
    for method, ranks in results.items():
        ranks = [r for r in ranks if r is not None]
        top = np.mean([r[1] for r in ranks])
        mean_rank = np.mean([r[0] for r in ranks])
        rows.append((method, top, mean_rank, np.mean([1 / r[0] for r in ranks])))
    return rows


def two_way(probabilities, modes=('AD', 'AR')):
    """The probabilities of two modes alone, scaled to add up to 1"""
    total = sum(probabilities[m] for m in modes)
    return {m: probabilities[m] / total for m in modes}


def call_of(probabilities, threshold=0.8):
    best = max(probabilities, key=probabilities.get)
    return best if probabilities[best] >= threshold else 'uncertain'


def posteriors(pedigree, **options):
    return {m.mode: m.posterior for m in infer_mode(pedigree, **options).modes}


def mode_rows(fams, mode, told=None):
    """
    For each method, its calls and whether the true mode (or kind) was the most probable. told, if given, is what
    the likelihood is told about the condition (its penetrance and phenocopy rate), for rows of its own.
    """
    pat_p = [classify_by_patterns(f.pedigree).probabilities for f in fams]
    likelihoods = {'likelihood': {}}
    if told:
        likelihoods['likelihood told the penetrance'] = told
    four = {name: [posteriors(f.pedigree, **options) for f in fams] for name, options in likelihoods.items()}
    kind = 'dominant' if mode in ('AD', 'XLD') else 'recessive'
    methods = {}

    def add(name, ps):
        methods[name] = ([call_of(p) for p in ps], [max(p, key=p.get) == mode for p in ps])

    if mode in ('AD', 'AR'):
        methods['original metric classifier, AD or AR'] = ([classify_by_metrics(f.pedigree) for f in fams], None)
        add('patterns, AD or AR', [two_way(p) for p in pat_p])
        for name, options in likelihoods.items():
            add(f'{name}, AD or AR', [posteriors(f.pedigree, modes=('AD', 'AR'), **options) for f in fams])
    for name, ps in [('patterns', pat_p)] + list(four.items()):
        add(f'{name}, all four modes', ps)
    for name, ps in [('patterns', pat_p)] + list(four.items()):
        calls, top = [], []
        for p in ps:
            dominant = p.get('AD', 0) + p.get('XLD', 0)
            share = dominant if kind == 'dominant' else 1 - dominant
            calls.append(kind if share >= 0.8 else ('uncertain' if share > 0.2 else 'wrong'))
            top.append(share > 0.5)
        methods[f'{name}, dominant or recessive'] = (calls, top)
    calibration = {'likelihood': [(max(p.values()), max(p, key=p.get) == mode) for p in four['likelihood']],
                   'patterns': [(max(p.values()), max(p, key=p.get) == mode) for p in pat_p]}
    return methods, kind, calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--families', type=int, default=200, help='families per mode (default 200)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--write', help='write the results as Markdown to this file')
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    start = time.time()
    lines = [f'Simulated families: {args.families} for each mode; seed {args.seed}.', '']

    # ---- the mode of inheritance ------------------------------------------------------------------------------
    test = {m: families(m, args.families, rng, 0.25) for m in MODES}
    lines += ['## The mode of inheritance', '',
              'The share of families each method calls correctly, wrongly, or leaves uncertain (when no mode reaches '
              'a probability of 0.8), and the share in which the true mode is the most probable, certain or not. '
              'The original metric classifier tells AD from AR only, by a vote of its network metrics; the other '
              'two are shown choosing between AD and AR as it does, and between all four modes. The rows for '
              'dominant or recessive call the condition dominant (AD or XLD) or recessive (AR or XLR) from the four '
              'modes\' probabilities, which is often clear when the exact mode isn\'t.', '',
              '| Families | Method | Right | Wrong | Uncertain | Right when certain | True mode most probable |',
              '|---|---|---|---|---|---|---|']
    calibration = {'likelihood': [], 'patterns': []}

    def add_rows(fams, mode, table, told=None):
        methods, kind, calib = mode_rows(fams, mode, told)
        for key in calibration:
            calibration[key] += calib[key]
        for name, (calls, top) in methods.items():
            truth = kind if 'dominant or recessive' in name else mode
            right = np.mean([c == truth for c in calls])
            wrong = np.mean([c not in (truth, 'uncertain') for c in calls])
            unsure = np.mean([c == 'uncertain' for c in calls])
            certain = [c for c in calls if c != 'uncertain']
            when = np.mean([c == truth for c in certain]) if certain else float('nan')
            most = f'{np.mean(top):.2f}' if top is not None else '–'
            table.append(f'| {mode} | {name} | {right:.2f} | {wrong:.2f} | {unsure:.2f} | {when:.2f} | {most} |')

    for mode, fams in test.items():
        add_rows(fams, mode, lines)
    lines += ['', 'How well the probabilities are calibrated, over all the families above: the most probable mode\'s '
              'probability against how often it was the true mode.', '',
              '| Probability of the most probable mode | Families (likelihood) | True mode (likelihood) | '
              'Families (patterns) | True mode (patterns) |', '|---|---|---|---|---|']
    for low, high in ((0.25, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.99), (0.99, 1.0001)):
        cells = []
        for key in ('likelihood', 'patterns'):
            group = [hit for p, hit in calibration[key] if low <= p < high]
            cells += [str(len(group)), f'{np.mean(group):.2f}' if group else '–']
        lines.append(f'| {low:.2f}–{min(high, 1):.2f} | ' + ' | '.join(cells) + ' |')

    lines += ['', '### Families unlike the training families', '',
              'The same, for families of two to four generations unlike those the classifiers were trained on, '
              'and outside the likelihood methods\' default range of penetrances: a penetrance drawn between 0.3 '
              'and 0.6, a phenocopy rate of 0.01, a quarter of phenotypes unknown, two to six children to a '
              'couple, more of whom have families of their own, and no consanguinity. A real family differs from '
              'any simulation in ways like these, and this shows how each method copes. These families still come '
              'from the same generator as the training families, though: the condition always enters through the '
              'first couple, consanguinity has one pattern, and there are no half-siblings or people of unknown '
              'sex, so even these results probably flatter the classifiers. The rows for the '
              'likelihood told the penetrance give it what would be known about such a condition: penetrances of '
              '0.3, 0.45 and 0.6 to average over, and the phenocopy rate.', '',
              '| Families | Method | Right | Wrong | Uncertain | Right when certain | True mode most probable |',
              '|---|---|---|---|---|---|---|']
    for mode in MODES:
        add_rows(unlike_families(mode, args.families, rng), mode, lines,
                 told={'penetrance': (0.3, 0.45, 0.6), 'phenocopy': 0.01})

    # ---- ranking the candidate variants -------------------------------------------------------------------------
    lines += ['', '## Ranking the candidate variants', '',
              'Each family has the causal variant and 20 rare background variants carried by the proband. '
              '"Top" is the share of families in which the causal variant scored higher than every other variant '
              '(a tie for the top counts as a miss), and tied candidates share the average of their ranks. The '
              'original network score and the '
              'concordance score are given the true mode; PedNet\'s Bayes factor is shown not knowing it, and '
              'knowing it. The "identical" row is the share of families in which no background variant has exactly '
              'the causal variant\'s genotypes in the people sequenced. Otherwise the two can\'t be told apart by '
              'their genotypes, and only their allele frequencies, which here say nothing about which is causal, '
              'can separate them.', '',
              '| Families | Method | Top | Mean rank | Mean of 1/rank |', '|---|---|---|---|---|']
    for mode, fams in test.items():
        results = {'Bayes factor, mode unknown': [], 'Bayes factor, mode known': [], 'concordance, mode known': []}
        if mode in ('AD', 'AR'):
            results['original network score, mode known'] = []
        for f in fams:
            ids = [v.id for v in f.variants]
            for name, options in (('Bayes factor, mode unknown', {}), ('Bayes factor, mode known', {'modes': (mode,)})):
                order = {s.id: s.log10_odds for s in rank_variants(f.pedigree, f.variants, **options)}
                values = [order[i] for i in ids]
                results[name].append((rank_of(values, 0), strictly_top(values, 0)))
            values = [_concordance(f.pedigree, v, mode) for v in f.variants]
            results['concordance, mode known'].append((rank_of(values, 0), strictly_top(values, 0)))
            if mode in ('AD', 'AR'):
                values = [segregation_network_score(f.pedigree, v.genotypes, mode) for v in f.variants]
                results['original network score, mode known'].append((rank_of(values, 0), strictly_top(values, 0)))
        for name, top, mean_rank, reciprocal in summarise_ranking(results):
            lines.append(f'| {mode} | {name} | {top:.2f} | {mean_rank:.2f} | {reciprocal:.2f} |')
        lines.append(f'| {mode} | no background variant identical | {np.mean([ceiling(f) for f in fams]):.2f} | | |')

    # ---- compound heterozygotes ---------------------------------------------------------------------------------
    lines += ['', '## Compound heterozygotes', '',
              f'{args.families} families with an autosomal recessive condition caused by two variants in one gene, '
              'each brought into the family by a different founder, simulated as above (three generations, at least '
              'two affected people). Besides the two causal variants, each family has 20 rare background variants '
              'and 5 background pairs of rare variants in one gene that the proband carries both of, passed down '
              'the pedigree as copies of the gene, so that both variants of a pair can be on one copy. PedNet '
              'ranks the pairs as compound heterozygotes alongside the single variants; the causal pair is the '
              'target. Without pairs, only the single variants can be ranked, and the row for that gives the '
              'better rank of the two causal variants. The concordance score here counts a person as having the '
              'disease genotype when they carry both variants of a pair (or two copies of either).', '',
              '| Method | Top | Mean rank | Mean of 1/rank |', '|---|---|---|---|']
    results = {'Bayes factor, mode unknown': [], 'Bayes factor, mode known': [],
               'Bayes factor, single variants only': [], 'concordance, mode known': []}
    fams = compound_families(args.families, rng)
    target = 'causal a + causal b'
    for f in fams:
        for name, options in (('Bayes factor, mode unknown', {}), ('Bayes factor, mode known', {'modes': ('AR',)})):
            scores = rank_variants(f.pedigree, f.variants, **options)
            ids = [s.id for s in scores]
            values = [s.log10_odds for s in scores]
            results[name].append((rank_of(values, ids.index(target)), strictly_top(values, ids.index(target))))
            if not options:
                concordance = [_concordance(f.pedigree, s.variant, 'AR') if s.partner is None
                               else _pair_concordance(f.pedigree, s.variant, s.partner) for s in scores]
                results['concordance, mode known'].append((rank_of(concordance, ids.index(target)),
                                                           strictly_top(concordance, ids.index(target))))
        singles = rank_variants(f.pedigree, f.variants, compound=False)
        ids = [s.id for s in singles]
        values = [s.log10_odds for s in singles]
        best = min(rank_of(values, ids.index(c)) for c in f.causal)
        results['Bayes factor, single variants only'].append((best, any(strictly_top(values, ids.index(c))
                                                                        for c in f.causal)))
    for name, top, mean_rank, reciprocal in summarise_ranking(results):
        lines.append(f'| {name} | {top:.2f} | {mean_rank:.2f} | {reciprocal:.2f} |')
    lines.append(f'| no background pair identical | {np.mean([pair_ceiling(f) for f in fams]):.2f} | | |')

    lines += ['', f'Took {time.time() - start:.0f} seconds.']
    text = '\n'.join(lines)
    print(text)
    if args.write:
        summary, usage, details = __doc__.strip().split('\n\n')
        with open(args.write, 'w') as handle:
            handle.write(f'# Benchmark\n\n{summary}\n\n```bash\n{usage.strip()}\n```\n\n{details}\n\n'
                         f'PedNet {pednet.__version__}, with its defaults unless the table says otherwise. {text}\n')


if __name__ == '__main__':
    main()
