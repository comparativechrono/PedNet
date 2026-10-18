"""
Compares PedNet's likelihood methods with the original ones (the graph-metric classifier of the mode of inheritance
and the network segregation score, in legacy/) and with a plain concordance score, on simulated families.

    python benchmarks/benchmark.py [--families 200] [--seed 1] [--write benchmarks/RESULTS.md]

Each family is simulated with pednet.simulate: three generations (four for the consanguineous ones), at least two
affected people, a penetrance drawn evenly between 0.6 and 1, a phenocopy rate of 0.001, 5% of phenotypes unknown,
each relative sequenced with probability 0.5 (the proband always), and 20 rare background variants that the proband
carries, passed down the pedigree like real variants. The penetrance PedNet assumes isn't the one simulated: it
averages over its default range, as it would for a real family. The original methods need the packages in
legacy/requirements.txt.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'legacy'))
warnings.filterwarnings('ignore')

import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve, auc  # noqa: E402

import pednet  # noqa: E402
from pednet import infer_mode, rank_variants  # noqa: E402
from pednet.segregation import _concordance, _pair_concordance, fair_ranks  # noqa: E402
from pednet.simulate import simulate_family  # noqa: E402

import PedigreeDAGAnalysis as legacy_dag  # noqa: E402
from ModeClassifierMain import MoI_classification, youden_threshold  # noqa: E402
from SegScoreMain import raw_categorical_scoring, segregation_network_score  # noqa: E402


def families(mode, n, rng, consanguineous_share=0.0):
    out = []
    for i in range(n):
        out.append(simulate_family(mode, rng, generations=3, consanguineous=rng.random() < consanguineous_share,
                                   penetrance=float(rng.uniform(0.6, 1.0)), phenocopy=0.001, unknown_phenotype=0.05,
                                   sequenced=0.5, background=20))
    return out


def legacy_graph(pedigree):
    rows = [{'FamilyID': pedigree.family, 'IndividualID': p.id, 'PaternalID': p.father or '0',
             'MaternalID': p.mother or '0', 'Sex': p.sex, 'Phenotype': p.phenotype} for p in pedigree.people]
    return legacy_dag.construct_pedigree_graph(pd.DataFrame(rows))


def train_legacy_classifier(training):
    """Thresholds for the original classifier, found as its own training finds them, on simulated AD and AR families"""
    metrics = pd.DataFrame([dict(legacy_dag.calc_pedigree_metrics(legacy_graph(f.pedigree)), true_mode=f.mode)
                            for f in training])
    thresholds = {}
    for metric in [c for c in metrics.columns if c != 'true_mode']:
        values = metrics[metric].values
        ad, ar = values[metrics.true_mode == 'AD'].mean(), values[metrics.true_mode == 'AR'].mean()
        positive = 'AD' if ad > ar else 'AR'
        fpr, tpr, thr = roc_curve(metrics.true_mode, values, pos_label=positive)
        area = auc(fpr, tpr)
        if area >= 0.7:
            thresholds[metric] = {'threshold': youden_threshold(fpr, tpr, thr),
                                  'direction': f'HIGH->{positive}', 'auc': area}
    return thresholds


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
    training = families('AD', args.families, rng) + families('AR', args.families, rng, 0.3)
    thresholds = train_legacy_classifier(training)
    test = {m: families(m, args.families, rng, 0.3 if m == 'AR' else 0.0) for m in ('AD', 'AR', 'XLR', 'XLD')}

    lines += ['## The mode of inheritance', '',
              'The share of families each method calls correctly, wrongly, or leaves uncertain. The original '
              'classifier tells AD from AR only, and was trained on another '
              f'{2 * args.families} simulated AD and AR families. PedNet is shown choosing between AD and AR, as the '
              'original classifier does, and between all four modes. The last row for each mode is whether PedNet '
              'calls the condition dominant (AD or XLD) or recessive (AR or XLR), from the four modes\' '
              'probabilities: often clear when the exact mode isn\'t. In the last column, the true mode (or kind) '
              'is the most probable one, whether or not the call is certain.', '',
              '| Families | Method | Right | Wrong | Uncertain | Right when certain | True mode most probable |',
              '|---|---|---|---|---|---|---|']
    calibration = []
    for mode, fams in test.items():
        methods = {}
        if mode in ('AD', 'AR'):
            calls = []
            for f in fams:
                call = MoI_classification(legacy_graph(f.pedigree), thresholds)
                calls.append('uncertain' if call == 'Uncertain' else call)
            methods['original classifier'] = (calls, None)
            two = [infer_mode(f.pedigree, modes=('AD', 'AR')) for f in fams]
            methods['PedNet, AD or AR'] = ([r.call for r in two], [r.modes[0].mode == mode for r in two])
        four = [infer_mode(f.pedigree) for f in fams]
        methods['PedNet, all four modes'] = ([r.call for r in four], [r.modes[0].mode == mode for r in four])
        kind = 'dominant' if mode in ('AD', 'XLD') else 'recessive'
        grouped = []
        for r in four:
            p = r.dominant if kind == 'dominant' else r.recessive
            grouped.append(kind if p >= 0.8 else ('uncertain' if p > 0.2 else 'wrong'))
        methods['PedNet, dominant or recessive'] = (grouped, [(r.dominant > 0.5) == (kind == 'dominant') for r in four])
        calibration += [(r.modes[0].posterior, r.modes[0].mode == mode) for r in four]
        for name, (calls, top) in methods.items():
            truth = kind if name == 'PedNet, dominant or recessive' else mode
            right = np.mean([c == truth for c in calls])
            wrong = np.mean([c not in (truth, 'uncertain') for c in calls])
            unsure = np.mean([c == 'uncertain' for c in calls])
            certain = [c for c in calls if c != 'uncertain']
            when = np.mean([c == truth for c in certain]) if certain else float('nan')
            most = f'{np.mean(top):.2f}' if top is not None else '–'
            lines.append(f'| {mode} | {name} | {right:.2f} | {wrong:.2f} | {unsure:.2f} | {when:.2f} | {most} |')
    lines += ['', 'How well PedNet\'s probabilities are calibrated (all four modes, every test family): the most '
              'probable mode\'s posterior probability against how often it was the true mode.', '',
              '| Posterior of the most probable mode | Families | True mode |', '|---|---|---|']
    for low, high in ((0.25, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.99), (0.99, 1.0001)):
        group = [hit for p, hit in calibration if low <= p < high]
        if group:
            lines.append(f'| {low:.2f}–{min(high, 1):.2f} | {len(group)} | {np.mean(group):.2f} |')

    # ---- ranking the candidate variants -------------------------------------------------------------------------
    lines += ['', '## Ranking the candidate variants', '',
              'Each family has the causal variant and 20 rare background variants carried by the proband. '
              '"Top" is the share of families in which the causal variant scored higher than every other variant '
              '(a tie for the top counts as a miss); the ranks share ties out. The original network score and the '
              'concordance score are given the true mode; PedNet is shown not knowing it, and knowing it. The '
              '"identical" row is the share of families in which no background variant has exactly the causal '
              'variant\'s genotypes in the people sequenced. Otherwise the two can\'t be told apart by their '
              'genotypes, and only their allele frequencies, which here say nothing about which is causal, can '
              'separate them.', '',
              '| Families | Method | Top | Mean rank | Mean of 1/rank |', '|---|---|---|---|---|']
    for mode, fams in test.items():
        results = {'PedNet, mode unknown': [], 'PedNet, mode known': [], 'concordance, mode known': []}
        if mode in ('AD', 'AR'):
            results['original network score, mode known'] = []
        for f in fams:
            ids = [v.id for v in f.variants]
            unknown = rank_variants(f.pedigree, f.variants)
            order = {s.id: s.log10_odds for s in unknown}
            values = [order[i] for i in ids]
            results['PedNet, mode unknown'].append((rank_of(values, 0), strictly_top(values, 0)))
            known = rank_variants(f.pedigree, f.variants, modes=(mode,))
            order = {s.id: s.log10_odds for s in known}
            values = [order[i] for i in ids]
            results['PedNet, mode known'].append((rank_of(values, 0), strictly_top(values, 0)))
            values = [_concordance(f.pedigree, v, mode) for v in f.variants]
            results['concordance, mode known'].append((rank_of(values, 0), strictly_top(values, 0)))
            if mode in ('AD', 'AR'):
                G = legacy_graph(f.pedigree)
                values = []
                for v in f.variants:
                    gt = {p: g for p, g in v.genotypes.items() if g is not None and p in G}
                    values.append(segregation_network_score(G, gt, mode, 'Original', raw_categorical_scoring(G, gt))
                                  if gt else None)
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
    results = {'PedNet, mode unknown': [], 'PedNet, mode known': [], 'PedNet, single variants only': [],
               'concordance, mode known': []}
    fams = compound_families(args.families, rng)
    target = 'causal a + causal b'
    for f in fams:
        for name, options in (('PedNet, mode unknown', {}), ('PedNet, mode known', {'modes': ('AR',)})):
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
        results['PedNet, single variants only'].append((best, any(strictly_top(values, ids.index(c))
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
