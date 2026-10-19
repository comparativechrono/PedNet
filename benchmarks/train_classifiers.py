"""
Trains the two graph classifiers of the mode of inheritance on simulated families, and writes what they learn to
src/pednet/_trained.py, which PedNet uses by default:

- the original PedNet's metric classifier: a threshold on each network metric, by Youden's J, from AD and AR families
- the pattern classifier: a multinomial logistic regression on the pattern features, from families of all four modes

    python benchmarks/train_classifiers.py [--families 1000] [--seed 2026]

The families are simulated with pednet.simulate, over a wider range than the benchmark's (benchmarks/benchmark.py):
two to five generations, one to five children to a couple, a penetrance drawn evenly between 0.4 and 1, a phenocopy
rate between 0.001 and 0.005, up to 15% of phenotypes unknown, and a quarter of the families of each mode those of
a first-cousin marriage. The seed differs from the benchmark's, so that the classifiers are tested there on families
they weren't trained on.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from pednet.features import PATTERN_FEATURES, train_metric_classifier, train_pattern_classifier  # noqa: E402
from pednet.models import MODES  # noqa: E402
from pednet.simulate import simulate_family  # noqa: E402


def families(mode, n, rng):
    out = []
    for _ in range(n):
        # a first-cousin marriage in a quarter of the families of every mode, not only the recessive ones, so that
        # the classifier learns what the pattern of affected people in such a family says rather than taking the
        # marriage itself as proof of recessive inheritance
        consanguineous = rng.random() < 0.25
        f = simulate_family(mode, rng, generations=int(rng.integers(2, 6)), children=(1, 5),
                            consanguineous=consanguineous, penetrance=float(rng.uniform(0.4, 1.0)),
                            phenocopy=float(rng.uniform(0.001, 0.005)), unknown_phenotype=float(rng.uniform(0, 0.15)),
                            background=0)
        out.append((f.pedigree, mode))
    return out


def number(x):
    return repr(float(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--families', type=int, default=1000, help='families of each mode (default 1000)')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out', default=os.path.join(ROOT, 'src', 'pednet', '_trained.py'))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    training = {mode: families(mode, args.families, rng) for mode in MODES}
    thresholds = train_metric_classifier(training['AD'] + training['AR'])
    patterns = train_pattern_classifier([f for mode in MODES for f in training[mode]])

    lines = ['"""',
             'What the graph classifiers learnt from simulated families: written by benchmarks/train_classifiers.py',
             f'(--families {args.families} --seed {args.seed}), so not to be edited by hand.',
             '"""',
             'import numpy as np',
             '',
             '# the original PedNet\'s metric classifier: for each metric kept, (threshold, the mode above it, AUC)',
             'METRIC_THRESHOLDS = {']
    for metric, t in thresholds.items():
        lines.append(f'    {metric!r}: ({number(t.threshold)}, {t.high!r}, {number(t.auc)}),')
    lines += ['}', '',
              '# the pattern classifier: a multinomial logistic regression on the standardised pattern features',
              f'PATTERN_MODES = {tuple(patterns.modes)!r}',
              f'PATTERN_FEATURES = {tuple(PATTERN_FEATURES)!r}',
              'PATTERN_MEAN = [' + ', '.join(number(v) for v in patterns.mean) + ']',
              'PATTERN_SCALE = [' + ', '.join(number(v) for v in patterns.scale) + ']',
              'PATTERN_WEIGHTS = [']
    for row in patterns.weights:
        lines.append('    [' + ', '.join(number(v) for v in row) + '],')
    lines += [']', '', '',
              'def pattern_classifier():',
              '    from .features import PatternClassifier',
              '    return PatternClassifier(PATTERN_MODES, PATTERN_FEATURES, np.array(PATTERN_MEAN), np.array(PATTERN_SCALE),',
              '                             np.array(PATTERN_WEIGHTS))', '']
    with open(args.out, 'w') as handle:
        handle.write('\n'.join(lines))
    print(f'Wrote {args.out}')
    for metric, t in thresholds.items():
        print(f'  {metric}: {t.threshold:.4f}, {t.high} above, AUC {t.auc:.3f}')


if __name__ == '__main__':
    main()
