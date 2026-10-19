import importlib

import numpy as np
import pytest

from pednet import pedigree_from_rows
from pednet.features import (METRICS, MetricThreshold, affected_child_of_unaffected_parents, classify_by_metrics,
                             classify_by_patterns, network_metrics, network_parts, pattern_features,
                             segregation_network_score, train_metric_classifier, train_pattern_classifier)
from pednet.simulate import simulate_family

# three generations of a dominant condition, with father-to-son transmission
DOMINANT = [('1', 0, 0, 1, 2), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2), ('4', '1', '2', 2, 1), ('5', '1', '2', 2, 2),
            ('6', 0, 0, 2, 1), ('7', '3', '6', 1, 2), ('8', '3', '6', 2, 1), ('9', '3', '6', 1, 2)]
# affected men related through unaffected women
X_RECESSIVE = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2), ('4', '1', '2', 2, 1),
               ('5', '1', '2', 2, 1), ('6', 0, 0, 1, 1), ('7', '6', '4', 1, 2), ('8', '6', '4', 2, 1),
               ('9', 0, 0, 1, 1), ('10', '9', '5', 1, 2), ('11', '9', '5', 1, 1)]
# two affected children of first cousins
COUSINS = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 1), ('4', '1', '2', 2, 1), ('5', 0, 0, 2, 1),
           ('6', 0, 0, 1, 1), ('7', '3', '5', 1, 1), ('8', '6', '4', 2, 1), ('9', '7', '8', 1, 2),
           ('10', '7', '8', 2, 2), ('11', '7', '8', 1, 1)]


def test_network_metrics_by_hand():
    m = network_metrics(pedigree_from_rows(DOMINANT))
    # affected: 1, 3, 5, 7, 9; with an affected parent: 3, 5, 7, 9
    assert m['ratio_aff_parent'] == pytest.approx(4 / 5)
    # siblings of affected people: 3 has 4 and 5, 5 has 3 and 4, 7 has 8 and 9, 9 has 7 and 8; 4 of 8 affected
    assert m['sibling_aff_ratio'] == pytest.approx(4 / 8)
    assert m['gen_cov'] == 1
    assert m['avg_bet_unaff'] >= 0
    assert set(m) == set(METRICS)


def test_affected_child_of_unaffected_parents():
    assert affected_child_of_unaffected_parents(pedigree_from_rows(COUSINS))
    assert not affected_child_of_unaffected_parents(pedigree_from_rows(DOMINANT))


def test_metric_classifier():
    assert classify_by_metrics(pedigree_from_rows(DOMINANT)) == 'AD'
    assert classify_by_metrics(pedigree_from_rows(COUSINS)) == 'AR'
    thresholds = {'gen_cov': MetricThreshold(0.5, 'AD', 0.9)}
    assert classify_by_metrics(pedigree_from_rows(COUSINS), thresholds) == 'AR'


def simulated(modes=('AD', 'AR'), n=12, seed=4, **options):
    rng = np.random.default_rng(seed)
    options.setdefault('background', 3)
    return [(simulate_family(mode, rng, **options), mode) for mode in modes for _ in range(n)]


def test_training_the_metric_classifier():
    families = [(f.pedigree, mode) for f, mode in simulated(n=40)]
    thresholds = train_metric_classifier(families)
    assert thresholds and all(t.high in ('AD', 'AR') and t.auc >= 0.7 for t in thresholds.values())
    with pytest.raises(ValueError):
        train_metric_classifier([(p, 'AD') for p, _ in families[:3]])


def test_pattern_features_by_hand():
    d = pattern_features(pedigree_from_rows(DOMINANT))
    assert d['male_to_male'] == 1 and d['consanguineous'] == 0
    # affected with parents: 3, 5, 7, 9, all with an affected parent
    assert d['vertical'] == pytest.approx((4 + 0.5) / 5) and d['skipped'] == pytest.approx(0.5 / 5)
    x = pattern_features(pedigree_from_rows(X_RECESSIVE))
    assert x['male_to_male'] == 0 and x['male_share'] == pytest.approx((3 + 0.5) / 4)
    # the affected men are an uncle and his nephews through his sisters, more related on the X than elsewhere
    assert x['x_kinship_ratio'] > 1.5
    assert x['skipped'] == pytest.approx((3 + 0.5) / 4)
    c = pattern_features(pedigree_from_rows(COUSINS))
    assert c['consanguineous'] == 1 and c['generations'] == 4


def test_pattern_classifier_on_classic_pedigrees():
    assert classify_by_patterns(pedigree_from_rows(DOMINANT)).call == 'AD'
    x = classify_by_patterns(pedigree_from_rows(X_RECESSIVE))
    assert next(iter(x.probabilities)) == 'XLR' and x.x_linked > 0.8
    c = classify_by_patterns(pedigree_from_rows(COUSINS))
    assert next(iter(c.probabilities)) == 'AR' and c.recessive > 0.9
    assert sum(c.probabilities.values()) == pytest.approx(1)
    assert 'from the pattern of affected people' in str(c)
    nobody = classify_by_patterns(pedigree_from_rows([(pid, f, m, s, 1) for pid, f, m, s, _ in DOMINANT]))
    assert nobody.call == 'uncertain'


def test_training_the_pattern_classifier():
    families = [(f.pedigree, mode) for f, mode in simulated(('AD', 'XLR'), n=60)]
    model = train_pattern_classifier(families[::2])
    assert model.modes == ('AD', 'XLR') and model.weights.shape == (2, len(model.features) + 1)
    right = [max(model.probabilities(pattern_features(p)).items(), key=lambda x: x[1])[0] == mode
             for p, mode in families[1::2]]
    assert np.mean(right) > 0.8


def test_network_score_parts():
    ped = pedigree_from_rows(DOMINANT)
    follows = {'1': 1, '2': 0, '3': 1, '4': 0, '5': 1, '6': 0, '7': 1, '8': 0, '9': 1}
    parts = network_parts(ped, follows)
    assert parts['edge_consistency'] == 1 and parts['generation_continuity'] == 1
    # an unaffected heterozygous carrier lies between affected people in the graph
    assert network_parts(ped, dict(follows, **{'4': 1}))['carrier_betweenness'] >= 0
    broken = dict(follows, **{'3': 2, '2': 0, '1': 0})
    assert network_parts(ped, broken)['edge_consistency'] < 1
    assert 0 <= segregation_network_score(ped, follows) <= 1
    assert 0 <= segregation_network_score(ped, follows, 'AR', extended=True) <= 1


# -- the same as the original PedNet, in legacy/ ------------------------------------------------------------------
def need(name):
    """The module, or the test skipped if it can't be imported (not installed, or installed but broken)"""
    try:
        return importlib.import_module(name)
    except ImportError as error:
        pytest.skip(f'needs {name} ({error})')


def legacy():
    for name in ('networkx', 'pandas', 'sklearn', 'scipy'):
        need(name)
    need('matplotlib').use('Agg')
    return need('PedigreeDAGAnalysis'), need('ModeClassifierMain'), need('SegScoreMain')


def legacy_graph(dag, pedigree):
    import pandas as pd
    rows = [{'FamilyID': pedigree.family, 'IndividualID': p.id, 'PaternalID': p.father or '0',
             'MaternalID': p.mother or '0', 'Sex': p.sex, 'Phenotype': p.phenotype}
            for p in pedigree.people if not p.placeholder]
    for row in rows:
        for key in ('PaternalID', 'MaternalID'):
            if row[key] != '0' and pedigree[row[key]].placeholder:
                row[key] = '0'
    return dag.construct_pedigree_graph(pd.DataFrame(rows))


def test_the_same_metrics_as_the_original():
    dag, _, _ = legacy()
    for f, mode in simulated(('AD', 'AR', 'XLR', 'XLD'), n=8, unknown_phenotype=0.1):
        ours = network_metrics(f.pedigree)
        theirs = dag.calc_pedigree_metrics(legacy_graph(dag, f.pedigree))
        for metric in METRICS:
            assert ours[metric] == pytest.approx(theirs[metric]), metric


def test_the_same_classifier_and_thresholds_as_the_original():
    dag, main, _ = legacy()
    from sklearn.metrics import auc, roc_curve
    families = [(f.pedigree, mode) for f, mode in simulated(n=50, seed=8)]
    ours = train_metric_classifier(families)
    rows = [network_metrics(p) for p, _ in families]
    modes = np.array([mode for _, mode in families])
    for metric in METRICS:
        values = np.array([r[metric] for r in rows])
        high = 'AD' if values[modes == 'AD'].mean() > values[modes == 'AR'].mean() else 'AR'
        fpr, tpr, thr = roc_curve(modes, values, pos_label=high)
        if auc(fpr, tpr) >= 0.7:
            assert ours[metric].threshold == pytest.approx(main.youden_threshold(fpr, tpr, thr))
            assert ours[metric].auc == pytest.approx(auc(fpr, tpr))
        else:
            assert metric not in ours
    legacy_thresholds = {m: {'threshold': t.threshold, 'direction': f'HIGH->{t.high}', 'auc': t.auc}
                         for m, t in ours.items()}
    for pedigree, _ in families[:30]:
        expected = main.MoI_classification(legacy_graph(dag, pedigree), legacy_thresholds)
        assert classify_by_metrics(pedigree, ours) == ('uncertain' if expected == 'Uncertain' else expected)


def test_the_same_network_score_as_the_original():
    dag, _, seg = legacy()
    for f, mode in simulated(('AD', 'AR'), n=6, seed=9, sequenced=0.7):
        G = legacy_graph(dag, f.pedigree)
        for v in f.variants:
            gt = {p: g for p, g in v.genotypes.items() if g is not None and p in G}
            if not gt:
                continue
            expected = seg.raw_categorical_scoring(G, gt)
            ours = network_parts(f.pedigree, v.genotypes)
            for part, value in expected.items():
                assert ours[part] == pytest.approx(value), part
            for method, extended in (('Original', False), ('Extended', True)):
                weights = {'w_edge': 0.6, 'w_gen': 0.2, 'w_bet': 0.2} if not extended else \
                    {'w_edge': 0.6, 'w_gen': 0.1, 'w_bet': 0.1, 'w_found': 0.1, 'w_depth': 0.1}
                assert segregation_network_score(f.pedigree, v.genotypes, mode, extended) == pytest.approx(
                    seg.segregation_network_score(G, gt, mode, method, expected, weights))
