"""
Tests of the original modules in legacy/, kept for comparison with the new methods. They need the original
requirements (pip install -r legacy/requirements.txt), and are skipped without them.
"""
import math
import random

import pytest

for _name in ('networkx', 'pandas', 'matplotlib', 'sklearn', 'scipy'):
    pytest.importorskip(_name)
# legacy/ is on the path when the tests run from the repository (see pyproject.toml), not from a source distribution
pytest.importorskip('PedigreeDataGeneration')

import matplotlib  # noqa: E402

matplotlib.use('Agg')

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import PedigreeDataGeneration as PDG  # noqa: E402
from ModeClassifierMain import metric_thresholds_determination, youden_threshold  # noqa: E402
from PedigreeDAGAnalysis import (aff_child_with_unaff_parents, aff_parents_with_unaffected_child, consanguinity_analysis,
                                 construct_pedigree_graph, generations)
from SegScoreMain import alt_depth_ratio, linked_variant_rank, trial_based_segregation_scoring_weight_optimization  # noqa: E402


def pedigree(rows):
    '''A pedigree dataframe from (IndividualID, PaternalID, MaternalID, Sex, Phenotype) rows'''
    return pd.DataFrame([('F',) + row for row in rows],
                        columns=['FamilyID', 'IndividualID', 'PaternalID', 'MaternalID', 'Sex', 'Phenotype'])


def hardy_weinberg(q):
    p = 1 - q
    return {0: p * p, 1: 2 * p * q, 2: q * q}


# ---------------------------------------------------------------------
# Pedigree and variant generation
# ---------------------------------------------------------------------
@pytest.mark.parametrize('q', [0.02, 0.1, 0.3, 0.5])
def test_ancestors_keep_hardy_weinberg_proportions(q):
    # someone drawn from the population, with their parents chosen by the reverse weights, must have parents who are
    # themselves in Hardy-Weinberg proportions
    weights = PDG.inheritance_weights(1 - q, q)
    hw = hardy_weinberg(q)
    father = dict.fromkeys(hw, 0.0)
    mother = dict.fromkeys(hw, 0.0)
    for child, p_child in hw.items():
        pairs, w = weights['reverse_genotypes'][child], weights['reverse_weights'][child]
        for (f, m), wi in zip(pairs, w):
            father[f] += p_child * wi / sum(w)
            mother[m] += p_child * wi / sum(w)
    for g in hw:
        assert father[g] == pytest.approx(hw[g])
        assert mother[g] == pytest.approx(hw[g])
    # two carriers can have a child without the alternate allele
    assert (1, 1) in weights['reverse_genotypes'][0]


@pytest.mark.parametrize('mode', ['AD', 'AR'])
def test_generated_pedigrees(mode):
    weights = PDG.inheritance_weights(0.8, 0.2)
    random.seed(1)
    for i in range(30):
        df = PDG.pedigree_generator(f'F{i}', mode, 3, (4, 4), (20, 20), (50, 50), (50, 50), True).set_index('IndividualID')
        G = construct_pedigree_graph(df.reset_index())
        # as many generations as asked for
        assert max(generations(G).values()) == 3
        for person, row in df.iterrows():
            g = int(row['Genotype'])
            affected = g >= 1 if mode == 'AD' else g == 2
            assert row['Phenotype'] == (2 if affected else 1)
            father, mother = int(row['PaternalID']), int(row['MaternalID'])
            if father:
                pair = (int(df.loc[father, 'Genotype']), int(df.loc[mother, 'Genotype']))
                assert g in weights['forward_genotypes'][pair]


def mendelian_background(df, table):
    '''For each background variant (not the linked one), whether every child's genotype could come from the child's parents'''
    rows = df.set_index('IndividualID')
    weights = PDG.inheritance_weights(0.5, 0.5)
    results = []
    for variant, genotypes in table.items():
        if variant == 'chr1:100000_A>T':
            continue
        results.append(all(genotypes[person] in weights['forward_genotypes'][(genotypes[int(row['PaternalID'])], genotypes[int(row['MaternalID'])])]
                           for person, row in rows.iterrows() if int(row['PaternalID'])))
    return results


def test_inherited_background_variants_are_mendelian():
    random.seed(2)
    df = PDG.pedigree_generator('F', 'AR', 3, (4, 4), (20, 20), (50, 50), (50, 50), True)
    inherited = mendelian_background(df, PDG.simulate_variant_table(df, (100, 100), (20, 20), (10, 40), inherited_background=True))
    assert len(inherited) == 20 and all(inherited)
    # drawn independently for each person, as by default, most background variants break Mendel's laws somewhere in the family
    independent = mendelian_background(df, PDG.simulate_variant_table(df, (100, 100), (20, 20), (10, 40)))
    assert len(independent) == 20 and sum(independent) < 10


def test_variant_table_check_tries_again(monkeypatch):
    # with almost no one sequenced, the table fails its check (more than two people sequenced) and is made again,
    # up to 51 times, rather than being kept after the first try
    calls = []
    real = PDG.simulate_variant_table

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(PDG, 'simulate_variant_table', counted)
    random.seed(3)
    PDG.PedGraph_VarTable_generator(1, 'AD', 3, (3, 3), (50, 50), (50, 50), True, (2, 2), (1, 1), (10, 10))
    assert len(calls) > 1


# ---------------------------------------------------------------------
# Pedigree graph analysis
# ---------------------------------------------------------------------
def test_text_ids():
    G = construct_pedigree_graph(pedigree([('a', '0', '0', 1, 2), ('b', '0', '0', 2, 1), ('c', 'a', 'b', 1, 2)]))
    assert set(G.nodes) == {'a', 'b', 'c'}
    assert set(G.edges) == {('a', 'c'), ('b', 'c')}


def test_decimal_and_missing_parent_ids():
    # a parent column with a missing value is read as decimals, so its zeros are 0.0
    G = construct_pedigree_graph(pedigree([(1, 0.0, 0.0, 1, 2), (2, 0.0, 0.0, 2, 1), (3, 1.0, 2.0, 1, 2), (4, float('nan'), 0.0, 2, 1),
                                           (5, 3.0, 4.0, 2, 1)]))
    assert set(G.nodes) == {1, 2, 3, 4, 5}
    assert set(G.edges) == {(1, 3), (2, 3), (3, 5), (4, 5)}


def test_generations_of_people_who_married_in():
    # 4 married in with no parents in the pedigree, and 8 married in with their parents, 6 and 7, drawn too
    G = construct_pedigree_graph(pedigree([
        (1, 0, 0, 1, 2), (2, 0, 0, 2, 1), (3, 1, 2, 1, 2), (4, 0, 0, 2, 1), (5, 3, 4, 1, 2),
        (6, 0, 0, 1, 1), (7, 0, 0, 2, 1), (8, 6, 7, 2, 1), (10, 5, 8, 2, 2)]))
    assert generations(G) == {1: 0, 2: 0, 3: 1, 4: 1, 6: 1, 7: 1, 5: 2, 8: 2, 10: 3}


def test_generations_when_people_of_two_generations_have_children():
    # 4 has a child with his niece 6: whatever the order of the rows, each child is one generation below its lower parent
    rows = [(1, 0, 0, 1, 1), (2, 0, 0, 2, 1), (3, 1, 2, 1, 1), (4, 1, 2, 1, 2), (5, 0, 0, 2, 1), (6, 3, 5, 2, 2), (7, 4, 6, 1, 1)]
    for order in (rows, rows[::-1], rows[4:] + rows[:4]):
        assert generations(construct_pedigree_graph(pedigree(order))) == {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3}


def test_consanguinity():
    # 6 and 8 are first cousins, the grandchildren of 1 and 2, and 9 is their child
    G = construct_pedigree_graph(pedigree([
        (1, 0, 0, 1, 1), (2, 0, 0, 2, 1), (3, 1, 2, 1, 1), (4, 1, 2, 2, 1), (5, 0, 0, 2, 1), (7, 0, 0, 1, 1),
        (6, 3, 5, 1, 1), (8, 7, 4, 2, 1), (9, 6, 8, 1, 2), (10, 6, 8, 2, 1)]))
    ancestry, consanguineous = consanguinity_analysis(G)
    assert ancestry[9] == {1, 2, 3, 4, 5, 6, 7, 8}
    assert set(consanguineous) == {9, 10}
    assert consanguineous[9]['common_ancestor'] in {1, 2}
    assert consanguineous[9]['degree_separation'] == 4


def test_inheritance_flags():
    # two affected parents with an unaffected child rule out a recessive condition, whatever the siblings
    G = construct_pedigree_graph(pedigree([(1, 0, 0, 1, 2), (2, 0, 0, 2, 2), (3, 1, 2, 1, 1), (4, 1, 2, 2, 2)]))
    assert aff_parents_with_unaffected_child(G)
    assert not aff_child_with_unaff_parents(G)
    G = construct_pedigree_graph(pedigree([(1, 0, 0, 1, 1), (2, 0, 0, 2, 1), (3, 1, 2, 1, 2)]))
    assert aff_child_with_unaff_parents(G)
    assert not aff_parents_with_unaffected_child(G)


# ---------------------------------------------------------------------
# Mode of inheritance classification and segregation scoring
# ---------------------------------------------------------------------
def test_youden_threshold():
    fpr = np.array([0, 0.1, 0.5, 1])
    tpr = np.array([0, 0.6, 0.9, 1])
    thresholds = np.array([np.inf, 0.8, 0.5, 0.1])
    # Youden's J is 0, 0.5, 0.4 and 0, so the best threshold is 0.8
    assert youden_threshold(fpr, tpr, thresholds) == 0.8


def test_ties_are_shared_out():
    assert linked_variant_rank(0.6, [0.5, 0.2]) == 1
    assert linked_variant_rank(0.5, [0.5, 0.2]) == 1.5
    assert linked_variant_rank(0.4, [0.5, 0.4, 0.4]) == 3


def test_depth_of_a_pedigree_without_generations():
    G = nx.DiGraph()
    G.add_node(1, phenotype=2)
    assert alt_depth_ratio(G, {1: 1}) == 0


def test_mode_classifier_runs():
    random.seed(4)
    results, thresholds = metric_thresholds_determination(pedigree_count=40, generation_range=(3, 3))
    assert len(results) == 8
    assert {r['PredictedMode'] for r in results} <= {'AD', 'AR', 'Uncertain'}
    assert set(thresholds) == {'ratio_aff_parent', 'sibling_aff_ratio', 'gen_cov', 'avg_bet_unaff'}


@pytest.mark.parametrize('inherited_background', [False, True])
def test_segregation_scoring_runs(inherited_background):
    random.seed(5)
    results, weights = trial_based_segregation_scoring_weight_optimization(
        Mode='AR', pedigree_count=20, generation_range=(3, 3), inherited_background=inherited_background)
    assert len(results) == 4
    assert sum(weights.values()) == pytest.approx(1, abs=1e-3)
    for family in results.values():
        rank = family['Original']['RankLinkedRank']
        assert 1 <= rank <= len(family['VarTable']) and math.isfinite(rank)
