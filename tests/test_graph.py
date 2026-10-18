import networkx as nx
import numpy as np
import pytest

from pednet import pedigree_from_rows
from pednet import graph as g

# 1 and 2 have 3 and 4; 3 and 5 have 7; 6 and 4 have 8; the first cousins 7 and 8 have 9 and 10; 11 is the son of 1
# by another partner, who isn't in the file
COUSINS = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 1), ('4', '1', '2', 2, 1), ('5', 0, 0, 2, 1),
           ('6', 0, 0, 1, 1), ('7', '3', '5', 1, 1), ('8', '6', '4', 2, 1), ('9', '7', '8', 1, 2),
           ('10', '7', '8', 2, 2), ('11', '1', 0, 1, 1)]
# two brothers, c and d, married to two sisters, g and h: their children i and j are double first cousins
DOUBLE = [('a', 0, 0, 1, 1), ('b', 0, 0, 2, 1), ('c', 'a', 'b', 1, 1), ('d', 'a', 'b', 1, 1), ('e', 0, 0, 1, 1),
          ('f', 0, 0, 2, 1), ('g', 'e', 'f', 2, 1), ('h', 'e', 'f', 2, 1), ('i', 'c', 'g', 1, 2), ('j', 'd', 'h', 2, 1)]
# a father, a mother, two sons and two daughters, and a daughter's son
NUCLEAR = [('F', 0, 0, 1, 2), ('M', 0, 0, 2, 1), ('S1', 'F', 'M', 1, 1), ('S2', 'F', 'M', 1, 2),
           ('D1', 'F', 'M', 2, 1), ('D2', 'F', 'M', 2, 2), ('P', 0, 0, 1, 1), ('G', 'P', 'D1', 1, 2)]


def test_pedigree_graph():
    graph = g.pedigree_graph(pedigree_from_rows(COUSINS))
    assert isinstance(graph, nx.DiGraph) and nx.is_directed_acyclic_graph(graph)
    # the unknown mother PedNet adds for 11 is left out unless asked for
    assert '1~partner' not in graph and graph.number_of_nodes() == 11
    assert set(graph.predecessors('9')) == {'7', '8'} and graph.edges['7', '9']['parent'] == 'father'
    assert graph.nodes['9']['affected'] and graph.nodes['9']['generation'] == 3 and graph.nodes['1']['founder']
    with_placeholders = g.pedigree_graph(pedigree_from_rows(COUSINS), placeholders=True)
    assert set(with_placeholders.predecessors('11')) == {'1', '1~partner'}
    assert pedigree_from_rows(COUSINS).to_networkx().number_of_edges() == graph.number_of_edges()


def test_mating_and_moral_graphs():
    ped = pedigree_from_rows(COUSINS)
    mating = g.mating_graph(ped)
    couple = ('mating', '7', '8')
    assert mating.nodes[couple]['kind'] == 'mating'
    assert set(mating.predecessors(couple)) == {'7', '8'} and set(mating.successors(couple)) == {'9', '10'}
    moral = g.moral_graph(ped)
    assert moral.has_edge('7', '8') and moral.has_edge('3', '5') and moral.has_edge('1', '1~partner')
    assert not moral.has_edge('9', '10')


def test_loops():
    found = g.loops(pedigree_from_rows(COUSINS))
    assert len(found) == 1 and found[0].kind == 'consanguinity'
    assert set(found[0].people) == {'3', '4', '7', '8'} and ('7', '8') in found[0].couples
    assert g.loop_count(pedigree_from_rows(COUSINS)) == 1
    marriage = g.loops(pedigree_from_rows(DOUBLE))
    assert [loop.kind for loop in marriage] == ['marriage']
    assert g.loops(pedigree_from_rows(NUCLEAR)) == [] and g.loop_count(pedigree_from_rows(NUCLEAR)) == 0


@pytest.mark.parametrize('a,b,expected', [
    ('1', '3', 1 / 4), ('3', '4', 1 / 4), ('3', '11', 1 / 8), ('7', '8', 1 / 16), ('1', '7', 1 / 8),
    ('7', '4', 1 / 8), ('1', '2', 0), ('9', '9', 0.5 * (1 + 1 / 16)),
    # 9's parents are first cousins, so 9 is related to 7 through 8 too
    ('9', '7', 0.5 * (0.5 + 1 / 16)), ('9', '10', 0.5 * (0.5 + 1 / 16)),
])
def test_kinship(a, b, expected):
    assert g.kinship(pedigree_from_rows(COUSINS), a, b) == pytest.approx(expected)


def test_inbreeding():
    ped = pedigree_from_rows(COUSINS)
    F = g.inbreeding(ped)
    assert F['9'] == F['10'] == pytest.approx(1 / 16)
    assert F['7'] == 0 and F['1'] == 0
    assert g.kinship(pedigree_from_rows(DOUBLE), 'i', 'j') == pytest.approx(1 / 8)


@pytest.mark.parametrize('a,b,expected', [
    # the coefficients on the X: a man has his mother's X only, and a woman her father's and one of her mother's
    ('F', 'D1', 1 / 2), ('F', 'S1', 0), ('M', 'S1', 1 / 2), ('M', 'D1', 1 / 4), ('S1', 'S2', 1 / 2),
    ('D1', 'D2', 3 / 8), ('S1', 'D1', 1 / 4), ('F', 'F', 1), ('M', 'M', 1 / 2), ('G', 'S1', 1 / 4), ('G', 'F', 1 / 2),
])
def test_x_kinship(a, b, expected):
    assert g.kinship(pedigree_from_rows(NUCLEAR), a, b, x_linked=True) == pytest.approx(expected)


def drop_genes(ped, rng, x_linked=False):
    """Each person's two genes (or one, for a man on the X), dropped from founders whose genes are all different"""
    genes, label = {}, 0
    for pid in ped.order():
        p = ped[pid]
        haploid = x_linked and p.sex == 1
        if p.father is None:
            genes[pid] = (label,) if haploid else (label, label + 1)
            label += 2
            continue
        from_mother = genes[p.mother][rng.integers(len(genes[p.mother]))]
        if haploid:
            genes[pid] = (from_mother,)
        else:
            from_father = genes[p.father][rng.integers(len(genes[p.father]))]
            genes[pid] = (from_father, from_mother)
    return genes


@pytest.mark.parametrize('rows,x_linked', [(COUSINS, False), (DOUBLE, False), (NUCLEAR, True), (COUSINS, True)])
def test_kinship_by_gene_dropping(rows, x_linked):
    # the kinship coefficient is the chance that genes drawn from the two are identical by descent: a check by
    # simulation, independent of the recursion
    ped = pedigree_from_rows(rows)
    if x_linked:
        ped = pedigree_from_rows([(pid, f, m, s if s else 2, ph) for pid, f, m, s, ph in rows])
    ids, phi = g.kinship_matrix(ped, x_linked)
    rng = np.random.default_rng(3)
    trials = 10000
    same = np.zeros((len(ids), len(ids)))
    for _ in range(trials):
        genes = drop_genes(ped, rng, x_linked)
        draws = [genes[pid][rng.integers(len(genes[pid]))] for pid in ids]
        again = [genes[pid][rng.integers(len(genes[pid]))] for pid in ids]
        drawn = np.array(draws)
        same += drawn[:, None] == np.array(again)[None, :]
    assert np.abs(same / trials - phi).max() < 0.02


@pytest.mark.parametrize('a,b,name,degree', [
    ('9', '7', 'father', 1), ('9', '8', 'mother', 1), ('7', '9', 'son', 1), ('3', '4', 'sister', 1),
    ('9', '10', 'sister', 1), ('3', '11', 'half-brother', 2), ('9', '3', 'grandfather', 2),
    ('9', '1', 'great-grandfather', 3), ('7', '4', 'aunt', 2), ('4', '7', 'nephew', 2), ('7', '8', 'first cousin', 3),
    ('7', '11', 'half-uncle', 3), ('9', '4', 'grandmother', 2), ('1', '2', 'unrelated', None), ('9', '9', 'self', 0),
])
def test_relationship(a, b, name, degree):
    r = g.relationship(pedigree_from_rows(COUSINS), a, b)
    assert (r.name, r.degree) == (name, degree)


def test_relationship_details():
    r = g.relationship(pedigree_from_rows(DOUBLE), 'i', 'j')
    assert r.name == 'double first cousin' and r.degree == 3 and r.kinship == pytest.approx(1 / 8)
    assert r.coefficient == pytest.approx(1 / 4) and set(r.via) == {'a', 'b', 'e', 'f'}
    removed = pedigree_from_rows(COUSINS + [('12', 0, 0, 2, 1), ('13', '9', '12', 2, 1)])
    assert g.relationship(removed, '13', '8').name == 'grandmother'
    assert g.relationship(removed, '7', '13').name == 'granddaughter'
    # the sons of the half-brothers 3 and 11 share one grandparent
    half = pedigree_from_rows(COUSINS + [('12', 0, 0, 2, 1), ('13', '11', '12', 1, 1)])
    r = g.relationship(half, '7', '13')
    assert (r.name, r.degree, r.via) == ('half-first cousin', 4, ['1'])
    assert r.kinship == pytest.approx(1 / 32)


def test_centrality_and_measures():
    ped = pedigree_from_rows(NUCLEAR)
    betweenness = g.centrality(ped)
    assert max(betweenness, key=betweenness.get) in {'F', 'M', 'D1'}
    assert set(g.centrality(ped, 'degree')) == set(g.pedigree_graph(ped))
    with pytest.raises(ValueError):
        g.centrality(ped, 'pagerank')
    # five in the second generation and the grandson's father: none is another's ancestor
    assert g.pedigree_width(ped) == 5
    influence = g.founder_influence(ped)
    # F: paths to F, S1, S2, D1, D2, G, of which F, S2, D2 and G are affected
    assert influence['F'] == pytest.approx(4 / 6) and influence['P'] == pytest.approx(1 / 2)


def test_elimination():
    ped = pedigree_from_rows(COUSINS)
    plan = g.elimination(ped)
    assert sorted(plan.order) == sorted(ped.ids)
    assert plan.width >= 3 and plan.largest_table == 3 ** plan.width and plan.exact


def test_elimination_of_a_pedigree_too_tangled(monkeypatch):
    # the plan is still worked out and reported when exact likelihoods would need too large a table
    from pednet import engine
    monkeypatch.setattr(engine, 'MAX_STATES', 10)
    plan = g.elimination(pedigree_from_rows(COUSINS))
    assert not plan.exact and plan.largest_table > 10
    with pytest.raises(ValueError, match='too many loops'):
        engine.Network(pedigree_from_rows(COUSINS), False)


def test_table_sizes_never_overflow():
    # 3 ** 41 is beyond NumPy's 64-bit whole numbers, which would wrap round to a negative size and pass the check
    from pednet.engine import Plan
    plan = Plan([tuple(range(41))], [3] * 41, check=False)
    assert plan.largest == 3 ** 41 and not plan.feasible
    with pytest.raises(ValueError):
        Plan([tuple(range(41))], [3] * 41)


def test_loop_kinds_dont_depend_on_the_order_of_the_lines():
    # the double first cousins i and j marry: one loop from the marriages of the brothers and sisters, and one from
    # the cousins' marriage, whichever basis is chosen and however the lines of the file are ordered
    import random
    rows = DOUBLE + [('k', 'i', 'j', 1, 2)]
    first = g.loops(pedigree_from_rows(rows))
    assert [loop.kind for loop in first] == ['consanguinity', 'marriage']
    shuffle = random.Random(5)
    for _ in range(30):
        shuffled = rows[:]
        shuffle.shuffle(shuffled)
        again = g.loops(pedigree_from_rows(shuffled))
        assert [(loop.kind, loop.people) for loop in again] == [(loop.kind, loop.people) for loop in first]


def test_double_cousins_beyond_the_first():
    # the children of the double first cousins i and j (by partners from outside) are double second cousins
    rows = DOUBLE + [('x', 0, 0, 2, 1), ('y', 0, 0, 1, 1), ('m', 'i', 'x', 1, 1), ('n', 'y', 'j', 2, 1)]
    r = g.relationship(pedigree_from_rows(rows), 'm', 'n')
    assert (r.name, r.degree) == ('double second cousin', 5) and r.kinship == pytest.approx(2 / 64)
    once_removed = g.relationship(pedigree_from_rows(rows), 'i', 'n')
    assert once_removed.name == 'double first cousin once removed'


def test_relationship_with_the_matrix_given():
    ped = pedigree_from_rows(COUSINS)
    matrix = g.kinship_matrix(ped)
    assert g.relationship(ped, '7', '8', matrix) == g.relationship(ped, '7', '8')


def test_write_graph(tmp_path):
    ped = pedigree_from_rows(COUSINS)
    g.write_graph(ped, tmp_path / 'p.graphml')
    back = nx.read_graphml(tmp_path / 'p.graphml')
    assert set(back) == set(g.pedigree_graph(ped)) and back.number_of_edges() == 13
    g.write_graph(ped, tmp_path / 'm.graphml', mating=True)
    assert 'mating:7:8' in nx.read_graphml(tmp_path / 'm.graphml')
