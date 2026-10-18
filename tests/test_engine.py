"""The variable elimination engine against brute-force sums over every assignment of genotypes"""
import itertools

import numpy as np
import pytest

from pednet import engine, pedigree_from_rows
from pednet.compound import PairLocus
from pednet.engine import Network
from pednet.models import transmission

# a first-cousin marriage: a loop
COUSINS = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 1), ('4', '1', '2', 2, 1), ('5', 0, 0, 2, 1),
           ('6', 0, 0, 1, 1), ('7', '3', '5', 1, 1), ('8', '6', '4', 2, 1), ('9', '7', '8', 0, 2)]
# on the X: sons, a daughter, someone of unknown sex, and a grandson through the daughter
X_FAMILY = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2), ('4', '1', '2', 2, 1), ('5', '1', '2', 0, 1),
            ('6', 0, 0, 1, 1), ('7', '6', '4', 1, 2)]
TRIO = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2), ('4', '1', '2', 2, 2)]


def brute_force(net, unary, make=transmission):
    tables = [None if par is None else make(net.kinds[i]) for i, par in enumerate(net.parents)]
    total = 0.0
    for states in itertools.product(*[range(d) for d in net.domains]):
        p = 1.0
        for i, s in enumerate(states):
            p *= unary[i][s]
            if net.parents[i] is not None:
                f, m = net.parents[i]
                p *= tables[i][s, states[f], states[m]]
            if p == 0:
                break
        total += p
    return total


def random_unary(net, rng, batch=None):
    shape = (lambda d: (batch, d)) if batch else (lambda d: (d,))
    return [rng.uniform(0.05, 1.0, size=shape(d)) for d in net.domains]


@pytest.mark.parametrize('rows,x_linked', [(COUSINS, False), (X_FAMILY, True), (X_FAMILY, False)])
def test_matches_brute_force(rows, x_linked):
    rng = np.random.default_rng(1)
    net = Network(pedigree_from_rows(rows), x_linked)
    unary = random_unary(net, rng, batch=3)
    got = np.exp(net.log_likelihood(unary))
    for b in range(3):
        assert got[b] == pytest.approx(brute_force(net, [u[b] for u in unary]), rel=1e-10)


def test_kinds_on_the_x():
    net = Network(pedigree_from_rows(X_FAMILY), True)
    assert net.kinds == ['M', 'F', 'M', 'F', 'U', 'M', 'M']
    assert net.domains == [2, 3, 2, 3, 5, 2, 2]


def test_transmission_tables_sum_to_one():
    for kind in ('A', 'M', 'F'):
        assert np.allclose(transmission(kind).sum(axis=0), 1)
    assert np.allclose(transmission('U').sum(axis=0), 1)
    for kind in ('A', 'M', 'F', 'U'):
        assert np.allclose(PairLocus.transmission(kind).sum(axis=0), 1)


@pytest.mark.parametrize('x_linked', [False, True])
def test_pair_locus_matches_brute_force(x_linked):
    rng = np.random.default_rng(2)
    net = Network(pedigree_from_rows(TRIO), x_linked, PairLocus())
    unary = random_unary(net, rng, batch=2)
    got = np.exp(net.log_likelihood(unary))
    for b in range(2):
        assert got[b] == pytest.approx(brute_force(net, [u[b] for u in unary], PairLocus.transmission), rel=1e-10)


def test_mixed_and_unbatched_factors():
    rng = np.random.default_rng(3)
    net = Network(pedigree_from_rows(COUSINS), False)
    unary = random_unary(net, rng)
    single = net.log_likelihood(unary)
    assert single.shape == (1,)
    batched = [np.tile(u, (4, 1)) if i % 2 else u for i, u in enumerate(unary)]
    assert np.allclose(net.log_likelihood(batched), single[0])


def test_large_batches_are_split(monkeypatch):
    rng = np.random.default_rng(4)
    net = Network(pedigree_from_rows(COUSINS), False)
    unary = random_unary(net, rng, batch=50)
    whole = net.log_likelihood(unary)
    monkeypatch.setattr(engine, 'MAX_TABLE', net.plan.largest * 7)
    assert np.allclose(net.log_likelihood(unary), whole)


def test_zero_likelihood():
    net = Network(pedigree_from_rows(TRIO), False)
    unary = [np.ones(3) for _ in net.domains]
    unary[2] = np.zeros(3)
    assert net.log_likelihood(unary)[0] == -np.inf


def test_tiny_numbers_dont_underflow():
    # a hundred unrelated people with tiny factors: the likelihood is far below the smallest double
    rows = [(str(i), 0, 0, 1, 1) for i in range(100)]
    net = Network(pedigree_from_rows(rows), False)
    unary = [np.full(3, 1e-10) for _ in rows]
    assert net.log_likelihood(unary)[0] == pytest.approx(100 * np.log(3e-10))
