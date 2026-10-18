import numpy as np
import pytest

from pednet.simulate import simulate_family


@pytest.mark.parametrize('mode', ['AD', 'AR', 'XLR', 'XLD'])
def test_simulated_family(mode):
    rng = np.random.default_rng(7)
    for _ in range(5):
        f = simulate_family(mode, rng, background=5)
        ped = f.pedigree
        assert ped[f.proband].affected and len(ped.affected()) >= 2
        causal = f.variants[0]
        assert causal.id == 'causal' and f.causal == ('causal',)
        assert causal.x_linked == (mode in ('XLR', 'XLD'))
        copies = f.disease[f.proband]
        assert copies == 2 if mode == 'AR' else copies >= 1
        assert f.proband in f.sequenced and set(causal.genotypes) == set(f.sequenced)
        # the background variants are the proband's
        assert len(f.variants) == 6
        assert all(v.genotypes[f.proband] >= 1 for v in f.variants[1:])


def test_consanguineous_family():
    f = simulate_family('AR', np.random.default_rng(8), consanguineous=True)
    assert f.pedigree.consanguineous()


def test_compound_heterozygote():
    rng = np.random.default_rng(9)
    for _ in range(5):
        f = simulate_family('AR', rng, compound=True, background=3, background_pairs=2)
        a, b = f.variants[:2]
        assert f.causal == ('causal a', 'causal b') and a.gene == b.gene == 'GENE0'
        assert a.genotypes[f.proband] >= 1 and b.genotypes[f.proband] >= 1
        pairs = f.variants[5:]
        assert len(pairs) == 4 and pairs[0].gene == pairs[1].gene == 'PAIR1'
        assert all(v.genotypes[f.proband] >= 1 for v in pairs)
    with pytest.raises(ValueError):
        simulate_family('AD', rng, compound=True)


def test_genotype_errors():
    rng = np.random.default_rng(10)
    exact = simulate_family('AD', rng, background=0, sequenced=1.0)
    assert exact.variants[0].genotypes == exact.disease
    noisy = simulate_family('AD', rng, background=0, sequenced=1.0, error_rate=0.5)
    calls = noisy.variants[0].genotypes
    assert set(calls.values()) <= {0, 1, 2} and calls != noisy.disease
