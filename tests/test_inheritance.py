import numpy as np
import pytest

from pednet import MODES, infer_mode, infer_shared_mode, pedigree_from_rows
from pednet.inheritance import mode_log_likelihoods

# three generations of a dominant condition, with father-to-son transmission
DOMINANT = [('1', 0, 0, 1, 2), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2), ('4', '1', '2', 2, 1), ('5', '1', '2', 2, 2),
            ('6', 0, 0, 2, 1), ('7', '3', '6', 1, 2), ('8', '3', '6', 2, 1), ('9', '3', '6', 1, 2)]
# an X-linked recessive condition: affected men related through unaffected women
X_RECESSIVE = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2), ('4', '1', '2', 2, 1),
               ('5', '1', '2', 2, 1), ('6', 0, 0, 1, 1), ('7', '6', '4', 1, 2), ('8', '6', '4', 2, 1),
               ('9', 0, 0, 1, 1), ('10', '9', '5', 1, 2), ('11', '9', '5', 1, 1)]
# two affected children of first cousins
COUSINS = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 1), ('4', '1', '2', 2, 1), ('5', 0, 0, 2, 1),
           ('6', 0, 0, 1, 1), ('7', '3', '5', 1, 1), ('8', '6', '4', 2, 1), ('9', '7', '8', 1, 2),
           ('10', '7', '8', 2, 2), ('11', '7', '8', 1, 1), ('12', '7', '8', 2, 1)]


def test_dominant():
    result = infer_mode(pedigree_from_rows(DOMINANT, 'D'))
    assert result.call == 'AD'
    assert result.posterior('AD') > 0.95
    assert result.dominant > 0.95
    assert 'autosomal dominant' in str(result)


def test_x_linked_recessive():
    result = infer_mode(pedigree_from_rows(X_RECESSIVE))
    assert result.modes[0].mode == 'XLR'
    assert result.call == 'XLR'
    assert result.x_linked > 0.9


def test_consanguineous_recessive():
    result = infer_mode(pedigree_from_rows(COUSINS))
    assert result.modes[0].mode == 'AR'
    assert result.recessive > 0.9
    assert infer_mode(pedigree_from_rows(COUSINS), modes=('AD', 'AR')).call == 'AR'


def test_posteriors_sum_to_one_and_threshold():
    result = infer_mode(pedigree_from_rows(COUSINS), threshold=0.9999)
    assert sum(m.posterior for m in result.modes) == pytest.approx(1)
    assert result.dominant + result.recessive == pytest.approx(1)
    assert result.call == 'uncertain'
    assert [r['mode'] for r in result.table()] == [m.mode for m in result.modes]


def test_prior():
    result = infer_mode(pedigree_from_rows(DOMINANT), prior={'AD': 0, 'AR': 1, 'XLR': 1, 'XLD': 1})
    assert result.posterior('AD') == 0
    with pytest.raises(ValueError):
        infer_mode(pedigree_from_rows(DOMINANT), modes=('AD',), prior={'AR': 1})


def test_unknown_mode():
    with pytest.raises(ValueError, match='Unknown mode'):
        infer_mode(pedigree_from_rows(DOMINANT), modes=('AD', 'MT'))
    assert infer_mode(pedigree_from_rows(DOMINANT), modes='ad, ar').call == 'AD'


def test_no_one_affected():
    rows = [(pid, f, m, s, 1) for pid, f, m, s, _ in DOMINANT]
    result = infer_mode(pedigree_from_rows(rows))
    assert result.call == 'uncertain'
    assert all(m.posterior == pytest.approx(0.25) for m in result.modes)
    assert any('no one is affected' in n for n in result.notes)


def test_proband():
    ped = pedigree_from_rows(DOMINANT)
    assert infer_mode(ped, proband='7').call == 'AD'
    assert any('not recorded as affected' in n for n in infer_mode(ped, proband='8').notes)
    with pytest.raises(ValueError):
        infer_mode(ped, proband='nobody')


def test_unknown_sex_is_noted():
    rows = DOMINANT + [('10', '3', '6', 0, 1)]
    assert any('is unknown' in n for n in infer_mode(pedigree_from_rows(rows)).notes)


def test_order_of_people_doesnt_matter():
    a = mode_log_likelihoods(pedigree_from_rows(X_RECESSIVE))
    b = mode_log_likelihoods(pedigree_from_rows(X_RECESSIVE[::-1]))
    for mode in MODES:
        assert a[mode] == pytest.approx(b[mode])


def test_penetrance_weights():
    ped = pedigree_from_rows(DOMINANT)
    one = mode_log_likelihoods(ped, penetrance=0.8)
    weighted = mode_log_likelihoods(ped, penetrance=(0.8, 0.95), penetrance_weights=(1, 0))
    for mode in MODES:
        assert weighted[mode] == pytest.approx(one[mode])
    with pytest.raises(ValueError):
        mode_log_likelihoods(ped, penetrance=(0.8, 0.95), penetrance_weights=(1,))


def test_shared_mode_is_the_product_of_the_families():
    # two small recessive-looking families: each alone is unsure, together more sure
    sibs = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 2), ('4', '1', '2', 2, 2), ('5', '1', '2', 1, 1)]
    a, b = pedigree_from_rows(sibs, 'A'), pedigree_from_rows(sibs, 'B')
    alone = infer_mode(a)
    together = infer_shared_mode({'A': a, 'B': b})
    assert together.family == 'A, B'
    assert together.posterior('AR') > alone.posterior('AR')
    for m in together.modes:
        assert m.log10_likelihood == pytest.approx(2 * next(e.log10_likelihood for e in alone.modes if e.mode == m.mode))


def test_mode_log_likelihoods_are_conditional_probabilities():
    # given that he has the disease genotype, the chance that a man on his own is affected is the penetrance, here
    # averaged over the default penetrances
    logs = mode_log_likelihoods(pedigree_from_rows([('1', 0, 0, 1, 2)]))
    assert all(np.exp(v) == pytest.approx(np.mean([0.6, 0.8, 0.95, 0.99])) for v in logs.values())
