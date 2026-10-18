import io

import pytest

from pednet import Pedigree, PedigreeError, Person, pedigree_from_rows, read_ped, write_ped

PED = """# a comment
F1 1 0 0 1 2
F1 2 0 0 2 1
F1 3 1 2 1 2
F1 4 1 2 2 0
F2 a 0 0 1 1
F2 b 0 0 2 2
F2 c a b 2 -9
"""


def test_read_ped_from_text_file_and_path(tmp_path):
    path = tmp_path / 'f.ped'
    path.write_text(PED)
    for source in (PED, io.StringIO(PED), str(path), path):
        families = read_ped(source)
        assert set(families) == {'F1', 'F2'}
        f1 = families['F1']
        assert f1.ids == ['1', '2', '3', '4']
        assert f1['3'].father == '1' and f1['3'].mother == '2'
        assert f1.affected() == ['1', '3']
        assert f1['4'].phenotype == 0
        assert families['F2']['c'].phenotype == 0


def test_read_one_family():
    families = read_ped(PED, family='F2')
    assert list(families) == ['F2']
    with pytest.raises(PedigreeError):
        read_ped(PED, family='F3')


def test_short_line():
    with pytest.raises(PedigreeError, match='Line 1'):
        read_ped('F1 1 0 0 1\n')


def test_missing_parents_are_added():
    # 3's parents have no lines of their own, and 5 has only a mother in the file
    ped = pedigree_from_rows([('3', '1', '2', 0, 2), ('4', 0, 0, 2, 1), ('5', 0, '4', 1, 1)])
    assert ped['1'].added and ped['1'].sex == 1
    assert ped['2'].added and ped['2'].sex == 2
    father = ped['5'].father
    assert father == '4~partner' and ped[father].placeholder and ped[father].sex == 1
    assert len(ped.notes) == 3


def test_children_with_one_parent_share_the_other():
    # a father's children whose mother isn't in the file are full siblings, with one unknown mother between them
    ped = pedigree_from_rows([('1', 0, 0, 1, 1), ('3', '1', 0, 1, 2), ('4', '1', 0, 1, 2), ('5', '1', 0, 2, 1)])
    assert ped['3'].mother == ped['4'].mother == ped['5'].mother == '1~partner'
    assert len(ped) == 5
    assert any('3, 4 and 5 have only a father (1) in the file' in n for n in ped.notes)


def test_parents_written_as_decimals():
    # a file saved from a table with blanks in its parent columns can give the parents as 1.0 and 2.0
    ped = read_ped('F 1 0 0 1 1\nF 2 0 0 2 1\nF 3 1.0 2.0 1 2\n')['F']
    assert ped['3'].father == '1' and ped['3'].mother == '2' and len(ped) == 3
    assert any('1.0 is taken to be 1' in n for n in ped.notes)
    # but not when someone has the decimal ID itself
    ped = read_ped('F 1 0 0 1 1\nF 1.0 0 0 1 1\nF 2 0 0 2 1\nF 3 1.0 2 1 2\n')['F']
    assert ped['3'].father == '1.0'


def test_sex_from_role():
    ped = pedigree_from_rows([('1', 0, 0, 2, 1), ('2', 0, 0, 0, 1), ('3', '1', '2', 1, 2)])
    assert ped['1'].sex == 1 and ped['2'].sex == 2
    assert any('listed as female' in n for n in ped.notes)


def test_impossible_pedigrees():
    with pytest.raises(PedigreeError, match='more than once'):
        pedigree_from_rows([('1', 0, 0, 1, 1), ('1', 0, 0, 1, 1)])
    with pytest.raises(PedigreeError, match='both a father and a mother'):
        pedigree_from_rows([('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 1), ('4', '2', '1', 1, 1)])
    with pytest.raises(PedigreeError, match='own ancestors'):
        pedigree_from_rows([('1', '3', '2', 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 1)])


def test_no_parent_values():
    ped = pedigree_from_rows([('1', 0.0, '0', 1, 1), ('2', None, 'NA', 2, 1), ('3', '1', '2', 1, 2),
                              ('4', float('nan'), '-9', 2, 1)])
    assert ped.founders() == ['1', '2', '4']


def test_rows_as_dicts_and_pandas():
    rows = [{'FamilyID': 'X', 'IndividualID': 1, 'PaternalID': 0, 'MaternalID': 0, 'Sex': 1, 'Phenotype': 2},
            {'FamilyID': 'X', 'IndividualID': 2, 'PaternalID': 0, 'MaternalID': 0, 'Sex': 2, 'Phenotype': 1},
            {'FamilyID': 'X', 'IndividualID': 3, 'PaternalID': 1.0, 'MaternalID': 2.0, 'Sex': 2, 'Phenotype': 2}]
    ped = pedigree_from_rows(rows)
    # a whole number stored as a decimal, as pandas stores a column with blanks, is the same ID as the whole number
    assert ped.family == 'X' and ped['3'].father == '1'
    assert pedigree_from_rows([(1, 0, 0, 1.0, 2.0), (2.0, None, None, 2.0, 1.0), (3, 1.0, 2.0, 2.0, 2.0)])['3'].sex == 2
    try:
        import pandas as pd
    except ImportError as error:
        pytest.skip(f'needs pandas ({error})')
    frame = pd.DataFrame(rows)
    frame['PaternalID'] = frame['PaternalID'].astype(int)
    frame['MaternalID'] = frame['MaternalID'].astype(int)
    ped = pedigree_from_rows(frame)
    assert ped['3'].father == '1' and ped.affected() == ['1', '3']


def test_generations_and_order():
    rows = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 1), ('4', 0, 0, 2, 1), ('5', '3', '4', 1, 2),
            ('6', 0, 0, 1, 1), ('7', 0, 0, 2, 1), ('8', '6', '7', 2, 1), ('9', '5', '8', 2, 2)]
    expected = {'1': 0, '2': 0, '3': 1, '4': 1, '6': 1, '7': 1, '5': 2, '8': 2, '9': 3}
    for order in (rows, rows[::-1], rows[4:] + rows[:4]):
        ped = pedigree_from_rows(order)
        assert ped.generations() == expected
        position = {pid: k for k, pid in enumerate(ped.order())}
        for p in ped:
            if p.father:
                assert position[p.father] < position[p.id] and position[p.mother] < position[p.id]


def test_generations_across_generations():
    # 4 has a child with his niece 6
    rows = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 1), ('4', '1', '2', 1, 2), ('5', 0, 0, 2, 1),
            ('6', '3', '5', 2, 2), ('7', '4', '6', 1, 1)]
    for order in (rows, rows[::-1]):
        assert pedigree_from_rows(order).generations() == {'1': 0, '2': 0, '3': 1, '4': 1, '5': 1, '6': 2, '7': 3}


def test_consanguinity_and_ancestors():
    rows = [('1', 0, 0, 1, 1), ('2', 0, 0, 2, 1), ('3', '1', '2', 1, 1), ('4', '1', '2', 2, 1), ('5', 0, 0, 2, 1),
            ('6', 0, 0, 1, 1), ('7', '3', '5', 1, 1), ('8', '6', '4', 2, 1), ('9', '7', '8', 1, 2)]
    ped = pedigree_from_rows(rows)
    assert ped.ancestors('9') == {'1', '2', '3', '4', '5', '6', '7', '8'}
    related = ped.consanguineous()
    assert set(related) == {'9'} and related['9'] in {'1', '2'}
    assert ped.children('1') == ['3', '4']


def test_write_ped_round_trip(tmp_path):
    ped = pedigree_from_rows([('3', '1', '2', 0, 2), ('5', 0, '4', 1, 1), ('4', 0, 0, 2, 1)], family='Z')
    text = write_ped(ped, tmp_path / 'out.ped')
    assert '~' not in text
    again = read_ped(str(tmp_path / 'out.ped'))['Z']
    assert again['3'].father == '1' and again['5'].mother == '4'
    assert again['5'].father == '4~partner'


def test_problems():
    ped = Pedigree([Person('1', sex=0, phenotype=1)], 'P')
    assert ped.problems() == ['1 has no sex recorded', 'no one is affected']
