import json

import pytest
from pedigree_ga4gh import Kindred, Member, Relation, Term, dumps, kin, to_ga4gh_pedigree, to_phenopackets_family

from pednet import PedigreeError, infer_mode, read_ga4gh, read_ped, read_pedigree
from pednet.cli import main
from pednet.pedigree import AFFECTED, FEMALE, MALE, UNAFFECTED, UNKNOWN

CF = ('MONDO:0009061', 'cystic fibrosis')
HL = ('HP:0000365', 'Hearing impairment')


def kindred(terms, family='FAM1'):
    """A couple, their three children, and the husband's parents; terms gives each person's conditions as
    (id, label, excluded)"""
    sexes = {'gf': 'MALE', 'gm': 'FEMALE', 'f': 'MALE', 'm': 'FEMALE', 'a': 'FEMALE', 'b': 'MALE', 'c': 'FEMALE'}
    members = [Member(pid, sex=sex, terms=[Term(label, cid, excluded=excluded) for cid, label, excluded in terms.get(pid, [])])
               for pid, sex in sexes.items()]
    relations = [Relation('gf', kin.BIOLOGICAL_FATHER, 'f'), Relation('gm', kin.BIOLOGICAL_MOTHER, 'f')]
    for child in ('a', 'b', 'c'):
        relations += [Relation('f', kin.BIOLOGICAL_FATHER, child), Relation('m', kin.BIOLOGICAL_MOTHER, child)]
    return Kindred(family, members=members, relations=relations)


def pedigree_doc(terms):
    doc, _ = to_ga4gh_pedigree(kindred(terms), created_by='test')
    return doc


def phenotypes(pedigree):
    return {p.id: p.phenotype for p in pedigree.people}


def test_a_pedigree_drawn_in_the_studio():
    # studio2ga4gh records only the affected: everyone else is taken as unaffected, and a note says so
    families = read_ga4gh(pedigree_doc({'a': [CF + (False,)], 'c': [CF + (False,)]}))
    pedigree = families['FAM1']
    assert phenotypes(pedigree) == {'gf': 1, 'gm': 1, 'f': 1, 'm': 1, 'a': 2, 'b': 1, 'c': 2}
    assert pedigree['a'].father == 'f' and pedigree['a'].mother == 'm' and pedigree['f'].father == 'gf'
    assert pedigree['b'].sex == MALE and pedigree['m'].sex == FEMALE
    assert any('5 people with no record of the condition are taken as unaffected, as the file records only the '
               'affected' in n for n in pedigree.notes)


def test_a_pedigree_that_records_the_unaffected():
    # PedGen and PedInterp mark the unaffected as not having the condition: anyone with no record of it is unknown
    terms = {pid: [CF + (pid in ('b', 'f', 'm'),)] for pid in ('a', 'b', 'c', 'f', 'm')}
    pedigree = read_ga4gh(pedigree_doc(terms))['FAM1']
    assert phenotypes(pedigree) == {'gf': 0, 'gm': 0, 'f': 1, 'm': 1, 'a': 2, 'b': 1, 'c': 2}
    assert any('of unknown phenotype, as the file marks others as not having it' in n for n in pedigree.notes)
    # unless told otherwise
    pedigree = read_ga4gh(pedigree_doc(terms), unrecorded='unaffected')['FAM1']
    assert phenotypes(pedigree)['gf'] == UNAFFECTED and any('as asked' in n for n in pedigree.notes)


def test_the_same_as_the_ped_file():
    # a GA4GH Pedigree that records the affected and the unaffected reads as the PED file of the same family does
    ped = ('FAM1 gf 0 0 1 1\nFAM1 gm 0 0 2 0\nFAM1 f gf gm 1 1\nFAM1 m 0 0 2 1\n'
           'FAM1 a f m 2 2\nFAM1 b f m 1 1\nFAM1 c f m 2 2\n')
    status = {'gf': 1, 'f': 1, 'm': 1, 'a': 2, 'b': 1, 'c': 2}
    terms = {pid: [CF + (s == 1,)] for pid, s in status.items()}
    from_json, from_ped = read_ga4gh(pedigree_doc(terms))['FAM1'], read_ped(ped)['FAM1']
    assert [(p.id, p.father, p.mother, p.sex, p.phenotype) for p in from_json.people] == \
           [(p.id, p.father, p.mother, p.sex, p.phenotype) for p in from_ped.people]
    a, b = infer_mode(from_json), infer_mode(from_ped)
    assert a.call == b.call and [(m.mode, m.posterior) for m in a.modes] == [(m.mode, m.posterior) for m in b.modes]


def test_several_conditions():
    terms = {'a': [CF + (False,)], 'b': [HL + (False,)], 'c': [CF + (False,), HL + (False,)]}
    pedigree = read_ga4gh(pedigree_doc(terms))['FAM1']
    # any of them, with a note
    assert {pid for pid, s in phenotypes(pedigree).items() if s == AFFECTED} == {'a', 'b', 'c'}
    assert any('several conditions (cystic fibrosis, Hearing impairment)' in n for n in pedigree.notes)
    # or one of them, by its ID or its label, whatever its case
    for condition in ('MONDO:0009061', 'Cystic Fibrosis'):
        pedigree = read_ga4gh(pedigree_doc(terms), condition=condition)['FAM1']
        assert {pid for pid, s in phenotypes(pedigree).items() if s == AFFECTED} == {'a', 'c'}
        assert any('no record of cystic fibrosis are taken as unaffected' in n for n in pedigree.notes)
    with pytest.raises(PedigreeError, match=r'No one in the file has the condition asthma\. The conditions it names: '
                                            r'cystic fibrosis \(MONDO:0009061\); Hearing impairment \(HP:0000365\)'):
        read_ga4gh(pedigree_doc(terms), condition='asthma')


def test_a_phenopackets_family():
    # a Family has an affected status of its own, which is used unless a condition is chosen
    k = kindred({'a': [CF + (False,)], 'c': [HL + (False,)]})
    for m in k.members:
        m.affected = True if m.id == 'a' else (False if m.id in ('f', 'm', 'b') else None)
    doc, _ = to_phenopackets_family(k, created_by='test', proband_id='a')
    pedigree = read_ga4gh(doc)['FAM1']
    # (c, with a condition of her own, is affected in the Family as pedigree-ga4gh writes it)
    assert phenotypes(pedigree) == {'gf': UNKNOWN, 'gm': UNKNOWN, 'f': 1, 'm': 1, 'a': 2, 'b': 1, 'c': 2}
    assert pedigree['a'].father == 'f' and pedigree['f'].mother == 'gm'
    assert any("The Family's own affected status is used" in n for n in pedigree.notes)
    # a condition chosen is looked for in the people's Phenopackets
    pedigree = read_ga4gh(doc, condition='HP:0000365')['FAM1']
    assert {pid for pid, s in phenotypes(pedigree).items() if s == AFFECTED} == {'c'}


def test_read_pedigree_reads_either_kind(tmp_path):
    doc = pedigree_doc({'a': [CF + (False,)]})
    (tmp_path / 'fam_ga4gh-pedigree.json').write_text(dumps(doc))
    (tmp_path / 'fam.ped').write_text('FAM1 a 0 0 2 2\n')
    for source in (str(tmp_path / 'fam_ga4gh-pedigree.json'), dumps(doc), doc, open(tmp_path / 'fam_ga4gh-pedigree.json')):
        assert phenotypes(read_pedigree(source)['FAM1'])['a'] == AFFECTED
    assert list(read_pedigree(str(tmp_path / 'fam.ped'))) == ['FAM1']
    assert list(read_pedigree('FAM1 a 0 0 2 2\n')) == ['FAM1']   # a PED file's text, as read_ped takes it
    with pytest.raises(PedigreeError, match='are for GA4GH files'):
        read_pedigree(str(tmp_path / 'fam.ped'), condition='cystic fibrosis')
    assert list(read_pedigree(doc, family='FAM1')) == ['FAM1']
    with pytest.raises(PedigreeError, match='There is no family FAM2 in the file: its family is FAM1'):
        read_pedigree(doc, family='FAM2')


def test_what_is_not_a_pedigree():
    with pytest.raises(PedigreeError, match='This is not a JSON file'):
        read_ga4gh('{"id": ')
    with pytest.raises(PedigreeError, match='Phenopackets phenopacket, which has no pedigree'):
        read_ga4gh({'id': 'p1', 'subject': {'id': 'a', 'sex': 'MALE'}, 'metaData': {'created': '2026-01-01T00:00:00Z',
                    'createdBy': 'test', 'phenopacketSchemaVersion': '2.0'}})
    # the draft of version 1 of the standard, which no tool writes any more
    draft = {'id': 'F', 'individuals': [{'id': 'a', 'sexAssignedAtBirth': 'MALE'}], 'relationships': []}
    with pytest.raises(PedigreeError, match='cannot be upgraded'):
        read_ga4gh(draft)
    with pytest.raises(PedigreeError, match="unrecorded must be 'unaffected' or 'unknown'"):
        read_ga4gh(pedigree_doc({}), unrecorded='affected')


def test_the_commands_read_ga4gh_files(tmp_path, capsys):
    path = tmp_path / 'fam_ga4gh-pedigree.json'
    terms = {'a': [CF + (False,)], 'c': [CF + (False,)], 'b': [HL + (False,)]}
    path.write_text(dumps(pedigree_doc(terms)))
    assert main(['check', str(path)]) == 0
    out = capsys.readouterr().out
    assert 'Family FAM1: 7 people, 3 affected' in out and 'note: The file names several conditions' in out
    assert main(['check', str(path), '--condition', 'cystic fibrosis', '--unrecorded', 'unknown']) == 0
    out = capsys.readouterr().out
    assert 'Family FAM1: 7 people, 2 affected' in out and '5 people with no record of cystic fibrosis are taken as of unknown phenotype, as asked' in out
    assert main(['mode', str(path), '--condition', 'MONDO:0009061']) == 0
    assert 'AR' in capsys.readouterr().out
    assert main(['graph', str(path)]) == 0
    capsys.readouterr()
    ped = tmp_path / 'fam.ped'
    ped.write_text('FAM1 a 0 0 2 2\n')
    assert main(['check', str(ped), '--condition', 'cystic fibrosis']) == 1
    assert 'are for GA4GH files' in capsys.readouterr().err
