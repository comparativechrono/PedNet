import os
import subprocess
import sys
from pathlib import Path

import pytest

import pednet
from pednet.cli import main

PED = """F1 1 0 0 1 2
F1 2 0 0 2 1
F1 3 1 2 1 2
F1 4 1 2 2 1
F1 5 0 0 2 1
F1 6 3 5 1 2
F1 7 3 5 2 1
F1 8 3 5 2 2
"""
TABLE = """variant\tchrom\tgene\taf\t1\t2\t3\t4\t5\t6\t7\t8
causal\t1\tG1\t0.0001\t1\t0\t1\t0\t0\t1\t0\t1
everyone\t1\tG2\t0.001\t1\t1\t1\t1\t1\t1\t1\t1
common\t1\tG3\t0.2\t1\t0\t1\t0\t0\t1\t0\t1
"""
CALLS = {'causal': ['0/1', '0/0', '0/1', '0/0', '0/0', '0/1', '0/0', '0/1'],
         'other': ['0/0', '0/1', '0/1', '0/1', '0/0', '0/1', '0/1', '0/0']}


def vcf(samples, calls):
    lines = ['##fileformat=VCFv4.2',
             '\t'.join(['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'] + samples)]
    for pos, (name, gts) in enumerate(calls.items(), start=1):
        lines.append('\t'.join(['1', str(100 * pos), name, 'A', 'G', '.', 'PASS', f'AF=0.0001;GENE=G{pos}', 'GT'] + gts))
    return '\n'.join(lines) + '\n'


def second_family(text):
    """The same family again as F2, with b before each ID"""
    lines = []
    for line in text.splitlines():
        _, pid, father, mother, sex, phenotype = line.split()
        lines.append(' '.join(['F2', 'b' + pid, '0' if father == '0' else 'b' + father,
                               '0' if mother == '0' else 'b' + mother, sex, phenotype]))
    return '\n'.join(lines) + '\n'


@pytest.fixture
def files(tmp_path):
    people = [str(i) for i in range(1, 9)]
    (tmp_path / 'f.ped').write_text(PED)
    (tmp_path / 'v.tsv').write_text(TABLE)
    (tmp_path / 'v.vcf').write_text(vcf(people + ['stranger'], {k: v + ['0/0'] for k, v in CALLS.items()}))
    (tmp_path / 'two.ped').write_text(PED + second_family(PED))
    (tmp_path / 'two.vcf').write_text(vcf(people + ['b' + p for p in people], {k: v + v for k, v in CALLS.items()}))
    return tmp_path


def test_check(files, capsys):
    assert main(['check', str(files / 'f.ped')]) == 0
    assert 'Family F1: 8 people, 4 affected, 3 generations' in capsys.readouterr().out


def test_mode(files, capsys):
    assert main(['mode', str(files / 'f.ped')]) == 0
    out = capsys.readouterr().out
    assert 'Family F1: autosomal dominant' in out and 'dominant 0.99' in out
    assert main(['mode', str(files / 'f.ped'), '--tsv', '--modes', 'AD,AR']) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'family\tmode\tposterior\tlog10_likelihood' and len(lines) == 3


def test_mode_together(files, capsys):
    assert main(['mode', str(files / 'two.ped'), '--together']) == 0
    assert 'Family F1, F2: autosomal dominant' in capsys.readouterr().out


def test_rank_table(files, capsys):
    out_file = files / 'scores.tsv'
    assert main(['rank', str(files / 'f.ped'), str(files / 'v.tsv'), '--out', str(out_file),
                 '--max-frequency', '0.01']) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0].startswith('Family F1: 2 variants')
    assert lines[2].split()[:2] == ['1.0', 'causal'] and 'PP1_' in lines[2]
    assert 'allele frequency above 0.01' in captured.err
    rows = out_file.read_text().splitlines()
    assert rows[0].startswith('rank\tcandidate\tkind') and len(rows) == 3


def test_rank_vcf(files, capsys):
    assert main(['rank', str(files / 'f.ped'), str(files / 'v.vcf'), '--modes', 'AD', '--penetrance', '0.9']) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines()[2].split()[1] == 'causal'
    assert 'stranger' in captured.err


def test_rank_across_families(files, capsys):
    assert main(['rank', str(files / 'two.ped'), str(files / 'two.vcf')]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith('Families F1, F2')
    assert out[2].split()[1] == 'causal' and 'F1' in out[2] and 'F2' in out[2]


def test_errors(files, capsys):
    assert main(['mode', str(files / 'missing.ped')]) == 1
    assert 'pednet:' in capsys.readouterr().err
    (files / 'strangers.vcf').write_text(vcf(['x', 'y'], {'v': ['0/1', '0/0']}))
    assert main(['rank', str(files / 'f.ped'), str(files / 'strangers.vcf')]) == 1
    assert 'None of the genotyped samples' in capsys.readouterr().err
    assert main(['rank', str(files / 'two.ped'), str(files / 'v.tsv'), '--proband', '3']) == 1
    assert '--family' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(['--version'])


def test_module_runs(files):
    source = str(Path(pednet.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([source, os.environ.get('PYTHONPATH', '')]))
    result = subprocess.run([sys.executable, '-m', 'pednet', 'check', str(files / 'f.ped')], capture_output=True,
                            text=True, env=env)
    assert result.returncode == 0 and 'Family F1' in result.stdout
