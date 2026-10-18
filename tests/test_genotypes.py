import gzip

import pytest

from pednet import read_genotype_table, read_vcf

VCF = """##fileformat=VCFv4.2
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">
##INFO=<ID=CSQ,Number=.,Type=String,Description="Consequence annotations from Ensembl VEP. Format: Allele|Consequence|IMPACT|SYMBOL|Gene">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\tC
1\t100\trs1\tA\tG\t50\tPASS\tAF=0.001;CSQ=G|missense_variant|MODERATE|GENE1|ENSG1\tGT:DP\t0/1:10\t1|1:12\t./.:0
1\t200\t.\tC\tT,CA\t50\tLowQual\tAF=0.01,0.002;CSQ=T|stop_gained|HIGH|GENE2|ENSG2,CA|frameshift|HIGH|GENE3|ENSG3\tGT\t1/2\t0/0\t2/2
X\t300\t.\tG\tA\t50\t.\tANN=A|missense_variant|MODERATE|GENEX|ENSGX\tGT\t1\t0/1\t.
"""


def write(tmp_path, name, text, compressed=False):
    path = tmp_path / name
    if compressed:
        with gzip.open(path, 'wt') as handle:
            handle.write(text)
    else:
        path.write_text(text)
    return str(path)


@pytest.mark.parametrize('compressed', [False, True])
def test_read_vcf(tmp_path, compressed):
    variants = read_vcf(write(tmp_path, 'v.vcf.gz' if compressed else 'v.vcf', VCF, compressed))
    assert [v.id for v in variants] == ['rs1', '1:200:C>T', '1:200:C>CA', 'X:300:G>A']
    rs1, t, ca, x = variants
    assert rs1.genotypes == {'A': 1, 'B': 2, 'C': None}
    assert rs1.allele_frequency == 0.001 and rs1.gene == 'GENE1' and rs1.pos == 100
    # each alternate allele of a site is a variant of its own
    assert t.genotypes == {'A': 1, 'B': 0, 'C': 0} and ca.genotypes == {'A': 1, 'B': 0, 'C': 2}
    assert (t.allele_frequency, ca.allele_frequency) == (0.01, 0.002)
    assert (t.gene, ca.gene) == ('GENE2', 'GENE3')
    # a man's haploid call on the X, and SnpEff's annotation
    assert x.genotypes == {'A': 1, 'B': 1, 'C': None} and x.x_linked and x.gene == 'GENEX'
    assert x.allele_frequency is None


def test_read_vcf_options(tmp_path):
    path = write(tmp_path, 'v.vcf', VCF)
    assert [v.id for v in read_vcf(path, passed_only=True)] == ['rs1', 'X:300:G>A']
    assert set(read_vcf(path, samples=['B'])[0].genotypes) == {'B'}
    assert read_vcf(path, gene_key='ANN')[0].gene is None


def test_gene_fields(tmp_path):
    text = VCF.split('#CHROM')[0] + '#CHROM' + VCF.split('#CHROM')[1].split('\n')[0] + '\n'
    rows = ['1\t1\t.\tA\tG\t.\t.\tGENE=ABC\tGT\t0/1\t0/0\t0/0',
            '1\t2\t.\tA\tG\t.\t.\tGene.refGene=DEF\\x3bGHI\tGT\t0/1\t0/0\t0/0',
            '1\t3\t.\tA\tG\t.\t.\tGENEINFO=JKL:123|MNO:456\tGT\t0/1\t0/0\t0/0',
            '1\t4\t.\tA\tG\t.\t.\tBCSQ=missense|PQR|ENST1|protein_coding\tGT\t0/1\t0/0\t0/0',
            '1\t5\t.\tA\tG\t.\t.\tDP=3\tGT\t0/1\t0/0\t0/0']
    path = write(tmp_path, 'g.vcf', text + '\n'.join(rows) + '\n')
    assert [v.gene for v in read_vcf(path)] == ['ABC', 'DEF', 'JKL', 'PQR', None]


def test_read_genotype_table(tmp_path):
    text = 'variant\tchrom\tpos\tgene\taf\tA\tB\tC\nv1\t1\t10\tG1\t0.001\t1\t0/1\t\nv2\tX\t\t\t\t2\t./.\t1|1\n'
    v1, v2 = read_genotype_table(write(tmp_path, 't.tsv', text))
    assert v1.genotypes == {'A': 1, 'B': 1, 'C': None}
    assert v1.gene == 'G1' and v1.pos == 10 and v1.allele_frequency == 0.001
    assert v2.genotypes == {'A': 2, 'B': None, 'C': 2} and v2.x_linked and v2.gene is None
    csv = read_genotype_table(write(tmp_path, 't.csv', text.replace('\t', ',')))
    assert [v.genotypes for v in csv] == [v1.genotypes, v2.genotypes]


def test_unnamed_variants(tmp_path):
    variants = read_genotype_table(write(tmp_path, 't.tsv', 'A\tB\n1\t0\n'))
    assert variants[0].id == 'variant 1' and variants[0].genotypes == {'A': 1, 'B': 0}
