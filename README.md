# PedNet

PedNet answers two questions about a family with an inherited condition: how is the condition inherited, and which of the family's candidate variants co-segregates with it? It works both out from exact likelihoods: the probability of what was seen in the family, the phenotypes and the genotype calls, summed over every genotype that the people who weren't sequenced could have. Pedigrees with loops, such as the children of cousins, are handled exactly too.

Its answers are in the terms that variant classification uses: the posterior probability of each mode of inheritance, and for each candidate the co-segregation Bayes factor, with its strength as PP1 evidence for co-segregation (or BS4 against it) under the ACMG/AMP guidelines (Richards et al., 2015). Pairs of variants in one gene are scored as compound heterozygotes, and the evidence of several families can be added together.

```bash
pip install PedNet
```

It needs Python 3.9 or later and NumPy, and nothing else.

## On the command line

```bash
pednet check family.ped                    # what PedNet makes of a PED file
pednet mode family.ped                     # the mode of inheritance
pednet rank family.ped family.vcf.gz       # the candidate variants, the most likely cause first
pednet rank family.ped family.vcf.gz --out scores.tsv --max-frequency 0.01
```

With the example family in [examples](https://github.com/comparativechrono/PedNet/tree/main/examples):

```
$ pednet mode examples/dominant.ped
Family FAM1: autosomal dominant
  AD    0.999   (autosomal dominant)
  XLR   0.001   (X-linked recessive)
  XLD   0.000   (X-linked dominant)
  AR    0.000   (autosomal recessive)
  dominant 0.999, recessive 0.001
  X-linked 0.001, autosomal 0.999

$ pednet rank examples/dominant.ped examples/dominant.vcf
Family FAM1: 6 variants, the most likely cause first
rank  candidate  log10 odds  log10 BF  evidence    best mode  affected carriers  unaffected carriers  note
1.0   var1       2.05        2.05      PP1_Strong  AD         6/6                0/3
2.0   var2       -2.78       -2.78     BS4_Strong  AD         5/6                2/3
3.0   var4       -4.96       -4.96     BS4_Strong  AD         4/6                1/3
4.0   var5       -5.36       -2.49     BS4_Strong  XLD        6/6                0/3                  3 Mendelian errors; possibly de novo in III-1
5.0   var3       -14.09      -14.09    BS4_Strong  AR         1/6                2/3
6.0   var6       -14.79      -14.79    BS4_Strong  AR         1/6                0/3                  1 Mendelian error; possibly de novo in III-1
```

The options say what is known about the condition: `--modes AD` if the mode of inheritance is known (or `--modes AD,AR` to choose between two), `--penetrance 0.9` if the penetrance is (by default PedNet averages over a range), `--phenocopy` for the chance that someone is affected without the disease genotype, and `--proband` for the person through whom the family came to notice. `pednet mode --help` and `pednet rank --help` list them all. `--out` writes every candidate's scores to a file separated by tabs, with the likelihood ratio for each mode and the counts behind the table.

A PED file with several families is read family by family. `pednet mode --together` takes the families as having the same condition and gives one answer for them all, and `pednet rank` on several families adds up each candidate's evidence over the families in which an affected person carries it.

## In Python

```python
import pednet

family = pednet.read_ped('examples/recessive.ped')['FAM2']
print(pednet.infer_mode(family))

variants = pednet.read_vcf('examples/recessive.vcf', samples=family.ids)
for score in pednet.rank_variants(family, variants)[:4]:
    print(score.rank, score.id, round(score.log10_bf, 2), score.evidence)
```

```
Family FAM2: uncertain
  AR    0.746   (autosomal recessive)
  AD    0.112   (autosomal dominant)
  XLD   0.084   (X-linked dominant)
  XLR   0.057   (X-linked recessive)
  dominant 0.197, recessive 0.803
  X-linked 0.142, autosomal 0.858
1.0 var9 0.76 PP1_Moderate
2.0 var7a + var7b 0.82 PP1_Moderate
3.0 var10 -0.19
4.5 var7a -0.77 BS4_Moderate
```

Two affected children can't settle the mode, though they make a recessive condition likely. Of the candidates, a variant that both affected children have two copies of and a pair in one gene that they both carry, one variant from each parent, explain the family equally well.

`infer_mode` and `rank_variants` take the same options as the command (`modes`, `penetrance`, `phenocopy`, `proband` and more), and their docstrings describe them. `infer_shared_mode` and `rank_variants_across` work over several families. `pedigree_from_rows` makes a pedigree from rows or from a pandas DataFrame with the PED columns, as [PedGen](https://github.com/comparativechrono/pedigree-generator) and [PedInterp](https://github.com/comparativechrono/pedigree-interpreter) make them, and `Variant` holds a variant's genotypes if they come from somewhere else. `pednet.simulate` makes simulated families, with background variants passed down the pedigree, for testing.

## What goes in

**A PED file**, one person per line in six columns separated by spaces or tabs: family, person, father, mother, sex (1 male, 2 female, 0 unknown) and phenotype (2 affected, 1 unaffected, 0 or -9 unknown). A father or mother of 0 isn't in the file. Lines starting with `#` are skipped. A parent who has no line of their own is added, with an unknown phenotype, and so is the missing parent of someone with only one in the file; a parent's sex is taken from their role. `pednet check` lists anything PedNet changed or found odd.

**A VCF file** of the family's candidate variants, plain or gzipped. The genotypes come from the GT field, so the samples must be named as the people are in the PED file. Each alternate allele of a site is a candidate of its own. The allele frequency comes from the INFO field `AF`, or another (`--frequency-key AF_joint`, say, for gnomAD's), and the gene, which pairs variants up as compound heterozygotes, from the annotations of VEP (`CSQ`), SnpEff (`ANN`) or bcftools csq (`BCSQ`), or a field such as `GENE`, `Gene.refGene` (ANNOVAR) or `GENEINFO` (dbSNP); `--gene-key` names another. PedNet scores whatever it is given, so filter the VCF to the variants that could cause the condition first (rare, and protein-altering or otherwise plausible), or at least leave out common ones with `--max-frequency`. A whole exome's worth of variants takes well under a minute.

**Or a table** of variants, separated by tabs (or commas, for a `.csv` file): a column `variant` naming each, optional columns `chrom`, `pos`, `ref`, `alt`, `af` and `gene`, and a column for each person with the number of copies of the variant (0, 1 or 2) or a call such as `0/1`. An empty cell means the person wasn't genotyped.

## What comes out

### The mode of inheritance

The posterior probability of autosomal dominant (AD), autosomal recessive (AR), X-linked recessive (XLR) and X-linked dominant (XLD) inheritance, equally likely beforehand unless a prior is given. The call is the most probable mode if its probability reaches 0.8 (`--threshold`), and "uncertain" otherwise. With few affected people the four modes often can't be told apart, but whether the condition is dominant or recessive, and X-linked or autosomal, often can, so PedNet gives those probabilities too. The probabilities are well calibrated: in the benchmark, when the most probable mode had a probability of 0.8 or more, it was the true mode 87–97% of the time.

### The candidates

`log10 BF` is the log10 co-segregation Bayes factor: how much more probable the family's phenotypes are if this variant causes the condition than if some other locus does, given the genotypes. It is positive when the variant follows the condition through the family better than an unseen locus would be expected to, and negative when it follows it worse; a variant whose genotypes say nothing scores about 0. With full penetrance, a variant that follows the condition exactly has a Bayes factor of 2 to the power of the number of informative meioses (Jarvik and Browning, 2016). `evidence` gives its strength by the thresholds of Tavtigian et al. (2018), as Belman et al. (2020) use them for co-segregation:

| Bayes factor | log10 BF | evidence |
|---|---|---|
| 18.7 or more | 1.27 or more | PP1_Strong |
| 4.33 or more | 0.64 or more | PP1_Moderate |
| 2.08 or more | 0.32 or more | PP1_Supporting |
| between 1/2.08 and 2.08 | | none |
| 1/2.08 (0.48) or less | −0.32 or less | BS4_Supporting |
| 1/4.33 (0.23) or less | −0.64 or less | BS4_Moderate |
| 1/18.7 (0.053) or less | −1.27 or less | BS4_Strong |

The Bayes factor is worked out under the modes that suit the variant's chromosome, AD and AR for an autosomal variant and XLR and XLD for one on the X, each weighted by how well it explains the phenotypes. `log10 odds` adds the log10 probability that the mode is one that suits the chromosome, and the candidates are ranked by it, so that an X-linked variant ranks low in a family whose condition is clearly autosomal, and the other way round. Where the chromosome is the same, as it is for most candidates, the two rise and fall together.

`best mode` is the mode under which the candidate does best. The carriers are the affected and unaffected people genotyped who carry the variant; `note` points out Mendelian errors (children whose genotype their genotyped parents can't have given them, which suggests a calling error: the Bayes factor allows for errors, but such a variant is worth checking) and possible new mutations.

For variant classification, the Bayes factor should be worked out with what is known about the gene: its mode of inheritance (`--modes`), its penetrance (`--penetrance`) and the family's proband (`--proband`). PP1 applies only to a gene known to cause the condition, and the strengths are evidence to weigh with the rest, not a classification.

### Compound heterozygotes

A recessive condition is often caused by two different variants in one gene, one on each copy of the gene. Neither variant then follows the condition as a recessive variant should, so PedNet also scores each pair of variants in a gene that an affected person (the proband, if given) carries both of, as a single candidate (`var7a + var7b` above). The gene is then a locus with four kinds of copy: neither variant, one, the other, or both on the same copy, in cis. The family's genotypes show whether two variants are on different copies, as a compound heterozygote's are, or passed down together on one copy, which explains nothing (`var8a + var8b` in the example). Pairs need the variants' genes; `--no-compound` turns them off.

### Several families

The evidence of several families with the same condition is the sum of their log10 Bayes factors. `rank_variants_across`, and `pednet rank` on a PED file with several families, add it up over the families in which an affected person (the family's proband, if given) carries the candidate; a family in which no affected person carries it is left out, since the condition may have another cause there.

## How it works

Each person's genotype at the locus is a variable: the number of copies of the variant, 0, 1 or 2, and on the X chromosome 0 or 1 for a man. Someone of unknown sex who has no children can be either. Founders, the people with no parents in the pedigree, have Hardy-Weinberg proportions, and children have their parents' genotypes by Mendel's laws. Someone with the disease genotype is affected with probability equal to the penetrance, and someone without it with probability equal to the phenocopy rate (for XLR, women with one copy are affected at the phenocopy rate too, unless `carrier_penetrance` says otherwise). A genotype call is one copy out with probability 1 in 10,000 (`--error-rate`), and two copies out with that probability squared.

This makes the pedigree a Bayesian network, and the likelihood of what was seen in the family is a sum over every assignment of genotypes to everyone. PedNet works the sum out exactly by variable elimination, as Superlink does for linkage (Fishelson and Geiger, 2002): it sums the genotypes out one at a time, in an order chosen to keep the intermediate tables small. That is exact for pedigrees with loops as for any other, at a cost that grows with how tangled the loops are rather than with the size of the family. Many likelihoods are worked out at once, for many penetrances, probands or variants together.

**Ascertainment.** A family comes to notice through an affected person, the proband, so PedNet conditions on the proband having the disease genotype. Without a proband given, it averages over each affected person as the proband. This allows for how the family was found, and makes the answers depend very little on how common the disease allele is.

**The penetrance.** It is rarely known well, so by default PedNet averages the likelihood over penetrances of 0.6, 0.8, 0.95 and 0.99, equally weighted, with a phenocopy rate of 0.001. Give `--penetrance` when it is known.

**The Bayes factor** for a variant is the probability of the phenotypes given the genotypes, with the variant as the cause, over their probability with an unseen locus as the cause, carried by the proband: the full-likelihood Bayes factor of Thompson et al. (2003), which segregatr also computes. The variant's founders carry it at its allele frequency, but at no less than 0.001 (`--min-frequency`): a variant seen in the family is there however rare it is elsewhere, rarer frequencies change little, and much rarer ones would make a calling error seem a likelier explanation than a real carrier.

**Checks.** The tests compare the likelihoods with brute-force sums over every assignment of genotypes, for pedigrees with loops, on the X and for pairs of variants. On segregatr's documented example (four affected brothers, penetrance 0.9, phenocopy rate 0.01, allele frequency 0.0001), PedNet's likelihoods give segregatr's Bayes factor of 7.732161 exactly when conditioned as segregatr conditions, on the proband's genotype call. PedNet's own Bayes factor, conditioned on the proband having the disease genotype (one copy or two) so that every variant is judged against the same alternative, is 7.731.

## How well it works

[benchmarks/RESULTS.md](https://github.com/comparativechrono/PedNet/blob/main/benchmarks/RESULTS.md) compares PedNet with the original PedNet methods (in [legacy](https://github.com/comparativechrono/PedNet/tree/main/legacy)) and with a plain concordance score, on simulated families of three or four generations with incomplete penetrance, unknown phenotypes and half the relatives sequenced. Some of its results, for 200 families of each kind:

| | AD | AR | XLR | XLD |
|---|---|---|---|---|
| Mode called right, AD or AR (original classifier) | 0.94 (0.70) | 0.91 (0.98) | | |
| Mode called wrong, AD or AR (original classifier) | 0.04 (0.11) | 0.03 (0.01) | | |
| Called right as dominant or recessive, of all four modes | 0.82 | 0.81 | 0.97 | 0.78 |
| Called wrong as dominant or recessive | 0.07 | 0.02 | 0.02 | 0.05 |
| Mean rank of the causal variant among 21 (original network score) | 3.60 (9.94) | 1.08 (6.48) | 4.44 | 4.88 |

PedNet's ranks are with the mode known, as the original score's are. The causal variant often can't be told from a background variant passed down with it: in 73% of the AD families, some background variant had exactly the causal variant's genotypes in the people sequenced. For compound heterozygotes, among 32 single variants and 6 pairs, the causal pair ranked first in 57% of families and had a mean rank of 1.7, with the mode known; ranking single variants only, the better of the two causal variants had a mean rank of 6.6.

```bash
pip install pandas networkx matplotlib scikit-learn scipy
python benchmarks/benchmark.py --families 200 --seed 1
```

## What it doesn't do

- **One locus.** The condition is taken to come from one locus with one mode of inheritance, without polygenic background or modifiers. Variants are scored one at a time, or two in a gene; three or more variants together are not.
- **New mutations.** A variant new in the proband isn't modelled, and it scores against itself, since it doesn't follow the condition back through the family. Such variants are flagged as possibly de novo, to be judged on their own (PS2 or PM6, not PP1).
- **Age and sex.** The penetrance is the same for everyone with the disease genotype. Relatives too young to show a condition of later onset should be given an unknown phenotype (0) rather than recorded as unaffected.
- **Other modes.** Mitochondrial, Y-linked, imprinted and sex-limited inheritance aren't among the modes. A variant in a pseudoautosomal region of the X is inherited like an autosomal one, so give it an autosomal chromosome name.
- **Linkage.** A variant passed down with the causal one, near it on the chromosome, follows the condition just as well, and only its allele frequency can set it apart. PedNet ranks variants, not regions.
- **Genotype quality.** Calls are taken as they are, with the same small error rate for all, rather than with their read-level likelihoods.
- **Very tangled pedigrees.** A pedigree with many interlocking loops can need tables too large for exact likelihoods; PedNet then says so.

PedNet doesn't draw pedigrees: [Pedigree Studio](https://github.com/comparativechrono/pedigreestudio) does.

## The original PedNet

PedNet began as Nick Taylor's notebooks and modules, which treat the pedigree as a directed graph and use measures of it to classify the mode of inheritance and to score variants. They are kept in [legacy](https://github.com/comparativechrono/PedNet/tree/main/legacy), with their own [README](https://github.com/comparativechrono/PedNet/blob/main/legacy/README.md) and tests, so that the comparison in the benchmark can be rerun. They aren't part of the package on PyPI.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests of the original modules run too when their requirements (`legacy/requirements.txt`) are installed, and are skipped otherwise.

## References

- Belman S, Parsons MT, Spurdle AB, Goldgar DE, Feng BJ. Considerations in assessing germline variant pathogenicity using cosegregation analysis. *Genetics in Medicine* 2020;22:2052–2059.
- Fishelson M, Geiger D. Exact genetic linkage computations for general pedigrees. *Bioinformatics* 2002;18 Suppl 1:S189–S198.
- Jarvik GP, Browning BL. Consideration of cosegregation in the pathogenicity classification of genomic variants. *American Journal of Human Genetics* 2016;98:1077–1081.
- Richards S, Aziz N, Bale S, et al. Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the American College of Medical Genetics and Genomics and the Association for Molecular Pathology. *Genetics in Medicine* 2015;17:405–424.
- Tavtigian SV, Greenblatt MS, Harrison SM, et al. Modeling the ACMG/AMP variant classification guidelines as a Bayesian classification framework. *Genetics in Medicine* 2018;20:1054–1060.
- Thompson D, Easton DF, Goldgar DE. A full-likelihood method for the evaluation of causality of sequence variants from family data. *American Journal of Human Genetics* 2003;73:652–655.
- segregatr, segregation analysis for variant interpretation: an R package by Magnus Dehli Vigeland, on CRAN.

## Licence

MIT. PedNet is by Tim Hearn and Nick Taylor.
