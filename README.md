# PedNet

PedNet analyses pedigrees as graphs. A pedigree is a directed acyclic graph, with a node for each person and an edge from each parent to each child, and PedNet answers three kinds of question from it:

- **How is the family put together?** Its loops, how closely any two people are related, how inbred each person is, and network measures of the people and of the whole pedigree (`pednet.graph`).
- **What does the pattern of affected people say?** How the affected people sit in the graph, measured by the original PedNet's network metrics and by the signs clinical geneticists read from a pedigree, each with a classifier of the mode of inheritance (`pednet.features`).
- **What do exact likelihoods say?** With the graph as a Bayesian network of genotypes, the probability of each mode of inheritance, and how well each candidate variant co-segregates with the condition (`pednet.inheritance` and `pednet.segregation`).

The answers about inheritance are in the terms that variant classification uses: the posterior probability of each mode of inheritance, and for each candidate variant the co-segregation Bayes factor, with its strength as PP1 evidence for co-segregation (or BS4 against it) under the ACMG/AMP guidelines (Richards et al., 2015).

```bash
pip install PedNet
```

It needs Python 3.9 or later, NumPy and networkx.

## A pedigree is a graph

PedNet reads a pedigree into three graphs, each a networkx graph that the rest of networkx can work on, or that network tools such as Gephi and Cytoscape can open:

- the **pedigree graph** (`pedigree_graph`), the directed acyclic graph itself, with each person's sex, phenotype and generation on their node;
- the **mating graph** (`mating_graph`), which puts a node for each couple between the parents and their children, as a pedigree drawing has a line for each partnership. Its cycles are the pedigree's loops;
- the **moral graph** (`moral_graph`), the pedigree graph with each child's parents joined and the directions dropped, on which the likelihoods are worked out.

Most of what PedNet does is read off these graphs. The loops are a cycle basis of the mating graph. Kinship coefficients are sums over the paths that join two people through their common ancestors, and relationships are named from the shortest of those paths. The pattern features count the edges along which the condition passes, from father to son, say, or through an unaffected mother. The exact likelihoods sum each person's genotype out of the moral graph in turn, and how hard that is depends on the moral graph's treewidth.

## On the command line

```bash
pednet check family.ped                  # what PedNet makes of a PED file
pednet graph family.ped                  # the structure of the family, from the graph
pednet mode family.ped                   # the mode of inheritance
pednet rank family.ped family.vcf.gz     # the candidate variants, the most likely cause first
```

### The structure of the family

With the consanguineous family in [examples](https://github.com/comparativechrono/PedNet/tree/main/examples), in which two of the four children of first cousins are affected:

```
$ pednet graph examples/consanguineous.ped
Family FAM3: 12 people in 4 generations, 2 affected; 4 founders and 4 couples with children
  a loop, from consanguinity: III-1, II-1, II-2, III-2
  inbred: IV-1 (F = 0.0625), IV-2 (F = 0.0625), IV-3 (F = 0.0625), IV-4 (F = 0.0625)
  width 4 (the most people no one of whom is another's ancestor)
  exact likelihoods: elimination width 4, so the moral graph's treewidth is at most 3
  affected relatives, the closest first:
    IV-2 is the sister of IV-1 (first degree, kinship 0.2812)
  network metrics (the original PedNet's):
    ratio_aff_parent 0.000, sibling_aff_ratio 0.333, gen_cov 0.250, avg_bet_unaff 0.000
  pattern features:
    affected_share 0.192, affected_generations 0.250, vertical 0.167, skipped 0.833
    male_to_male 0.000, father_daughters 0.500, father_sons 0.500, mother_sons 0.500
    mother_daughters 0.500, male_share 0.500, sibship_share 0.357, consanguineous 1.000
    x_kinship_ratio 0.889, log_affected 1.099, generations 4.000
  most central (betweenness): III-1 0.406, III-2 0.406, II-1 0.167
```

The cousins' marriage makes a loop, and gives their children an inbreeding coefficient of 1/16. The affected brother and sister have a kinship of 0.2812 rather than a quarter, because their parents are related. `--kinship kinship.tsv` writes the kinship coefficients of everyone (`--x` for those on the X chromosome), and `--graphml family.graphml` writes the graph (`--mating` for the mating graph).

### The mode of inheritance

PedNet has three ways of judging the mode of inheritance. By default it works out the exact likelihood of the phenotypes under each mode; `--method patterns` uses the classifier of the pattern of affected people, and `--method metrics` the original PedNet's classifier, which chooses between AD and AR only.

```
$ pednet mode examples/consanguineous.ped
Family FAM3: autosomal recessive
  AR    0.928   (autosomal recessive)
  XLR   0.027   (X-linked recessive)
  AD    0.026   (autosomal dominant)
  XLD   0.020   (X-linked dominant)
  dominant 0.045, recessive 0.955
  X-linked 0.047, autosomal 0.953

$ pednet mode examples/consanguineous.ped --method patterns
Family FAM3, from the pattern of affected people: autosomal recessive
  AR    0.941   (autosomal recessive)
  AD    0.021   (autosomal dominant)
  XLD   0.021   (X-linked dominant)
  XLR   0.017   (X-linked recessive)
  dominant 0.042, recessive 0.958
  X-linked 0.038, autosomal 0.962

$ pednet mode examples/consanguineous.ped --method metrics
Family FAM3, by the original PedNet metric classifier: AR
  ratio_aff_parent 0.000, sibling_aff_ratio 0.333, gen_cov 0.250, avg_bet_unaff 0.000
```

### The candidate variants

With the example family with a dominant condition and its VCF file:

```
$ pednet rank examples/dominant.ped examples/dominant.vcf
Family FAM1: 6 variants, the most likely cause first
rank  candidate  log10 odds  log10 BF  evidence    best mode  affected carriers  unaffected carriers  note
1.0   var1       2.05        2.05      PP1_Strong  AD         6/6                0/3
2.0   var2       -2.78       -2.78     BS4_Strong  AD         5/6                2/3
3.0   var4       -4.96       -4.96     BS4_Strong  AD         4/6                1/3
4.0   var5       -5.36       -2.49     BS4_Strong  XLD        6/6                0/3                  3 Mendelian errors; possibly de novo in III-1
5.0   var3       -14.09      -14.09    BS4_Strong  AD         1/6                2/3
6.0   var6       -14.79      -14.79    BS4_Strong  AD         1/6                0/3                  1 Mendelian error; possibly de novo in III-1
```

The options of `mode` and `rank` say what is known about the condition: `--modes AD` if the mode of inheritance is known (or `--modes AD,AR` to choose between two), `--penetrance 0.9` if the penetrance is known (by default PedNet averages over a range), `--phenocopy` for the chance that someone is affected without the disease genotype, and `--proband` for the person through whom the family came to notice. `pednet mode --help` and `pednet rank --help` list them all. `--out` writes every candidate's scores to a file separated by tabs, with the likelihood ratio for each mode and the counts behind the table.

A PED file with several families is read family by family. `pednet mode --together` takes the families as having the same condition and gives one answer for them all, and `pednet rank` on several families combines each candidate's evidence over the families in which an affected person carries it (see Several families, below).

## In Python

```python
import networkx as nx
import pednet

family = pednet.read_ped('examples/consanguineous.ped')['FAM3']

graph = pednet.pedigree_graph(family)           # a networkx DiGraph, from each parent to each child
print(sorted(nx.ancestors(graph, 'IV-1')))
print(pednet.loops(family)[0].kind, pednet.relationship(family, 'III-1', 'III-2').name)
print(pednet.inbreeding(family)['IV-1'])
print(pednet.kinship(family, 'IV-1', 'IV-2'), pednet.kinship(family, 'IV-1', 'IV-2', x_linked=True))
print(pednet.classify_by_patterns(family).call, pednet.infer_mode(family).call)
```

```
['I-1', 'I-2', 'II-1', 'II-2', 'II-3', 'II-4', 'III-1', 'III-2']
consanguinity first cousin
0.0625
0.28125 0.25
AR AR
```

`pednet.graph` has the rest of the network measures (`centrality`, `pedigree_width`, `founder_influence`, `founder_cover`, `elimination`) and `write_graph` for GraphML. For candidate variants:

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
4.5 var7a -0.77
```

Two affected children can't settle the mode, though they make a recessive condition likely. Of the candidates, a variant that both affected children have two copies of, and a pair in one gene that they both carry, one variant from each parent, explain the family almost equally well. var7a on its own has no evidence given, since it should be judged with its pair.

`infer_mode` and `rank_variants` take the same options as the command (`modes`, `penetrance`, `phenocopy`, `proband` and more), and their docstrings describe them. `infer_shared_mode` and `rank_variants_across` work over several families. `pedigree_from_rows` makes a pedigree from rows or from a pandas DataFrame with the PED columns, as [PedGen](https://github.com/comparativechrono/pedigree-generator) and [PedInterp](https://github.com/comparativechrono/pedigree-interpreter) make them, and `Variant` holds a variant's genotypes if they come from somewhere else. `pednet.simulate` makes simulated families, with background variants passed down the pedigree, for testing.

## What goes in

**A PED file**, one person per line in six columns separated by spaces or tabs: family, person, father, mother, sex (1 male, 2 female, 0 unknown) and phenotype (2 affected, 1 unaffected, 0 or -9 unknown). A father or mother of 0 isn't in the file. Lines starting with `#` are skipped. A parent who has no line of their own is added, with an unknown phenotype, and so is the missing parent of someone with only one in the file (one unknown partner for each known parent, so that children with the same known parent are taken as full siblings); a parent's sex is taken from their role. `pednet check` lists anything PedNet changed or found odd.

**A VCF file** of the family's candidate variants, plain or gzipped. The genotypes come from the GT field, so the samples must be named as the people are in the PED file. Each alternate allele of a site is a candidate of its own. The allele frequency comes from the INFO field `AF`, or another (`--frequency-key AF_joint`, say, for gnomAD's), and the gene, which pairs variants up as compound heterozygotes, from the annotations of VEP (`CSQ`), SnpEff (`ANN`) or bcftools csq (`BCSQ`), or a field such as `GENE`, `Gene.refGene` (ANNOVAR) or `GENEINFO` (dbSNP); `--gene-key` names another. PedNet scores whatever it is given, so filter the VCF to the variants that could cause the condition first (rare, and protein-altering or otherwise plausible), or at least leave out common ones with `--max-frequency`. A whole exome's worth of variants takes well under a minute.

**Or a table** of variants, separated by tabs (or commas, for a `.csv` file): a column `variant` naming each, optional columns `chrom`, `pos`, `ref`, `alt`, `af` and `gene`, and a column for each person with the number of copies of the variant (0, 1 or 2) or a call such as `0/1`. An empty cell means the person wasn't genotyped.

## What comes out

### The structure of the family

**Loops.** A loop is a cycle in the mating graph, once its directions are dropped. PedNet finds a cycle basis, a set of independent loops from which every other can be made, and marks each loop as coming from consanguinity, when it passes through a couple who share an ancestor, or as a marriage loop, as when two brothers marry two sisters. The number of independent loops is the mating graph's cyclomatic number: its edges, less its nodes, plus the number of its connected parts. Many bases are possible, but the marriage loops are found first, as a basis of the graph without the related couples, so that how many loops are of each kind doesn't depend on the choice, or on the order of the lines in the PED file.

**Kinship and inbreeding.** The kinship coefficient of two people is the probability that a gene drawn at random from each, at the same place, is identical by descent: a quarter for a parent and child or for two siblings, 1/16 for first cousins. PedNet works out every pair's at once, in the order of the graph, parents before children, by the recursion that sums Wright's path coefficients over the paths through common ancestors: a person's kinship with anyone not descended from them is the average of their parents' kinships with that person. On the X chromosome a man has only his mother's copy, so a father and son have a kinship of 0 there, and two brothers one of a half. A person's inbreeding coefficient is their parents' kinship.

**Relationships.** `relationship(family, a, b)` names b's relationship to a from the shortest path up from a to a common ancestor and down again to b: father, half-sister, great-aunt, first cousin once removed. The relationship is full rather than half when the closest common ancestors include a couple, and double when they include two couples, as for double first cousins. Its degree is the number of steps on the path, less one for a full relationship, so that siblings are first-degree relatives and first cousins third-degree. The kinship coefficient comes with it, summed over every path, so that it shows when two people are more closely related than the name says.

**Network measures.** Each person's centrality in the graph (degree, betweenness or closeness, as networkx works them out). The pedigree's width: the most people of whom none is another's ancestor, which by Dilworth's theorem is also the fewest lines of descent that between them take in everyone, found from a maximum matching. Each founder's influence, the share of the paths down the graph from them that end at an affected person, and a small set of founders from whom everyone descends, as the original PedNet had them. And the elimination width, the most genotypes in one table when the likelihoods are worked out, which says how hard they are: the moral graph's treewidth is at most one less.

### The pattern of affected people

Before any genotypes are known, the way the affected people sit in the graph says much about how a condition is inherited. PedNet measures it in two ways, each with a classifier of the mode of inheritance.

**The original PedNet's network metrics** (`network_metrics`) are four: the share of affected people with an affected parent (`ratio_aff_parent`), the share of affected people's siblings who are affected (`sibling_aff_ratio`), the share of generations with someone affected (`gen_cov`), and how central unaffected people are among the affected (`avg_bet_unaff`, the betweenness of each in the graph of the affected people and them alone). The original classifier (`classify_by_metrics`) sets a threshold on each metric from simulated families, by Youden's J, keeps the metrics whose area under the ROC curve is 0.7 or more, and lets them vote for AD or AR, with two votes for AR when an affected child has unaffected parents and one for AD otherwise; a mode with three quarters of the votes is called. It gives the same answers as the original code, as the tests check.

**The pattern features** (`pattern_features`) are the signs clinical geneticists look for, as features of the graph: the condition passing down the generations; from father to son, which rules out the X; from affected fathers to their daughters and sons, and from affected mothers to theirs; generations skipped; the share of the affected who are men; affected siblings; an affected child of related parents; and how much more closely the affected people are related on the X than on an autosome, which is high for men related through women and nil for a father and son. The pattern classifier (`classify_by_patterns`) is a multinomial logistic regression on these features, for all four modes, trained on 4,000 simulated families of two to five generations with penetrances from 0.4 to 1 (`benchmarks/train_classifiers.py`). Its probabilities are only as good as the simulated families are like the real one, and for a family far from them the likelihood, which can be told what is known about the condition, is the safer guide.

**The original network score** of a candidate variant (`segregation_network_score`) combines how many of the graph's parent-child edges its genotypes fit by Mendel's laws, how many generations carry it, and how central its unaffected carriers are. It too is kept, with the original's results, for comparison.

### The mode of inheritance

`pednet mode` gives the posterior probability of autosomal dominant (AD), autosomal recessive (AR), X-linked recessive (XLR) and X-linked dominant (XLD) inheritance, equally likely beforehand unless a prior is given. The call is the most probable mode if its probability reaches 0.8 (`--threshold`), and "uncertain" otherwise. With few affected people the four modes often can't be told apart, but whether the condition is dominant or recessive, and X-linked or autosomal, often can, so PedNet gives those probabilities too. In the benchmark, the probabilities were roughly calibrated: when the most probable mode had a probability between 0.8 and 0.9, it was the true mode in 87% of families, and between 0.9 and 0.99, in 91%.

### The candidates

`log10 BF` is the log10 co-segregation Bayes factor: how much more probable the family's phenotypes are if this variant causes the condition than if some other locus does, given the genotypes. It is positive when the variant follows the condition through the family better than an unseen locus would be expected to, and negative when it follows it worse; a variant whose genotypes say nothing scores about 0. With full penetrance, a variant that follows the condition exactly has a Bayes factor of 2 to the power of the number of informative meioses (Jarvik and Browning, 2016). `evidence` gives its strength by the thresholds of Tavtigian et al. (2018), as Belman et al. (2020) use them for co-segregation:

| Bayes factor | log10 BF | evidence |
|---|---|---|
| 18.7 or more | 1.2718 or more | PP1_Strong |
| 4.33 or more | 0.6365 or more | PP1_Moderate |
| 2.08 or more | 0.3181 or more | PP1_Supporting |
| between 1/2.08 and 2.08 | between −0.3181 and 0.3181 | none |
| 1/2.08 (0.481) or less | −0.3181 or less | BS4_Supporting |
| 1/4.33 (0.231) or less | −0.6365 or less | BS4_Moderate |
| 1/18.7 (0.053) or less | −1.2718 or less | BS4_Strong |

The Bayes factor is worked out under the modes that suit the variant's chromosome, AD and AR for an autosomal variant and XLR and XLD for one on the X, each weighted by how well it explains the phenotypes. `log10 odds` adds the log10 probability that the mode is one that suits the chromosome, and the candidates are ranked by it, so that an X-linked variant ranks low in a family whose condition is clearly autosomal, and the other way round. Where the chromosome is the same, as it is for most candidates, the two rise and fall together.

`best mode` is the most probable mode of inheritance if the candidate is the cause. The carriers are the affected and unaffected people genotyped who carry the variant. `note` points out Mendelian errors (children whose genotype their genotyped parents can't have given them, which suggests a calling error: the Bayes factor allows for errors, but such a variant is worth checking) and possible new mutations. The evidence is left out, and the note says why, for a variant that may be new in an affected person and otherwise fits the family, which is for PS2 or PM6 to judge rather than PP1 or BS4, and for a variant whose pair, scored as a compound heterozygote, is evidence for co-segregation (PP1) and better evidence than the variant on its own. A candidate whose genotypes can't have been passed down without a calling error isn't scored at all with `--error-rate 0`.

For variant classification, the Bayes factor should be worked out with what is known about the gene: its mode of inheritance (`--modes`), its penetrance (`--penetrance`) and the family's proband (`--proband`). PP1 applies only to a gene known to cause the condition, and the strengths are evidence to weigh with the rest, not a classification.

### Compound heterozygotes

A recessive condition is often caused by two different variants in one gene, one on each copy of the gene. Neither variant then follows the condition as a recessive variant should, so PedNet also scores each pair of variants in a gene that an affected person (the proband, if given) carries both of, as a single candidate (`var7a + var7b` above). The gene is then a locus with four kinds of copy: neither variant, one, the other, or both on the same copy, in cis. The family's genotypes show whether two variants are on different copies, as a compound heterozygote's are, or passed down together on one copy, which explains nothing: `var8a + var8b` in `examples/recessive.vcf` are such a pair, and rank last. Pairs need the variants' genes; `--no-compound` turns them off.

### Several families

`rank_variants_across`, and `pednet rank` on a PED file with several families, take the families to have the same condition, with the same mode of inheritance in every family, and combine each candidate's evidence over the families in which an affected person (the family's proband, if given) carries it. A family in which no affected person carries it counts only towards the mode of inheritance, since the condition may have another cause there. When the mode is given (`--modes AD`, say), the combined log10 Bayes factor is the sum of the families' log10 Bayes factors. Otherwise each mode is weighted by how well it explains all the families together, so the combined evidence is less than the sum when the families point to different modes: a family that looks dominant and one that looks recessive can't both be right. The table shows each family's own log10 Bayes factor beside the combined one.

## How the likelihoods work

Each person's genotype at the locus is a variable: the number of copies of the variant, 0, 1 or 2, and on the X chromosome 0 or 1 for a man. Someone of unknown sex who has no children can be either. Founders, the people with no parents in the pedigree, have Hardy-Weinberg proportions, and children have their parents' genotypes by Mendel's laws. Someone with the disease genotype is affected with probability equal to the penetrance, and someone without it with probability equal to the phenocopy rate (for XLR, women with one copy are affected at the phenocopy rate too, unless `carrier_penetrance` says otherwise). A genotype call is one copy out with probability 1 in 10,000 (`--error-rate`), and two copies out with that probability squared.

This makes the pedigree graph a Bayesian network, with an edge from each parent's genotype to each child's, and the likelihood of what was seen in the family is a sum over every assignment of genotypes to everyone. PedNet works the sum out exactly by variable elimination on the moral graph, as Superlink does for linkage (Fishelson and Geiger, 2002): it sums the genotypes out one at a time, in an order chosen greedily to add as few new edges to the moral graph as it can, which keeps the intermediate tables small. That is exact for pedigrees with loops as for any other, at a cost that grows with the moral graph's treewidth, how tangled its loops are, rather than with the size of the family. Many likelihoods are worked out at once, for many penetrances, probands or variants together.

**Ascertainment.** A family comes to notice through an affected person, the proband, so PedNet conditions on the proband having the disease genotype. Without a proband given, it averages over each affected person as the proband. This allows for how the family was found, and makes the answers depend very little on how common the disease allele is.

**The penetrance.** It is rarely known well, so by default PedNet averages the likelihood over penetrances of 0.6, 0.8, 0.95 and 0.99, equally weighted, with a phenocopy rate of 0.001. Give `--penetrance` and `--phenocopy` when they are known.

**The Bayes factor** for a variant is the probability of the phenotypes given the genotypes, with the variant as the cause, over their probability with an unseen locus as the cause, carried by the proband: the full-likelihood Bayes factor of Thompson et al. (2003), which segregatr also computes. The variant's founders carry it at its allele frequency, but at no less than 0.001 (`--min-frequency`): a variant seen in the family is there however rare it is elsewhere, rarer frequencies change little, and much rarer ones would make a calling error seem a likelier explanation than a real carrier.

**Checks.** The tests compare the likelihoods with brute-force sums over every assignment of genotypes, for pedigrees with loops, on the X and for pairs of variants, and the kinship coefficients with simulations of genes dropped down the pedigree. On segregatr's documented example (four affected brothers, penetrance 0.9, phenocopy rate 0.01, allele frequency 0.0001), PedNet's likelihoods give segregatr's Bayes factor of 7.732161 exactly when conditioned as segregatr conditions, on the proband's genotype call. PedNet's own Bayes factor, conditioned instead on the proband having the disease genotype (one copy or two) so that every variant is judged against the same alternative, is 7.731 with no genotyping errors and a smallest allele frequency of 0.0001, and 7.728 with its defaults.

## How well it works

[benchmarks/RESULTS.md](https://github.com/comparativechrono/PedNet/blob/main/benchmarks/RESULTS.md) compares the methods on simulated families of three generations (in a quarter of them four, with a first-cousin marriage), with a penetrance between 0.6 and 1, unknown phenotypes and half the relatives sequenced. The likelihood isn't told the penetrance simulated, but averages over its defaults. For 200 families of each mode, the share in which each method called the condition right, and in brackets the share it called wrongly; it left the rest uncertain, with no answer reaching a probability of 0.8:

| | AD | AR | XLR | XLD |
|---|---|---|---|---|
| Dominant or recessive, by the likelihood | 0.78 (0.04) | 0.86 (0.01) | 0.96 (0.00) | 0.73 (0.07) |
| Dominant or recessive, by the pattern classifier | 0.83 (0.04) | 0.93 (0.01) | 0.97 (0.01) | 0.90 (0.04) |
| AD or AR, by the likelihood | 0.94 (0.04) | 0.94 (0.01) | | |
| AD or AR, by the pattern classifier | 0.82 (0.03) | 0.97 (0.01) | | |
| AD or AR, by the original metric classifier | 0.64 (0.13) | 0.99 (0.01) | | |

On families like those it was trained on, the pattern classifier tells dominant from recessive more often than the likelihood does, and as rarely wrongly; choosing between AD and AR alone, the likelihood calls more of the AD families. The original classifier can never call AD when an affected child has unaffected parents, as often happens with incomplete penetrance, so it does well on AR families but leaves many AD families uncertain or calls them AR.

The benchmark also tries every method on families unlike those the classifiers learnt from, with a penetrance between 0.3 and 0.6, a phenocopy rate of 0.01 (twice the highest the classifiers were trained with, and ten times the likelihood's default) and a quarter of phenotypes unknown. All do worse there. Telling dominant from recessive, the pattern classifier was right in 56–79% of families and wrong in 4–17%, and the likelihood, with its default penetrances, right in 48–74% and wrong in 4–18%. Told the penetrance and phenocopy rate, as it can be when they are known, the likelihood was wrong in only 2–5% of these families, and uncertain about many more, as it should be with so little to go on. The classifiers can't be told. Even these families come from the same simulator as the training families, with the condition always entering through the first couple, one pattern of consanguinity, and no half-siblings or people of unknown sex, so the classifiers would probably do somewhat worse on real families than here.

Ranking the candidate variants, the mean rank of the causal variant among 21, with the mode of inheritance known as the original network score needs it, and in brackets the share of families in which it ranked first:

| | AD | AR | XLR | XLD |
|---|---|---|---|---|
| Co-segregation Bayes factor | 2.85 (0.39) | 1.06 (0.94) | 4.59 (0.22) | 4.16 (0.23) |
| Concordance of genotypes with phenotypes | 3.46 (0.23) | 1.25 (0.90) | 5.67 (0.10) | 4.88 (0.14) |
| Original network score | 9.81 (0.01) | 6.71 (0.05) | | |

The causal variant often can't be told from a background variant passed down with it: in 66% of the AD families, and in 72% and 81% of the X-linked ones, some background variant had exactly the causal variant's genotypes in the people sequenced. With the mode unknown, the Bayes factor's mean ranks were 2.85, 1.45, 4.71 and 4.23. For compound heterozygotes, among 32 single variants and 6 pairs, the causal pair had a mean rank of 2.5 with the mode unknown and 1.7 with it known, and ranked first in 51% and 54% of families; ranking single variants only, the better of the two causal variants had a mean rank of 6.5.

```bash
python benchmarks/benchmark.py --families 200 --seed 1
```

## Where the methods come from

None of PedNet's methods is new on its own. What PedNet does is put them together on one graph of the pedigree, for the PED and VCF files that clinical and research genetics use.

- **The structure of the family** is classical graph theory and population genetics: kinship by Wright's path coefficients (1922) and Malécot's coefficient (1948), loops as a cycle basis (Paton's algorithm, which networkx uses), the width by Dilworth's theorem (1950), and centrality as networkx works it out (betweenness after Freeman, 1977).
- **The network metrics, the original classifier and the network score** are the original PedNet's, by Nick Taylor.
- **The pattern features** are the textbook signs of each mode of inheritance. Reading them off the graph as features and training a classifier on simulated families is PedNet's own, but it is a simple and open approach rather than a new one: the classifier is a logistic regression whose weights can be read in `src/pednet/_trained.py`.
- **The likelihoods** are those of classical pedigree analysis, which worked them out by peeling, summing the genotypes out a person or a family at a time (Elston and Stewart, 1971; Cannings, Thompson and Skolnick, 1978). With the pedigree seen as a Bayesian network (Lauritzen and Sheehan, 2003), that is variable elimination on its moral graph, the graph Lauritzen and Spiegelhalter (1988) made the basis of exact calculations in such networks, as Superlink does it for linkage (Fishelson and Geiger, 2002). Comparing modes of inheritance by their likelihoods is the idea of segregation analysis, and the co-segregation Bayes factor is that of Thompson et al. (2003), with the strengths of evidence of Tavtigian et al. (2018) and Belman et al. (2020). Scoring a pair of variants in one gene as one locus with four kinds of copy is a small extension of the same model.

## What it doesn't do

- **One locus.** The condition is taken to come from one locus with one mode of inheritance, without polygenic background or modifiers. Variants are scored one at a time, or two in a gene; three or more variants together are not.
- **New mutations.** A variant new in the proband isn't modelled, and it scores against itself, since it doesn't follow the condition back through the family. Such variants are flagged as possibly de novo, to be judged on their own (PS2 or PM6, not PP1).
- **Age and sex.** The penetrance is the same for everyone with the disease genotype. Relatives too young to show a condition of later onset should be given an unknown phenotype (0) rather than recorded as unaffected.
- **Other modes.** Mitochondrial, Y-linked, imprinted and sex-limited inheritance aren't among the modes. A variant in a pseudoautosomal region of the X is inherited like an autosomal one, so give it an autosomal chromosome name.
- **Linkage.** A variant passed down with the causal one, near it on the chromosome, follows the condition just as well, and only its allele frequency can set it apart. PedNet ranks variants, not regions.
- **Genotype quality.** Calls are taken as they are, with the same small error rate for all, rather than with their read-level likelihoods.
- **Very tangled pedigrees.** A pedigree with many interlocking loops can need tables too large for exact likelihoods; PedNet then says so.
- **Families unlike the simulated ones.** The classifiers learnt from simulated families, and know nothing of a family's condition beyond its pattern; their answers for families far from those, much larger ones, say, or with a very low penetrance, deserve less trust than the likelihood's.

PedNet doesn't draw pedigrees: [Pedigree Studio](https://github.com/comparativechrono/pedigreestudio) does.

## The original PedNet

PedNet began as Nick Taylor's notebooks and modules, which treat the pedigree as a directed graph and use measures of it to classify the mode of inheritance and to score variants. Those methods are now part of the package, in `pednet.features` and `pednet.graph`, rewritten to need only NumPy and networkx. The original modules are kept in [legacy](https://github.com/comparativechrono/PedNet/tree/main/legacy), with their own [README](https://github.com/comparativechrono/PedNet/blob/main/legacy/README.md), and the tests check that the package gives the same numbers as they do. They aren't part of the package on PyPI.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests that compare the package with the original modules run too when the modules' requirements (`legacy/requirements.txt`) are installed, and are skipped otherwise.

## References

- Belman S, Parsons MT, Spurdle AB, Goldgar DE, Feng BJ. Considerations in assessing germline variant pathogenicity using cosegregation analysis. *Genetics in Medicine* 2020;22:2052–2059.
- Cannings C, Thompson EA, Skolnick MH. Probability functions on complex pedigrees. *Advances in Applied Probability* 1978;10:26–61.
- Dilworth RP. A decomposition theorem for partially ordered sets. *Annals of Mathematics* 1950;51:161–166.
- Elston RC, Stewart J. A general model for the genetic analysis of pedigree data. *Human Heredity* 1971;21:523–542.
- Fishelson M, Geiger D. Exact genetic linkage computations for general pedigrees. *Bioinformatics* 2002;18 Suppl 1:S189–S198.
- Freeman LC. A set of measures of centrality based on betweenness. *Sociometry* 1977;40:35–41.
- Hagberg AA, Schult DA, Swart PJ. Exploring network structure, dynamics, and function using NetworkX. *Proceedings of the 7th Python in Science Conference* 2008:11–15.
- Jarvik GP, Browning BL. Consideration of cosegregation in the pathogenicity classification of genomic variants. *American Journal of Human Genetics* 2016;98:1077–1081.
- Lauritzen SL, Sheehan NA. Graphical models for genetic analyses. *Statistical Science* 2003;18:489–514.
- Lauritzen SL, Spiegelhalter DJ. Local computations with probabilities on graphical structures and their application to expert systems. *Journal of the Royal Statistical Society, Series B* 1988;50:157–224.
- Malécot G. *Les mathématiques de l'hérédité.* Paris: Masson, 1948.
- Paton K. An algorithm for finding a fundamental set of cycles of a graph. *Communications of the ACM* 1969;12:514–518.
- Richards S, Aziz N, Bale S, et al. Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the American College of Medical Genetics and Genomics and the Association for Molecular Pathology. *Genetics in Medicine* 2015;17:405–424.
- Tavtigian SV, Greenblatt MS, Harrison SM, et al. Modeling the ACMG/AMP variant classification guidelines as a Bayesian classification framework. *Genetics in Medicine* 2018;20:1054–1060.
- Thompson D, Easton DF, Goldgar DE. A full-likelihood method for the evaluation of causality of sequence variants from family data. *American Journal of Human Genetics* 2003;73:652–655.
- Wright S. Coefficients of inbreeding and relationship. *American Naturalist* 1922;56:330–338.
- segregatr: segregation analysis for variant interpretation, an R package by Magnus Dehli Vigeland and Christian Carrizosa (version 0.5.0, on CRAN).

## Licence

MIT. PedNet is by Tim Hearn and Nick Taylor.
