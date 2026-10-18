# Changelog

## 0.1.0

The first release. PedNet becomes a package that analyses a pedigree as a directed acyclic graph, in three ways.
The original modules, which it grew from, are kept in `legacy/` and aren't part of the package.

The structure of the family, in `pednet.graph`:

- The pedigree as a networkx DAG, with the mating graph (a node for each couple) and the moral graph.
- The loops of a pedigree, from consanguinity or from families who married into each other twice.
- Kinship and inbreeding coefficients, by the recursion that sums Wright's path coefficients, for an autosome and
  for the X.
- Relationships named from their paths through the closest common ancestors (half-sister, first cousin once
  removed, double first cousin), with their degree.
- Network measures: centrality, the width of the pedigree, founder influence, and the elimination order and
  treewidth bound for exact likelihoods. `pednet graph` prints them, and writes the kinship matrix and GraphML.

The pattern of affected people, in `pednet.features`:

- The original PedNet's four network metrics, its classifier of AD against AR, and its network score of a variant,
  rewritten without its other requirements; the tests check that they give the same numbers as the original.
- Pattern features read off the graph: transmission down the generations, father to son, through unaffected women,
  skipped generations, affected children of related parents, and kinship on the X. A classifier of all four modes
  is built on them, trained on 4,000 simulated families of two to five generations
  (`benchmarks/train_classifiers.py`).

The graph as a Bayesian network of genotypes, in `pednet.inheritance` and `pednet.segregation`:

- Exact likelihoods by variable elimination on the moral graph, for pedigrees with loops as for any other.
- `infer_mode`: the posterior probability of AD, AR, XLR and XLD inheritance, given that the proband has the
  disease genotype and averaged over a range of penetrances, with the probability that the condition is dominant
  or recessive, and X-linked or autosomal. `infer_shared_mode` does the same for several families with one
  condition.
- `rank_variants`: candidate variants ranked by the full-likelihood co-segregation Bayes factor (Thompson et al.,
  2003), with its strength as PP1 or BS4 evidence by the thresholds of Tavtigian et al. (2018). Genotyping errors
  are allowed for, possible new mutations and Mendelian errors are flagged, and X-linked variants are scored under
  the X-linked modes.
- Pairs of variants in one gene are scored as compound heterozygotes, summing over whether the two are on one copy
  of the gene or on different copies.
- `rank_variants_across` combines several families, with the mode of inheritance shared between them.

Reading and the command line:

- PED files, VCF files (with the allele frequency from INFO, and the gene from VEP, SnpEff, bcftools csq, ANNOVAR or
  dbSNP annotations) and tables of genotypes. `pedigree_from_rows` reads the pedigrees of PedGen and PedInterp.
- The `pednet` command: `graph`, `mode` (with `--method` for the two classifiers), `rank` and `check`.
- `pednet.simulate` makes simulated families with background variants passed down the pedigree, and
  `benchmarks/benchmark.py` compares all the methods on them.
- Needs NumPy and networkx, on Python 3.9 or later.
