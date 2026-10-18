# Changelog

## 0.1.0

The first release, and the first as a package. PedNet now works out its answers from exact likelihoods of the
family's phenotypes and genotypes. They replace the original graph measures, which are kept in `legacy/` for
comparison and aren't part of the package.

- `infer_mode`: the posterior probability of autosomal dominant, autosomal recessive, X-linked recessive and X-linked
  dominant inheritance, given that the proband has the disease genotype and averaged over a range of penetrances,
  with the probability that the condition is dominant or recessive, and X-linked or autosomal.
  `infer_shared_mode` does the same for several families with one condition.
- `rank_variants`: candidate variants ranked by co-segregation, with the full-likelihood Bayes factor of Thompson et
  al. (2003) and its strength as PP1 or BS4 evidence by the thresholds of Tavtigian et al. (2018). Genotyping
  errors are allowed for, variants new in a child and Mendelian errors are flagged, and X-linked variants are scored
  under the X-linked modes. `rank_variants_across` adds up the evidence of several families.
- Pairs of variants in one gene are scored as compound heterozygotes, with the phase worked out from the family.
- Exact for pedigrees with loops, by variable elimination; people of unknown sex are allowed for on the X.
- Reads PED files, VCF files (with the allele frequency from INFO, and the gene from the annotations of VEP,
  SnpEff, bcftools csq, ANNOVAR or dbSNP) and tables of genotypes; `pedigree_from_rows` reads the pedigrees of PedGen
  and PedInterp.
- The `pednet` command: `check`, `mode` and `rank`.
- `pednet.simulate` makes simulated families with background variants passed down the pedigree, and
  `benchmarks/benchmark.py` compares the new methods with the original ones on them.
- Needs only NumPy, on Python 3.9 or later.
