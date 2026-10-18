# Benchmark

Compares PedNet's likelihood methods with the original ones (the graph-metric classifier of the mode of inheritance
and the network segregation score, in legacy/) and with a plain concordance score, on simulated families.

```bash
python benchmarks/benchmark.py [--families 200] [--seed 1] [--write benchmarks/RESULTS.md]
```

Each family is simulated with pednet.simulate: three generations (four for the consanguineous ones), at least two
affected people, a penetrance drawn evenly between 0.6 and 1, a phenocopy rate of 0.001, 5% of phenotypes unknown,
each relative sequenced with probability 0.5 (the proband always), and 20 rare background variants that the proband
carries, passed down the pedigree like real variants. The penetrance PedNet assumes isn't the one simulated: it
averages over its default range, as it would for a real family. The original methods need the packages in
legacy/requirements.txt.

PedNet 0.1.0, with its defaults unless the table says otherwise. Simulated families: 200 for each mode; seed 1.

## The mode of inheritance

The share of families each method calls correctly, wrongly, or leaves uncertain. The original classifier tells AD from AR only, and was trained on another 400 simulated AD and AR families. PedNet is shown choosing between AD and AR, as the original classifier does, and between all four modes. The last row for each mode is whether PedNet calls the condition dominant (AD or XLD) or recessive (AR or XLR), from the four modes' probabilities: often clear when the exact mode isn't. In the last column, the true mode (or kind) is the most probable one, whether or not the call is certain.

| Families | Method | Right | Wrong | Uncertain | Right when certain | True mode most probable |
|---|---|---|---|---|---|---|
| AD | original classifier | 0.70 | 0.11 | 0.18 | 0.87 | – |
| AD | PedNet, AD or AR | 0.94 | 0.04 | 0.02 | 0.96 | 0.96 |
| AD | PedNet, all four modes | 0.39 | 0.07 | 0.54 | 0.84 | 0.56 |
| AD | PedNet, dominant or recessive | 0.82 | 0.07 | 0.11 | 0.93 | 0.90 |
| AR | original classifier | 0.98 | 0.01 | 0.01 | 0.99 | – |
| AR | PedNet, AD or AR | 0.91 | 0.03 | 0.07 | 0.97 | 0.95 |
| AR | PedNet, all four modes | 0.39 | 0.01 | 0.60 | 0.97 | 0.89 |
| AR | PedNet, dominant or recessive | 0.81 | 0.02 | 0.17 | 0.98 | 0.98 |
| XLR | PedNet, all four modes | 0.34 | 0.02 | 0.64 | 0.95 | 0.89 |
| XLR | PedNet, dominant or recessive | 0.97 | 0.02 | 0.01 | 0.98 | 0.98 |
| XLD | PedNet, all four modes | 0.27 | 0.03 | 0.71 | 0.91 | 0.81 |
| XLD | PedNet, dominant or recessive | 0.78 | 0.05 | 0.17 | 0.94 | 0.92 |

How well PedNet's probabilities are calibrated (all four modes, every test family): the most probable mode's posterior probability against how often it was the true mode.

| Posterior of the most probable mode | Families | True mode |
|---|---|---|
| 0.25–0.50 | 99 | 0.55 |
| 0.50–0.70 | 264 | 0.67 |
| 0.70–0.80 | 133 | 0.89 |
| 0.80–0.90 | 98 | 0.87 |
| 0.90–0.99 | 144 | 0.92 |
| 0.99–1.00 | 62 | 0.97 |

## Ranking the candidate variants

Each family has the causal variant and 20 rare background variants carried by the proband. "Top" is the share of families in which the causal variant scored higher than every other variant (a tie for the top counts as a miss); the ranks share ties out. The original network score and the concordance score are given the true mode; PedNet is shown not knowing it, and knowing it. The "identical" row is the share of families in which no background variant has exactly the causal variant's genotypes in the people sequenced. Otherwise the two can't be told apart by their genotypes, and only their allele frequencies, which here say nothing about which is causal, can separate them.

| Families | Method | Top | Mean rank | Mean of 1/rank |
|---|---|---|---|---|
| AD | PedNet, mode unknown | 0.29 | 3.60 | 0.52 |
| AD | PedNet, mode known | 0.29 | 3.60 | 0.52 |
| AD | concordance, mode known | 0.16 | 4.20 | 0.44 |
| AD | original network score, mode known | 0.00 | 9.94 | 0.15 |
| AD | no background variant identical | 0.27 | | |
| AR | PedNet, mode unknown | 0.84 | 1.51 | 0.89 |
| AR | PedNet, mode known | 0.93 | 1.08 | 0.96 |
| AR | concordance, mode known | 0.79 | 1.40 | 0.91 |
| AR | original network score, mode known | 0.06 | 6.48 | 0.26 |
| AR | no background variant identical | 0.97 | | |
| XLR | PedNet, mode unknown | 0.17 | 4.58 | 0.40 |
| XLR | PedNet, mode known | 0.17 | 4.44 | 0.41 |
| XLR | concordance, mode known | 0.07 | 5.87 | 0.29 |
| XLR | no background variant identical | 0.21 | | |
| XLD | PedNet, mode unknown | 0.19 | 4.91 | 0.42 |
| XLD | PedNet, mode known | 0.19 | 4.88 | 0.42 |
| XLD | concordance, mode known | 0.10 | 5.13 | 0.34 |
| XLD | no background variant identical | 0.14 | | |

## Compound heterozygotes

200 families with an autosomal recessive condition caused by two variants in one gene, each brought into the family by a different founder, simulated as above (three generations, at least two affected people). Besides the two causal variants, each family has 20 rare background variants and 5 background pairs of rare variants in one gene that the proband carries both of, passed down the pedigree as copies of the gene, so that both variants of a pair can be on one copy. PedNet ranks the pairs as compound heterozygotes alongside the single variants; the causal pair is the target. Without pairs, only the single variants can be ranked, and the row for that gives the better rank of the two causal variants. The concordance score here counts a person as having the disease genotype when they carry both variants of a pair (or two copies of either).

| Method | Top | Mean rank | Mean of 1/rank |
|---|---|---|---|
| PedNet, mode unknown | 0.54 | 2.73 | 0.71 |
| PedNet, mode known | 0.57 | 1.68 | 0.75 |
| PedNet, single variants only | 0.04 | 6.59 | 0.26 |
| concordance, mode known | 0.22 | 2.41 | 0.59 |
| no background pair identical | 0.82 | | |

Took 34 seconds.
