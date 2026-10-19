# Benchmark

Compares PedNet's ways of reading a pedigree graph on simulated families: for the mode of inheritance, the original
PedNet's network metric classifier, the pattern classifier and the likelihood of the graph as a Bayesian network;
for ranking candidate variants, the original network score, a plain concordance score and the Bayes factor.

```bash
python benchmarks/benchmark.py [--families 200] [--seed 1] [--write benchmarks/RESULTS.md]
```

Each family is simulated with pednet.simulate: three generations, or in a quarter of the families of each mode four,
with a first-cousin marriage; at least two affected people, a penetrance drawn evenly between 0.6 and 1, a phenocopy
rate of 0.001, 5% of phenotypes unknown, each relative sequenced with probability 0.5 (the proband always), and 20
rare background variants that the proband carries, passed down the pedigree like real variants. Consanguinity is as
common whatever the mode, so that no method can take it alone as a sign of recessive inheritance, though it is in
fact commoner in families with a recessive condition. The likelihood methods don't know the penetrance simulated:
they average over their default range, as they would for a real family. The two classifiers were trained on other
families from the same simulator (benchmarks/train_classifiers.py, with another seed and over a wider range of
families), which favours them here, so the benchmark also tests all three on families unlike those they were trained
on.

PedNet 2.0.0, with its defaults unless the table says otherwise. Simulated families: 200 for each mode; seed 1.

## The mode of inheritance

The share of families each method calls correctly, wrongly, or leaves uncertain (when no mode reaches a probability of 0.8), and the share in which the true mode is the most probable, certain or not. The original metric classifier tells AD from AR only, by a vote of its network metrics; the other two are shown choosing between AD and AR as it does, and between all four modes. The rows for dominant or recessive call the condition dominant (AD or XLD) or recessive (AR or XLR) from the four modes' probabilities, which is often clear when the exact mode isn't.

| Families | Method | Right | Wrong | Uncertain | Right when certain | True mode most probable |
|---|---|---|---|---|---|---|
| AD | original metric classifier, AD or AR | 0.64 | 0.13 | 0.23 | 0.83 | – |
| AD | patterns, AD or AR | 0.82 | 0.03 | 0.15 | 0.97 | 0.95 |
| AD | likelihood, AD or AR | 0.94 | 0.04 | 0.03 | 0.96 | 0.95 |
| AD | patterns, all four modes | 0.34 | 0.06 | 0.61 | 0.86 | 0.58 |
| AD | likelihood, all four modes | 0.40 | 0.04 | 0.56 | 0.90 | 0.54 |
| AD | patterns, dominant or recessive | 0.83 | 0.04 | 0.13 | 0.96 | 0.94 |
| AD | likelihood, dominant or recessive | 0.78 | 0.04 | 0.18 | 0.95 | 0.90 |
| AR | original metric classifier, AD or AR | 0.99 | 0.01 | 0.01 | 0.99 | – |
| AR | patterns, AD or AR | 0.97 | 0.01 | 0.01 | 0.99 | 0.98 |
| AR | likelihood, AD or AR | 0.94 | 0.01 | 0.04 | 0.98 | 0.97 |
| AR | patterns, all four modes | 0.65 | 0.01 | 0.35 | 0.99 | 0.81 |
| AR | likelihood, all four modes | 0.37 | 0.01 | 0.62 | 0.97 | 0.82 |
| AR | patterns, dominant or recessive | 0.93 | 0.01 | 0.07 | 0.99 | 0.98 |
| AR | likelihood, dominant or recessive | 0.86 | 0.01 | 0.13 | 0.99 | 0.98 |
| XLR | patterns, all four modes | 0.61 | 0.01 | 0.39 | 0.99 | 0.90 |
| XLR | likelihood, all four modes | 0.40 | 0.01 | 0.59 | 0.99 | 0.87 |
| XLR | patterns, dominant or recessive | 0.97 | 0.01 | 0.01 | 0.99 | 0.99 |
| XLR | likelihood, dominant or recessive | 0.96 | 0.00 | 0.04 | 1.00 | 1.00 |
| XLD | patterns, all four modes | 0.23 | 0.04 | 0.72 | 0.84 | 0.74 |
| XLD | likelihood, all four modes | 0.18 | 0.07 | 0.76 | 0.73 | 0.73 |
| XLD | patterns, dominant or recessive | 0.90 | 0.04 | 0.06 | 0.95 | 0.93 |
| XLD | likelihood, dominant or recessive | 0.73 | 0.07 | 0.20 | 0.92 | 0.85 |

How well the probabilities are calibrated, over all the families above: the most probable mode's probability against how often it was the true mode.

| Probability of the most probable mode | Families (likelihood) | True mode (likelihood) | Families (patterns) | True mode (patterns) |
|---|---|---|---|---|
| 0.25–0.50 | 73 | 0.34 | 51 | 0.39 |
| 0.50–0.70 | 295 | 0.61 | 260 | 0.53 |
| 0.70–0.80 | 137 | 0.86 | 103 | 0.83 |
| 0.80–0.90 | 99 | 0.87 | 228 | 0.93 |
| 0.90–0.99 | 135 | 0.91 | 140 | 0.96 |
| 0.99–1.00 | 61 | 1.00 | 18 | 1.00 |

### Families unlike the training families

The same, for families of two to four generations unlike those the classifiers were trained on, and outside the likelihood methods' default range of penetrances: a penetrance drawn between 0.3 and 0.6, a phenocopy rate of 0.01, a quarter of phenotypes unknown, two to six children to a couple, more of whom have families of their own, and no consanguinity. A real family differs from any simulation in ways like these, and this shows how each method copes. These families still come from the same generator as the training families, though: the disease allele always enters through the first couple (and for a recessive condition, a second copy through someone who married in), consanguinity has one pattern, and there are no half-siblings or people of unknown sex, so even these results probably flatter the classifiers. The rows for the likelihood told the penetrance give it what would be known about such a condition: penetrances of 0.3, 0.45 and 0.6 to average over, and the phenocopy rate.

| Families | Method | Right | Wrong | Uncertain | Right when certain | True mode most probable |
|---|---|---|---|---|---|---|
| AD | original metric classifier, AD or AR | 0.21 | 0.48 | 0.30 | 0.30 | – |
| AD | patterns, AD or AR | 0.54 | 0.17 | 0.29 | 0.75 | 0.69 |
| AD | likelihood, AD or AR | 0.64 | 0.16 | 0.20 | 0.80 | 0.71 |
| AD | likelihood told the penetrance, AD or AR | 0.57 | 0.02 | 0.41 | 0.97 | 0.74 |
| AD | patterns, all four modes | 0.18 | 0.12 | 0.69 | 0.60 | 0.42 |
| AD | likelihood, all four modes | 0.30 | 0.10 | 0.59 | 0.75 | 0.41 |
| AD | likelihood told the penetrance, all four modes | 0.21 | 0.01 | 0.78 | 0.93 | 0.41 |
| AD | patterns, dominant or recessive | 0.56 | 0.17 | 0.27 | 0.76 | 0.69 |
| AD | likelihood, dominant or recessive | 0.48 | 0.18 | 0.34 | 0.73 | 0.65 |
| AD | likelihood told the penetrance, dominant or recessive | 0.41 | 0.05 | 0.54 | 0.89 | 0.68 |
| AR | original metric classifier, AD or AR | 0.96 | 0.03 | 0.01 | 0.97 | – |
| AR | patterns, AD or AR | 0.65 | 0.06 | 0.29 | 0.91 | 0.85 |
| AR | likelihood, AD or AR | 0.58 | 0.07 | 0.34 | 0.89 | 0.85 |
| AR | likelihood told the penetrance, AD or AR | 0.14 | 0.04 | 0.82 | 0.75 | 0.81 |
| AR | patterns, all four modes | 0.21 | 0.04 | 0.74 | 0.82 | 0.62 |
| AR | likelihood, all four modes | 0.18 | 0.04 | 0.78 | 0.80 | 0.65 |
| AR | likelihood told the penetrance, all four modes | 0.01 | 0.01 | 0.98 | 0.25 | 0.59 |
| AR | patterns, dominant or recessive | 0.58 | 0.07 | 0.34 | 0.89 | 0.81 |
| AR | likelihood, dominant or recessive | 0.49 | 0.05 | 0.46 | 0.91 | 0.84 |
| AR | likelihood told the penetrance, dominant or recessive | 0.08 | 0.03 | 0.89 | 0.73 | 0.74 |
| XLR | patterns, all four modes | 0.34 | 0.04 | 0.62 | 0.89 | 0.75 |
| XLR | likelihood, all four modes | 0.34 | 0.06 | 0.60 | 0.85 | 0.67 |
| XLR | likelihood told the penetrance, all four modes | 0.17 | 0.01 | 0.82 | 0.92 | 0.69 |
| XLR | patterns, dominant or recessive | 0.79 | 0.04 | 0.17 | 0.95 | 0.90 |
| XLR | likelihood, dominant or recessive | 0.74 | 0.04 | 0.22 | 0.96 | 0.92 |
| XLR | likelihood told the penetrance, dominant or recessive | 0.40 | 0.02 | 0.58 | 0.95 | 0.90 |
| XLD | patterns, all four modes | 0.14 | 0.07 | 0.79 | 0.67 | 0.54 |
| XLD | likelihood, all four modes | 0.19 | 0.13 | 0.68 | 0.59 | 0.51 |
| XLD | likelihood told the penetrance, all four modes | 0.15 | 0.03 | 0.82 | 0.86 | 0.60 |
| XLD | patterns, dominant or recessive | 0.64 | 0.12 | 0.24 | 0.84 | 0.76 |
| XLD | likelihood, dominant or recessive | 0.52 | 0.15 | 0.33 | 0.77 | 0.69 |
| XLD | likelihood told the penetrance, dominant or recessive | 0.45 | 0.03 | 0.53 | 0.95 | 0.74 |

## Ranking the candidate variants

Each family has the causal variant and 20 rare background variants carried by the proband. "Top" is the share of families in which the causal variant scored higher than every other variant (a tie for the top counts as a miss), and tied candidates share the average of their ranks. The original network score and the concordance score are given the true mode; PedNet's Bayes factor is shown not knowing it, and knowing it. The "identical" row is the share of families in which no background variant has exactly the causal variant's genotypes in the people sequenced. Otherwise the two can't be told apart by their genotypes, and only their allele frequencies, which here say nothing about which is causal, can separate them.

| Families | Method | Top | Mean rank | Mean of 1/rank |
|---|---|---|---|---|
| AD | Bayes factor, mode unknown | 0.39 | 2.85 | 0.62 |
| AD | Bayes factor, mode known | 0.39 | 2.85 | 0.62 |
| AD | concordance, mode known | 0.23 | 3.46 | 0.50 |
| AD | original network score, mode known | 0.01 | 9.81 | 0.16 |
| AD | no background variant identical | 0.34 | | |
| AR | Bayes factor, mode unknown | 0.89 | 1.45 | 0.93 |
| AR | Bayes factor, mode known | 0.94 | 1.06 | 0.97 |
| AR | concordance, mode known | 0.90 | 1.25 | 0.95 |
| AR | original network score, mode known | 0.05 | 6.71 | 0.25 |
| AR | no background variant identical | 0.98 | | |
| XLR | Bayes factor, mode unknown | 0.21 | 4.71 | 0.43 |
| XLR | Bayes factor, mode known | 0.22 | 4.59 | 0.44 |
| XLR | concordance, mode known | 0.10 | 5.67 | 0.31 |
| XLR | no background variant identical | 0.28 | | |
| XLD | Bayes factor, mode unknown | 0.23 | 4.23 | 0.47 |
| XLD | Bayes factor, mode known | 0.23 | 4.16 | 0.47 |
| XLD | concordance, mode known | 0.14 | 4.88 | 0.38 |
| XLD | no background variant identical | 0.19 | | |

## Compound heterozygotes

200 families with an autosomal recessive condition caused by two variants in one gene, each brought into the family by a different founder, simulated as above (three generations, at least two affected people). Besides the two causal variants, each family has 20 rare background variants and 5 background pairs of rare variants in one gene that the proband carries both of, passed down the pedigree as copies of the gene, so that both variants of a pair can be on one copy. PedNet ranks the pairs as compound heterozygotes alongside the single variants; the causal pair is the target. Without pairs, only the single variants can be ranked, and the row for that gives the better rank of the two causal variants. The concordance score here counts a person as having the disease genotype when they carry both variants of a pair (or two copies of either).

| Method | Top | Mean rank | Mean of 1/rank |
|---|---|---|---|
| Bayes factor, mode unknown | 0.51 | 2.52 | 0.69 |
| Bayes factor, mode known | 0.54 | 1.73 | 0.73 |
| Bayes factor, single variants only | 0.07 | 6.54 | 0.27 |
| concordance, mode known | 0.22 | 2.32 | 0.59 |
| no background pair identical | 0.85 | | |

Took 89 seconds.
