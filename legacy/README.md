# The original PedNet

This folder keeps the original PedNet, which Nick Taylor wrote from the notebooks in `notebooks/`. It treats a pedigree as a directed graph, with an edge from each parent to each child, and uses measures of the graph to do two things:

- **classify the mode of inheritance** of a pedigree, autosomal dominant (AD) or autosomal recessive (AR), from how the affected people are spread through it
- **score variants by how well they segregate** with the condition in the family, so that the variant most likely to cause it comes top

The PedNet package (in `src/pednet`) has these methods too, rewritten without the original's other requirements: the four network metrics, the classifier of the mode of inheritance and the network score of a variant are in `pednet.features`, and the graph measures in `pednet.graph`. The tests check that they give exactly the same numbers as the code here. The package adds a classifier of all four modes from the pattern of affected people, and the likelihood methods, which treat the pedigree's graph as a Bayesian network; `benchmarks/benchmark.py` compares them all. The original modules are kept so that the results already obtained with them can be reproduced. They aren't part of the package on PyPI.

The pedigrees they are benchmarked on are made by their own pedigree generator, an earlier form of [PedGen](https://github.com/comparativechrono/pedigree-generator), and the pipeline reads pedigree images with an earlier form of [PedInterp](https://github.com/comparativechrono/pedigree-interpreter).

## Modules

| Module | What it does |
|---|---|
| `PedigreeDataGeneration.py` | generates pedigrees by Mendelian inheritance, and variant tables with one variant linked to the condition and a background of unlinked ones |
| `PedigreeDAGAnalysis.py` | reads PED files, builds the pedigree graph, works out each person's generation, and computes the graph metrics and the consanguinity analysis |
| `ModeClassifierMain.py` | finds a threshold for each metric from ROC curves on generated pedigrees (by Youden's J), and classifies a pedigree as AD, AR or uncertain by a vote of the metrics |
| `ModeClassifierBenchmark.py`, `ModeClassifierStats.py` | benchmark the classifier over many trials for pedigrees of 3, 4 and 5 generations, and compare the sizes statistically |
| `SegScoreMain.py` | scores each variant from its edge consistency, generation continuity and carrier betweenness (and, in the extended score, founder influence and depth), and optimises the weights on generated pedigrees |
| `SegScoreBenchmark.py`, `SegScoreStats.py` | benchmark the scores over many trials, with default and optimised weights, and compare them statistically |
| `CVPedigreeAnalysis.py` | reads a pedigree image into a PED file. PedInterp is the later, packaged version, and has many fixes this copy doesn't |
| `PedNetPipeline.py` | runs everything on real pedigrees: images to PED files, classification, then scoring of the known variants among simulated background variants |

## Running

The modules import each other by name, so run them from this folder, after installing what they need:

```bash
cd legacy
pip install -r requirements.txt
python ModeClassifierBenchmark.py   # data/MoI_Benchmarking_Results/
python ModeClassifierStats.py       # statistics and plots from those results
python SegScoreBenchmark.py         # data/Segregation_Scoring_Trial_Results/
python SegScoreStats.py
```

Reading pedigree images also needs the [Tesseract OCR engine](https://tesseract-ocr.github.io/tessdoc/Installation.html). `plot_pedigree_tree` draws with Graphviz's `dot` layout when PyGraphviz is installed, and with a spring layout otherwise.

The pipeline reads its input from `data/` in this folder: pedigree images in `data/Pedigree_Images/Raw_Images/`, and the families' known genotypes in `data/ProbandInfo/`. That data isn't in the repository, and `.gitignore` keeps any `data/` folder out of it, since it may hold patients' data.

```bash
python PedNetPipeline.py
```

## Background variants

`simulate_variant_table`, `PedGraph_VarTable_generator`, `VarTableBackpadding` and `trial_based_segregation_scoring_weight_optimization` take `inherited_background`. By default (`False`, as in the original results) each person's genotype for a background variant is drawn independently from the population. Real variants are inherited, so relatives share them. With `inherited_background=True`, the background variants are passed down the pedigree by Mendel's laws, like the linked variant. That makes the linked variant harder to pick out, and it's the more realistic test.

## Ranks and ties

When the linked variant scores the same as an unlinked one, the rank is shared between them: a tie with one other variant for the top score gives a rank of 1.5, the linked variant's expected rank if ties were broken at random. In the benchmarks, a hit at 1 (`Top1`) means that the linked variant scored higher than every other variant: a tie for the top score counts as a miss. The inverse rank and DCG are worked out from the shared rank.

## Tests

The tests of these modules are in `tests/test_legacy.py`, and run with the rest when the requirements are installed:

```bash
pip install -r legacy/requirements.txt pytest
pytest tests/test_legacy.py
```
