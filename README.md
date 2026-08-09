# Greedy-Gnorm: A Gradient Matrix Norm-Based Alternative to Attention Entropy for Head Pruning

This repository contains computer code for reproducing the results described in the manuscript “Greedy-Gnorm: A Gradient Matrix Norm-Based Alternative to Attention Entropy for Head Pruning”.

ArXiv preprint link: https://arxiv.org/abs/2602.04491

## Table of Contents

[Getting Started](#getting-started)  
&nbsp;&nbsp;↳ [Environment Setup](#environment-setup)  
&nbsp;&nbsp;↳ [CPU / GPU Execution](#cpu--gpu-execution)

[Reproducing Plots and Tables](#reproducing-plots-and-tables)  
&nbsp;&nbsp;↳ [Data Preparation](#data-preparation)  
&nbsp;&nbsp;↳ [Gradient Analysis and Diagnostics](#gradient-analysis-and-diagnostics)  
&nbsp;&nbsp;↳ [Greedy Gnorm Based Pruning](#greedy-gnorm-based-pruning)  
&nbsp;&nbsp;↳ [Baseline and Comparative Pruning Methods](#baseline-and-comparative-pruning-methods)  
&nbsp;&nbsp;↳ [Multi-Model Experiments](#multi-model-experiments)  
&nbsp;&nbsp;↳ [Custom and Numerical Stability Experiments](#custom-and-numerical-stability-experiments)  
&nbsp;&nbsp;↳ [Result Aggregation and Visualization](#result-aggregation-and-visualization)

[Outputs](#outputs)  
&nbsp;&nbsp;↳ [Random Pruning Results](#random-pruning-results-excel-files)  
&nbsp;&nbsp;↳ [Accuracy vs. Pruned Heads](#accuracy-vs-pruned-heads-csv-files)  
&nbsp;&nbsp;↳ [Notes on Reproducibility](#notes-on-reproducibility)

[Figures and Visualizations](#figures-and-visualizations)  
&nbsp;&nbsp;↳ [Illustrative Figures](#illustrative-figures-manuscript-diagrams)  
&nbsp;&nbsp;↳ [Experimental Result Visualizations](#experimental-result-visualizations)

[Questions and Feedback](#questions-and-feedback)

[Citation](#citation)

---

## Getting Started

Clone this repository and change into the project directory:

```bash

git clone https://github.com/Sheridan-Stable/greedy-gnorm.git
cd greedy-gnorm

```

All experiments are implemented in Jupyter notebooks (`.ipynb`).

We recommend using **Python 3.9+**.

### Environment Setup

```bash
conda create -n prune-test python=3.9 -y
conda activate prune-test
```

#### CPU / GPU Execution

This implementation supports both CPU and GPU execution.

#### GPU Execution (Recommended)

If a CUDA-enabled GPU is available, we strongly recommend installing the GPU version of PyTorch and running the notebooks on GPU.

GPU execution significantly reduces runtime, as the pruning procedure involves repeated forward and backward passes through the model.

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install transformers==4.36.2 datasets==2.16.1 textpruner==1.1.post2 tqdm==4.66.1 matplotlib==3.8.2 pandas==2.1.4
```

Alternatively, core dependencies are listed in `requirements.txt`.

---

## Reproducing Plots and Tables

All experiments in this repository are implemented as **Jupyter notebooks (`.ipynb`)**.
Each notebook can be executed independently unless otherwise specified.

Unless noted otherwise, notebooks are expected to be run from the repository root directory.

### Data Preparation

#### `data_processing.ipynb` — Dataset Inspection

This notebook is used to **quickly inspect the raw dataset format** before connecting it to model inputs.

### Gradient Analysis and Diagnostics

#### `gradient_test.ipynb` — Gradient Behavior Analysis for Attention Head Pruning

This notebook investigates the **gradient behavior of attention head parameters** in BERT during pruning, focusing on how **gradients change after individual heads are removed**.

### Greedy Gnorm Based Pruning

#### `greedy_pruning.ipynb` — Greedy Gradient-Norm Pruning (Prototype)

This notebook provides an **early exploratory implementation** of greedy attention head pruning based on **Q/K/V gradient norms**, primarily used for **method validation and debugging**, rather than final experimental reporting.

#### `greedyprune(BERT,ALBERT,ROBERTA,XLM-ROBERTA).ipynb` — Greedy Gradient-Norm-Based Pruning

This notebook implements a **greedy attention head pruning algorithm based on gradient norm statistics**, where attention heads are iteratively removed according to their estimated contribution to the model output.

**Purpose**
Perform **greedy attention head pruning** using a **QKV gradient-norm–based importance score**, evaluated after each pruning step.

**Method Overview**

At each pruning iteration:

1. Compute per-head gradient norms for **Query (Q), Key (K), and Value (V)** projection matrices.
2. Aggregate head importance as the **product of Q, K, and V gradient norms**, followed by normalization.
3. Select the **least important remaining attention head** in a greedy manner.
4. Prune the selected head using a structured attention head mask.
5. Re-evaluate downstream task accuracy after pruning.

To correctly handle **previously pruned heads**, the implementation expands reduced-weight tensors back to the original dimensionality by inserting zero blocks, ensuring consistent per-head norm estimation throughout the pruning process.

**Supported Models**

- BERT
- ALBERT
- RoBERTa
- XLM-RoBERTa

Each model uses its corresponding pretrained checkpoint and task-specific dataset.

**Usage**

```text
Open and run all cells in the notebook sequentially.
```

The pruning procedure is fully self-contained and does not rely on precomputed importance scores.

**Results**

The notebook produces the following output files:

- `pruned_heads_accuracy(BERT)basedonGreedyGnorm.csv`
- `pruned_heads_accuracy(ALBERT)basedonGreedyGnorm.csv`
- `pruned_heads_accuracy(ROBERTA)basedonGreedyGnorm.csv`
- `pruned_heads_accuracy(XLM_ROBERTA)basedonGreedyGnorm.csv`

Each CSV file records downstream task accuracy as a function of the number of pruned attention heads under greedy gradient-norm–based pruning.

#### `InverseGreedyGnormPruning.ipynb` — Inverse Greedy-Gnorm Pruning

This notebook implements **inverse greedy gradient-norm–based attention head pruning** for Transformer models, where attention heads with **larger G-norm scores are pruned first**. Evaluate inverse greedy pruning behavior based on gradient-norm criteria and analyze worst-case pruning trajectories.

**Results**

- `pruned_heads_accuracy(BERT)basedonInverseGreedyGnorm.csv`
- `pruned_heads_accuracy(ALBERT)basedonInverseGreedyGnorm.csv`
- `pruned_heads_accuracy(ROBERTA)basedonInverseGreedyGnorm.csv`
- `pruned_heads_accuracy(XLM_ROBERTA)basedonInverseGreedyGnorm.csv`

Each file records downstream accuracy as a function of the number of pruned attention heads under **inverse greedy G-norm–based pruning**.

### Baseline and Comparative Pruning Methods

#### `random_pruning.ipynb` — Random Attention Head Pruning

This notebook implements **random attention head pruning** for Transformer models, where attention heads are removed **uniformly at random** at each pruning step. Provide a random pruning baseline and evaluate downstream performance degradation.

**Results**

- `BERTrandompruned_heads_data_50x25_1group.xlsx`
- `BERTrandompruned_heads_data_50x25_2group.xlsx`
- `BERTrandompruned_heads_data_50x25_concated.xlsx`
- `ALBERTrandompruned_heads_data_25x13_1group.xlsx`
- `ROBERTArandompruned_heads_data_25x25_1group.xlsx`
- `XLM_ROBERTArandompruned_heads_data_25x25_1group.xlsx`

Each file records downstream accuracy under **randomly selected attention head pruning**, used as a baseline for comparison with structured pruning methods.

#### `AE_pruning.ipynb` — Attention Entropy Pruning

This notebook implements **Attention Entropy (AE)–based attention head pruning** for Transformer models, where heads with higher AE scores are pruned first. Perform AE-based greedy pruning and evaluate downstream performance degradation.

**Results**

- `pruned_heads_accuracy(BERT)basedonAE.csv`
- `pruned_heads_accuracy(ALBERT)basedonAE.csv`
- `pruned_heads_accuracy(ROBERTA)basedonAE.csv`
- `pruned_heads_accuracy(XLM_ROBERTA)basedonAE.csv`

Each file records downstream accuracy as a function of the number of pruned attention heads.

#### `AE_inverse_pruning.ipynb` — Attention Entropy Inverse Pruning

This notebook implements **inverse Attention Entropy (AE)–based attention head pruning** for Transformer models, where heads are removed in ascending order of their AE scores. Perform inverse pruning based on Attention Entropy.

**Results**

- `pruned_heads_accuracy(BERT)basedonAEinverse.csv`
- `pruned_heads_accuracy(ALBERT)basedonAEinverse.csv`
- `pruned_heads_accuracy(ROBERTA)basedonAEinverse.csv`
- `pruned_heads_accuracy(XLM_ROBERTA)basedonAEinverse.csv`

Each file records downstream accuracy as a function of the number of pruned attention heads.

### Multi-Model Experiments

#### `four_models.ipynb` — Multi-Model Attention Head Pruning & Attention Entropy Analysis

This notebook conducts **attention head and FFN pruning experiments** across four Transformer models (BERT, ALBERT, RoBERTa, XLM-RoBERTa) and **collects attention entropy (AE) statistics** from their attention matrices.

### Custom and Numerical Stability Experiments

#### `customBERTpruning.ipynb` — Custom BERT Implementation Test

This notebook experiments with a **manually implemented BERT architecture** and compares it against the official HuggingFace BERT model. Test whether a custom-written BERT implementation can correctly load pretrained weights and interface with existing pruning utilities.

#### `solve_underflow.ipynb` — Attention Entropy Numerical Stability Diagnostics

This notebook **diagnoses and fixes numerical underflow and NaN issues** arising during Attention Entropy (AE) computation from Transformer attention matrices. Detect NaN / Inf values during AE computation and generate numerically stable AE matrices for downstream pruning.

**ALBERT**

- `ALBERT0_499AE.pt`
- `ALBERT500_999AE.pt`
- `ALBERT1000_1499AE.pt`
- `ALBERT1500_1999AE.pt`
- `ALBERT2000_2499AE.pt`
- `ALBERT2500_2999AE.pt`
- `ALBERT3000_3499AE.pt`
- `ALBERT3500_3999AE.pt`
- `ALBERT4000_4499AE.pt`
- `ALBERT4500_4999AE.pt`
- `ALBERT5000_5499AE.pt`
- `ALBERT5500_5999AE.pt`
- `ALBERT6000_6499AE.pt`
- `ALBERT6500_6999AE.pt`
- `ALBERT7000_7499AE.pt`
- `ALBERT7500_7999AE.pt`
- `ALBERT8000_8499AE.pt`
- `ALBERT8500_8999AE.pt`
- `ALBERT9000_9499AE.pt`
- `ALBERT9500_9814AE.pt`
- `ALBERT_9815AE.pt`

**BERT**

- `BERT_1029AE.pt`
- `BERT_1300AE.pt`
- `BERT_2329AE.pt`

**RoBERTa**

- `ROBERTA0_499AE.pt`
- `ROBERTA500_999AE.pt`
- `ROBERTA1000_1499AE.pt`
- `ROBERTA1500_1999AE.pt`
- `ROBERTA2000_2499AE.pt`
- `ROBERTA2500_2999AE.pt`
- `ROBERTA3000_3499AE.pt`
- `ROBERTA3500_3999AE.pt`
- `ROBERTA4000_4499AE.pt`
- `ROBERTA4500_4999AE.pt`
- `ROBERTA5000_5204AE.pt`
- `ROBERTA_5205AE.pt`

**XLM-RoBERTa**

- `XLM_ROBERTA0_999AE.pt`
- `XLM_ROBERTA1000_1999AE.pt`
- `XLM_ROBERTA2000_2999AE.pt`
- `XLM_ROBERTA3000_3999AE.pt`
- `XLM_ROBERTA4000_4999AE.pt`
- `XLM_ROBERTA5000_5999AE.pt`
- `XLM_ROBERTA6000_6999AE.pt`
- `XLM_ROBERTA7000_7722AE.pt`
- `XLM_ROBERTA7724_7999AE.pt`
- `XLM_ROBERTA8000_8999AE.pt`
- `XLM_ROBERTA9000_9999AE.pt`
- `XLM_ROBERTA_9999AE.pt`

Each file stores a **numerically stabilized AE matrix** used for head-ranking and pruning analysis.

> **Note:** `rectified` in the path denotes numerical stabilization of Attention Entropy using ε; one occurrence means ε is applied only to the log term, while two occurrences mean ε is applied to both the probability and log terms.

### Result Aggregation and Visualization

#### `show_pruning_result.ipynb` — Pruning Result Visualization and Comparison

This notebook visualizes and compares **attention head pruning trajectories** across different pruning strategies and Transformer models. The notebook reads precomputed accuracy files in CSV format.

**Results**

- Multi-panel pruning comparison figures (2×2 layout across models)
- Accuracy vs. pruned heads curves for each pruning strategy

#### `Pruning_summary.ipynb` — Final Pruning Configuration Summary

This notebook summarizes the **final pruning configurations** obtained from different pruning strategies and evaluates their **parameter reduction and accuracy retention**. Provide a consolidated view of **final attention head masks** and analyze their impact on:

**Results**

- Figures illustrating final attention head retention patterns
- Accuracy comparisons before and after pruning
- Model structure summaries reporting parameter and memory reductions

#### `greedy-gnorm-vis.ipynb` — Greedy Gnorm Pruning Visualization

This notebook visualizes and compares **attention head pruning behaviors** under Greedy Gnorm–based and baseline pruning strategies. Visualize pruning trajectories, accuracy degradation patterns, and final pruning configurations across different Transformer models.

**Results**

- `PruningResults.png` / `PruningResults.pdf`
- `RandomPruning.png` / `RandomPruning.pdf`
- `AlbertPruning.png` / `AlbertPruning.pdf`
- `allfinalsolutions.png` / `allfinalsolutions.pdf`
- `gradientpooling.png` / `gradientpooling.pdf`

## Outputs

This repository includes precomputed experimental results corresponding to the pruning experiments reported in the manuscript.
These files allow reproducing plots and tables **without rerunning the full pruning procedure**, which can be computationally expensive.

### Random Pruning Results (Excel Files)

The following `.xlsx` files contain results from **random attention head pruning experiments**, used as baselines for comparison.

**Files**

- `ALBERTrandompruned_heads_data_25x13_1group.xlsx`
- `BERTrandompruned_heads_data_50x25_1group.xlsx`
- `BERTrandompruned_heads_data_50x25_2group.xlsx`
- `BERTrandompruned_heads_data_50x25_concated.xlsx`
- `ROBERTArandompruned_heads_data_25x25_1group.xlsx`
- `XLM_ROBERTArandompruned_heads_data_25x25_1group.xlsx`

**Description**

- Each file records accuracy results under randomly selected attention head pruning.
- Different files correspond to different model architectures and experimental groupings.
- These results are used as **random pruning baselines** in comparative analyses.

**Usage**

- Loaded by analysis and visualization notebooks (e.g., `show_pruning_result.ipynb`)
- Used to compute averaged baseline performance curves

### Accuracy vs. Pruned Heads (CSV Files)

The following `.csv` files record **model accuracy as a function of the number of pruned attention heads**, under different pruning strategies.

Each file corresponds to a specific:

- Model architecture
- Pruning criterion

#### ALBERT

- `pruned_heads_accuracy(ALBERT)basedonAE.csv`
- `pruned_heads_accuracy(ALBERT)basedonAEinverse.csv`
- `pruned_heads_accuracy(ALBERT)basedonGreedyGnorm.csv`
- `pruned_heads_accuracy(ALBERT)basedonInverseGreedyGnorm.csv`

#### BERT

- `pruned_heads_accuracy(BERT)basedonAE.csv`
- `pruned_heads_accuracy(BERT)basedonAEinverse.csv`
- `pruned_heads_accuracy(BERT)basedonGreedyGnorm.csv`
- `pruned_heads_accuracy(BERT)basedonInverseGreedyGnorm.csv`

#### RoBERTa

- `pruned_heads_accuracy(ROBERTA)basedonAE.csv`
- `pruned_heads_accuracy(ROBERTA)basedonAEinverse.csv`
- `pruned_heads_accuracy(ROBERTA)basedonGreedyGnorm.csv`
- `pruned_heads_accuracy(ROBERTA)basedonInverseGreedyGnorm.csv`

#### XLM-RoBERTa

- `pruned_heads_accuracy(XLM_ROBERTA)basedonAE.csv`
- `pruned_heads_accuracy(XLM_ROBERTA)basedonAEinverse.csv`
- `pruned_heads_accuracy(XLM_ROBERTA)basedonGreedyGnorm.csv`
- `pruned_heads_accuracy(XLM_ROBERTA)basedonInverseGreedyGnorm.csv`

**Description**

- Each CSV file contains:
  - Number of pruned attention heads
  - Corresponding model accuracy

- Files are organized by **pruning strategy**:
  - Activation Energy (AE)
  - Inverse AE
  - Greedy Gnorm
  - Inverse Greedy Gnorm

**Usage**

- Used to generate accuracy–pruning curves
- Directly consumed by visualization and summary notebooks (e.g., `show_pruning_result.ipynb`, `Pruning_summary.ipynb`)

### Notes on Reproducibility

- The provided result files allow reproducing figures and tables **without rerunning full pruning experiments**.
- Full experiments can still be reproduced by executing the corresponding notebooks, but this may require substantial computation time, especially on CPU.

---

## Figures and Visualizations

This repository includes both **illustrative figures** used in the manuscript and **experimental result visualizations** generated from pruning experiments.

### Illustrative Figures (Manuscript Diagrams)

The following figures are **conceptual illustrations** included in the manuscript to explain the pruning methodology.
These figures were created using **draw.io** and are **not generated by code**.

**Files**

- `gradientpooling.pdf`
- `gradientpooling.png`
- `AlbertPruning.pdf`
- `AlbertPruning.png`

**Description**

- `gradientpooling.*`:
  Illustrates the gradient pooling mechanism used to compute importance scores for attention heads.
- `AlbertPruning.*`:
  Provides a schematic overview of attention head pruning in ALBERT.

**Notes**

- These figures are **static illustrations** for explanatory purposes.
- They are not produced by running the notebooks and are included directly for completeness and reference.

### Experimental Result Visualizations

The following figures visualize the **empirical results** of attention head pruning experiments.

**Files**

- `PruningResults.pdf`
- `PruningResults.png`
- `RandomPruning.pdf`
- `RandomPruning.png`
- `allfinalsolutions.pdf`
- `allfinalsolutions.png`

**Description**

- `PruningResults.*`:
  Accuracy trends under different pruning strategies.
- `RandomPruning.*`:
  Performance under random attention head pruning (baseline).
- `allfinalsolutions.*`:
  Aggregated comparison of pruning strategies across models.

**Usage**

- These figures are generated or reproduced using result files (`.csv`, `.xlsx`) and visualization notebooks such as:
  - `show_pruning_result.ipynb`
  - `Pruning_summary.ipynb`

### Notes on Reproducibility

- Illustrative figures (`gradientpooling.*`, `AlbertPruning.*`) are **manually designed diagrams** and are not reproduced by code.
- Experimental figures can be regenerated by running the corresponding notebooks, provided that the required result files or full pruning experiments are available.

## Questions and Feedback

If you have a technical question about the manuscript, feel free to post it as an [issue](https://github.com/Sheridan-Stable/greedy-gnorm/issues).

For more open-ended inquiries, we encourage starting a [discussion](https://github.com/Sheridan-Stable/greedy-gnorm/discussions).

## Citation

If you find anything useful please cite our work using:

```
@article{Guo2026,
  title={Greedy-Gnorm: A Gradient Matrix Norm-Based Alternative to Attention Entropy for Head Pruning},
  author={Guo, Yuxi and Ahmed, Zeyad and Sheridan, Paul and Farooque, Aitazaz A.},  journal={arXiv preprint arXiv:2602.04491},
  year={2026}
}
```
