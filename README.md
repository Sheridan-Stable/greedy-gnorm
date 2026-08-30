# Greedy-Gnorm: A gradient matrix norm-based method for attention head pruning

This repository contains computer code for reproducing the results numerical described in the manuscript “Greedy-Gnorm: A gradient matrix norm-based method for attention head pruning” (currently under review with Machine Learning - Springer Nature).

---

## Quickstart

We recommend **Python 3.9+** and a **CUDA-enabled NVIDIA GPU** for running pruning and fine-tuning experiments.

```bash
git clone https://github.com/Sheridan-Stable/greedy-gnorm.git
cd greedy-gnorm

conda create -n prune python=3.9 -y
conda activate prune
pip install -r requirements.txt
```

---

## Running Experiments

All scripts run from the repository root. Pass `--seed <SEED>` to set random seeds (default: 555).

### 1. Dynamic Head Pruning (Greedy-Gnorm & Baselines)

Runs Greedy-Gnorm, Taylor Importance, and Self-Attention Attribution:

```bash
python bert-glue.py          # BERT (SST-2, MNLI, QNLI)
python albert-glue.py        # ALBERT (SST-2, MNLI, QNLI)
python roberta-glue.py       # RoBERTa (SST-2, MNLI, QNLI)
python xlmroberta-glue.py    # XLM-RoBERTa (SST-2, MNLI, QQP)
```

-> **Output:** `experiments_results/glue/`

### 2. Attention Entropy Baseline

```bash
python bert-ae-glue.py
python albert-ae-glue.py
python roberta-ae-glue.py
python xlmroberta-ae-glue.py
```

-> **Output:** `experiments_results/glue/AE/`

### 3. Static Pruning Baselines

One-shot importance scoring at step 0:

```bash
python static-glue/bert_static.py
python static-glue/albert_static.py
python static-glue/roberta_static.py
python static-glue/xlmroberta_static.py
```

-> **Output:** `static-glue/experiments_results/`

### 4. Post-Pruning Fine-Tuning

Evaluates recovery after fine-tuning (50%, 75%, 83.3% pruned heads):

```bash
python bert-glue-finetune-all.py
python roberta-glue-finetune-all.py
python xlmroberta-glue-finetune-all.py
```

-> **Output:** `experiments_results/finetune/`

### 5. Gnorm Aggregation Variants

Compares Product, Sum, and Max aggregation of Q/K/V gradient norms on BERT:

```bash
python ggnorm-variants.py
```

-> **Output:** `experiments_results/glue/`

---

## Generating Figures & Tables

Precomputed CSVs in `experiments_results/` allow reproducing all plots and LaTeX tables without re-running pruning.

### Figures

```bash
python plot_all_static_dynamic_glue.py   # Dynamic vs. Static pruning trajectories
python plot_ggnorm_variants.py           # Score aggregation variants comparison
python generate_sst2_final_solutions.py  # SST-2 head retention masks
```

-> **Output:** `figures/` (saved as `.pdf` and `.png`)

### Tables

```bash
python generate_glue_results_table.py            # Dynamic benchmark results
python static-glue/generate_static_glue_table.py # Static baseline comparison
python generate_finetune_table.py                # Post-pruning fine-tuning recovery
python generate_timing_table.py                  # Wall-clock execution timing
```

-> **Output:** Printed directly to `stdout`

---

## Questions and Feedback

Post technical questions as an [issue](https://github.com/Sheridan-Stable/greedy-gnorm/issues) or start a [discussion](https://github.com/Sheridan-Stable/greedy-gnorm/discussions).

---

## Citation

```bibtex
@article{Guo2026,
  title={Greedy-Gnorm: A Gradient Matrix Norm-Based Method for Attention Head Pruning},
  author={Guo, Yuxi and Ahmed, Zeyad and Sheridan, Paul and Farooque, Aitazaz A.},
  journal={xxxxx},
  year={2026}
}
```
