## Added

- `bert-glue.py`: for BERT on GLUE.
- `xlmroberta-glue.py`: for XLM-RoBERTa on GLUE.
- `albert-glue.py`: for ALBERT on GLUE.
- `roberta-glue.py`: for RoBERTa on GLUE
- `plot_glue.py`: for plotting the pruning trajectories for each model and a master overview plot.
- Pruning results (3 seeds per task per model):
  - 3x3=9 `experiments_results/glue/BERT_*/`
  - 3x3=9 `experiments_results/glue/ALBERT_*/`
  - 3x3=9 `experiments_results/glue/ROBERTA_*/`
  - 3x3=9 `experiments_results/glue/XLM_ROBERTA_*/`
  - `experiments_results/summary_pruning_report.csv`
