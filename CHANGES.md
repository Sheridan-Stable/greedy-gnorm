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

- We output 4+4 (png + pdf) figures in the `figures` folder:
  - `bert_pruning_trajectories.*`
  - `albert_pruning_trajectories.*`
  - `roberta_pruning_trajectories.*`
  - `xlm_roberta_pruning_trajectories.*`
- And a consolidated figure `glue_master_overview.*`

## Updated

- `*-glue.py` now logs time for pruning; output at `experiments_results/glue/*_*_timing_*.csv`
