"""
Plot Greedy-Gnorm score aggregation variants (Product, Sum, Max) on BERT for SST-2 and MNLI.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['legend.framealpha'] = 0.85
plt.rcParams['savefig.bbox'] = 'tight'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLUE_DIR = os.path.join(BASE_DIR, "experiments_results", "glue")

VARIANTS_META = {
    "Accuracy_Product": {"label": "Greedy-Gnorm (Product)", "style": "-", "color": "#1f77b4"},
    "Accuracy_Sum": {"label": "Greedy-Gnorm (Sum)", "style": "-", "color": "#ff7f0e"},
    "Accuracy_Max": {"label": "Greedy-Gnorm (Max)", "style": "-", "color": "#2ca02c"}
}

TASK_DISPLAY_NAMES = {"sst2": "SST-2", "mnli": "MNLI"}
TASK_YMIN = {"sst2": 0.42, "mnli": 0.26}


def load_variant_runs(task_name: str):
    csv_pattern = os.path.join(GLUE_DIR, f"BERT_{task_name.lower()}_ggnorm_variants_seed_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        return None

    runs = []
    for p in csv_files:
        try:
            runs.append(pd.read_csv(p))
        except Exception:
            pass

    if not runs or 'Heads Pruned' not in runs[0].columns:
        return None

    heads_pruned = runs[0]['Heads Pruned'].values
    stats = {'Heads Pruned': heads_pruned, 'num_runs': len(runs)}

    for col in VARIANTS_META.keys():
        col_matrices = [df[col].values for df in runs if col in df.columns]
        if col_matrices:
            matrix = np.column_stack(col_matrices)
            stats[f"{col}_mean"] = np.nanmean(matrix, axis=1)
            stats[f"{col}_std"] = np.nanstd(matrix, axis=1, ddof=1) if matrix.shape[1] > 1 else np.zeros(len(heads_pruned))

    return pd.DataFrame(stats)


def plot_combined_overview(output_dir: str, dpi: int = 300):
    tasks = ["sst2", "mnli"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    fig.suptitle("BERT Head Pruning: Greedy-Gnorm Variants Comparison", fontsize=15, fontweight='bold', y=0.99)

    legend_handles, legend_labels = None, None

    for i, task in enumerate(tasks):
        ax = axes[i]
        stats_df = load_variant_runs(task)
        display_task = TASK_DISPLAY_NAMES.get(task, task.upper())
        ymin = TASK_YMIN.get(task.lower(), 0.35)

        ax.set_title(f"{display_task}", fontsize=13, fontweight='bold')

        if stats_df is not None and 'Heads Pruned' in stats_df.columns:
            x = stats_df['Heads Pruned']
            for var_col, meta in VARIANTS_META.items():
                mean_key = f"{var_col}_mean"
                std_key = f"{var_col}_std"

                if mean_key in stats_df.columns:
                    mean_acc = stats_df[mean_key]
                    std_acc = stats_df[std_key]

                    ax.plot(x, mean_acc, label=meta["label"], linestyle=meta["style"], color=meta["color"], linewidth=2.0)
                    if (std_acc > 0).any():
                        ax.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, color=meta["color"], alpha=0.20, edgecolor='none')

            ax.set_xlabel("Pruned Heads", fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.set_ylim([ymin, 1.01])

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
        else:
            ax.text(0.5, 0.5, f"No evaluation logs found for BERT_{task}", ha='center', va='center', transform=ax.transAxes, color='firebrick', fontsize=11)
            ax.set_ylim([ymin, 1.01])

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels), fontsize=11, frameon=True, facecolor="white", edgecolor="none", bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    save_png = os.path.join(output_dir, "bert_ggnorm_variants_overview.png")
    save_pdf = os.path.join(output_dir, "bert_ggnorm_variants_overview.pdf")
    plt.savefig(save_png, dpi=dpi)
    plt.savefig(save_pdf)
    plt.close()
    print(f"Saved: {save_png}")
    print(f"Saved: {save_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Plot Greedy-Gnorm variants overview.")
    parser.add_argument("--save-dir", type=str, default=os.path.join(BASE_DIR, "figures"), help="Directory to save figures")
    parser.add_argument("--dpi", type=int, default=300, help="Image DPI resolution")
    args = parser.parse_args()

    plot_combined_overview(args.save_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()

