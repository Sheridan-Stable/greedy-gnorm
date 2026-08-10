"""
Plots comparing Greedy-Gnorm variants (Product, Sum, Max) on BERT for SST-2 and MNLI tasks.

Outputs generated in --save-dir (default: figures):
1. bert_sst2_ggnorm_variants.png / .pdf
2. bert_mnli_ggnorm_variants.png / .pdf
3. bert_ggnorm_variants_overview.png / .pdf (Combined side-by-side plot)

Usage:
    python plot_ggnorm_variants.py [--save-dir SAVE_DIR] [--dpi DPI] [--show]
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
RESULTS_DIR = os.path.join(BASE_DIR, "experiments_results")
GLUE_DIR = os.path.join(RESULTS_DIR, "glue")

VARIANTS_META = {
    "Accuracy_Product": {
        "label": "Greedy-Gnorm (Product)",
        "style": "-",
        "color": "#1f77b4"  # Blue
    },
    "Accuracy_Sum": {
        "label": "Greedy-Gnorm (Sum)",
        "style": "-",
        "color": "#ff7f0e"  # Orange
    },
    "Accuracy_Max": {
        "label": "Greedy-Gnorm (Max)",
        "style": "-",
        "color": "#2ca02c"  # Green
    }
}

TASK_DISPLAY_NAMES = {
    "sst2": "SST-2",
    "mnli": "MNLI"
}

TASK_YMIN = {
    "sst2": 0.42,
    "mnli": 0.26
}


def get_task_ymin(task: str) -> float:
    return TASK_YMIN.get(task.lower(), 0.35)


def load_variant_runs(task_name: str):
    """
    Loads all variant evaluation runs for BERT on a given task from experiments_results/glue/.
    Flexible for any number of seed CSV files.
    """
    csv_pattern = os.path.join(GLUE_DIR, f"BERT_{task_name.lower()}_ggnorm_variants_seed_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    print(f"BERT {task_name.upper()} Gnorm Variants: found {len(csv_files)} run(s)")

    if not csv_files:
        return None

    runs = []
    for p in csv_files:
        try:
            df = pd.read_csv(p)
            runs.append(df)
        except Exception as e:
            print(f"Warning: Failed to read CSV file {p}: {e}")

    if not runs:
        return None

    base_df = runs[0]
    if 'Heads Pruned' not in base_df.columns:
        return None

    heads_pruned = base_df['Heads Pruned'].values
    stats = {'Heads Pruned': heads_pruned, 'num_runs': len(runs)}

    for col in VARIANTS_META.keys():
        col_matrices = [df[col].values for df in runs if col in df.columns]
        if col_matrices:
            matrix = np.column_stack(col_matrices)
            mean_vals = np.nanmean(matrix, axis=1)
            std_vals = np.nanstd(matrix, axis=1, ddof=1) if matrix.shape[1] > 1 else np.zeros_like(mean_vals)
            stats[f"{col}_mean"] = mean_vals
            stats[f"{col}_std"] = std_vals
            stats[f"{col}_count"] = matrix.shape[1]

    return pd.DataFrame(stats)


def plot_single_task(task_name: str, output_dir: str, dpi: int = 300, show: bool = False):
    """
    Generates an individual plot for a single task (SST2 or MNLI) comparing Gnorm variants
    with a flattened bottom legend.
    """
    stats_df = load_variant_runs(task_name)
    display_task = TASK_DISPLAY_NAMES.get(task_name.lower(), task_name.upper())
    ymin = get_task_ymin(task_name)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.set_title(f"BERT - {display_task}: Greedy-Gnorm Variants Comparison", fontsize=13, fontweight='bold')

    legend_handles, legend_labels = None, None

    if stats_df is not None and 'Heads Pruned' in stats_df.columns:
        x = stats_df['Heads Pruned']
        for var_col, meta in VARIANTS_META.items():
            mean_key = f"{var_col}_mean"
            std_key = f"{var_col}_std"

            if mean_key in stats_df.columns:
                mean_acc = stats_df[mean_key]
                std_acc = stats_df[std_key]

                ax.plot(
                    x, mean_acc,
                    label=meta["label"],
                    linestyle=meta["style"],
                    color=meta["color"],
                    linewidth=2.0
                )

                if (std_acc > 0).any():
                    ax.fill_between(
                        x,
                        mean_acc - std_acc,
                        mean_acc + std_acc,
                        color=meta["color"],
                        alpha=0.20,
                        edgecolor='none'
                    )

        ax.set_xlabel("Pruned Heads", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim([ymin, 1.01])

        legend_handles, legend_labels = ax.get_legend_handles_labels()
    else:
        ax.text(
            0.5, 0.5,
            f"No evaluation logs found for BERT {display_task} Gnorm variants",
            ha='center', va='center', transform=ax.transAxes, color='firebrick', fontsize=11
        )
        ax.set_ylim([ymin, 1.01])

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            fontsize=10.5,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            bbox_to_anchor=(0.5, -0.03)
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_png = os.path.join(output_dir, f"bert_{task_name.lower()}_ggnorm_variants.png")
    save_pdf = os.path.join(output_dir, f"bert_{task_name.lower()}_ggnorm_variants.pdf")
    plt.savefig(save_png, dpi=dpi)
    plt.savefig(save_pdf)
    print(f"Saved: {save_png}")
    print(f"Saved: {save_pdf}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_combined_overview(output_dir: str, dpi: int = 300, show: bool = False):
    """
    Generates a combined side-by-side 1x2 overview plot comparing Gnorm variants on SST-2 and MNLI
    with a flattened bottom legend.
    """
    tasks = ["sst2", "mnli"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    fig.suptitle(
        "BERT Head Pruning: Greedy-Gnorm Variants Comparison",
        fontsize=15, fontweight='bold', y=0.99
    )

    legend_handles, legend_labels = None, None

    for i, task in enumerate(tasks):
        ax = axes[i]
        stats_df = load_variant_runs(task)
        display_task = TASK_DISPLAY_NAMES.get(task, task.upper())
        ymin = get_task_ymin(task)

        ax.set_title(f"{display_task}", fontsize=13, fontweight='bold')

        if stats_df is not None and 'Heads Pruned' in stats_df.columns:
            x = stats_df['Heads Pruned']
            for var_col, meta in VARIANTS_META.items():
                mean_key = f"{var_col}_mean"
                std_key = f"{var_col}_std"

                if mean_key in stats_df.columns:
                    mean_acc = stats_df[mean_key]
                    std_acc = stats_df[std_key]

                    ax.plot(
                        x, mean_acc,
                        label=meta["label"],
                        linestyle=meta["style"],
                        color=meta["color"],
                        linewidth=2.0
                    )

                    if (std_acc > 0).any():
                        ax.fill_between(
                            x,
                            mean_acc - std_acc,
                            mean_acc + std_acc,
                            color=meta["color"],
                            alpha=0.20,
                            edgecolor='none'
                        )

            ax.set_xlabel("Pruned Heads", fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.set_ylim([ymin, 1.01])

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
        else:
            ax.text(
                0.5, 0.5,
                f"No evaluation logs found for BERT_{task}",
                ha='center', va='center', transform=ax.transAxes, color='firebrick', fontsize=11
            )
            ax.set_ylim([ymin, 1.01])

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            fontsize=11,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            bbox_to_anchor=(0.5, -0.03)
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_png = os.path.join(output_dir, "bert_ggnorm_variants_overview.png")
    save_pdf = os.path.join(output_dir, "bert_ggnorm_variants_overview.pdf")
    plt.savefig(save_png, dpi=dpi)
    plt.savefig(save_pdf)
    print(f"Saved: {save_png}")
    print(f"Saved: {save_pdf}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Greedy-Gnorm variants trajectories on BERT (SST-2 and MNLI).")
    parser.add_argument("--save-dir", type=str, default=os.path.join(BASE_DIR, "figures"), help="Directory to save figures")
    parser.add_argument("--dpi", type=int, default=300, help="Image DPI resolution")
    parser.add_argument("--show", action="store_true", help="Display interactive plot window")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Generating Greedy-Gnorm variants plots...")
    plot_single_task("sst2", args.save_dir, dpi=args.dpi, show=args.show)
    plot_single_task("mnli", args.save_dir, dpi=args.dpi, show=args.show)
    plot_combined_overview(args.save_dir, dpi=args.dpi, show=args.show)
    print("Done!")


if __name__ == "__main__":
    main()
