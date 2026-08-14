"""
Outputs generated:
1. Individual Per-Architecture Pruning Trajectories (BERT, RoBERTa, XLM-RoBERTa, ALBERT)
2. Combined Master Overview Grid Across All Models and GLUE Tasks
3. Consolidated Summary Performance Report (CSV with Mean +/- SD)

Usage:
    python plot_glue.py [--save-dir SAVE_DIR] [--dpi DPI] [--show]
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

MODELS_META = {
    "BERT": {
        "tasks": ["sst2", "mnli", "qnli"],
        "max_heads": 144,
        "color": "#1f77b4",  # Blue
        "display_name": "BERT"
    },
    "ROBERTA": {
        "tasks": ["sst2", "mnli", "qnli"],
        "max_heads": 144,
        "color": "#2ca02c",  # Green
        "display_name": "RoBERTa"
    },
    "XLM_ROBERTA": {
        "tasks": ["sst2", "mnli", "qqp"],
        "max_heads": 144,
        "color": "#9467bd",  # Purple
        "display_name": "XLM-RoBERTa"
    },
    "ALBERT": {
        "tasks": ["sst2", "mnli", "qnli"],
        "max_heads": 12,
        "color": "#d62728",  # Red
        "display_name": "ALBERT"
    }
}

METHODS_META = {
    "Accuracy_Gnorm": {
        "label": "Greedy-Gnorm",
        "style": "-",
        "color": "#1f77b4"
    },
    "Accuracy_Taylor": {
        "label": "Taylor Importance",
        "style": "-",
        "color": "#ff7f0e"
    },
    "Accuracy_Attr": {
        "label": "Self-Attention Attribution",
        "style": "-",
        "color": "#2ca02c"
    }
}

TASK_DISPLAY_NAMES = {
    "sst2": "SST-2",
    "mnli": "MNLI",
    "qnli": "QNLI",
    "qqp": "QQP"
}

TASK_YMIN = {
    "mnli": 0.26,
    "sst2": 0.42,
    "qnli": 0.37,
    "qqp": 0.32
}


def get_task_ymin(task: str) -> float:
    return TASK_YMIN.get(task.lower(), 0.37)


def load_all_runs(model_name: str, task_name: str):
    """
    Loads all evaluation runs for a given model and task from experiments_results/glue/.
    Flexible for any number of seed CSV files.

    Returns:
        pd.DataFrame with 'Heads Pruned' and mean/std/count for each accuracy metric,
        or None if no runs found.
    """
    runs = []

    # Search for seed CSV benchmark files inside experiments_results/glue
    csv_patterns = [
        os.path.join(GLUE_DIR, f"{model_name}_{task_name}_benchmark_seed_*.csv"),
        os.path.join(GLUE_DIR, f"{model_name}_{task_name.lower()}_benchmark_seed_*.csv")
    ]
    
    seen_csvs = set()
    for pattern in csv_patterns:
        for p in sorted(glob.glob(pattern)):
            if p not in seen_csvs and os.path.exists(p):
                seen_csvs.add(p)
                try:
                    df = pd.read_csv(p)
                    runs.append((os.path.basename(p), df))
                except Exception as e:
                    print(f"Warning: Failed to read CSV file {p}: {e}")

    n = len(runs)
    print(f"{model_name} {task_name.upper()}: n={n}")

    if not runs:
        return None

    # Normalize column names if needed
    normalized_runs = []
    for name, df in runs:
        df_norm = df.copy()
        if 'Pruned_Heads' in df_norm.columns and 'Heads Pruned' not in df_norm.columns:
            df_norm.rename(columns={'Pruned_Heads': 'Heads Pruned'}, inplace=True)
        normalized_runs.append(df_norm)

    base_df = normalized_runs[0]
    if 'Heads Pruned' not in base_df.columns:
        return None

    heads_pruned = base_df['Heads Pruned'].values
    stats = {'Heads Pruned': heads_pruned, 'num_runs': n}

    for method_col in METHODS_META.keys():
        col_matrices = []
        for df in normalized_runs:
            if method_col in df.columns:
                col_matrices.append(df[method_col].values)

        if col_matrices:
            matrix = np.column_stack(col_matrices)
            mean_vals = np.nanmean(matrix, axis=1)
            std_vals = np.nanstd(matrix, axis=1, ddof=1) if matrix.shape[1] > 1 else np.zeros_like(mean_vals)
            stats[f"{method_col}_mean"] = mean_vals
            stats[f"{method_col}_std"] = std_vals
            stats[f"{method_col}_count"] = matrix.shape[1]

    return pd.DataFrame(stats)


def plot_per_architecture_trajectories(output_dir: str, show: bool = False):
    """
    Plots individual per-architecture trajectory subplots (1 x N tasks) for each model architecture,
    showing Mean line and Standard Deviation shadow bands across seeds.
    """
    for model_name, meta in MODELS_META.items():
        tasks = meta["tasks"]
        display_name = meta["display_name"]

        fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4.8), sharey=False)
        if len(tasks) == 1:
            axes = [axes]

        fig.suptitle(
            f"{display_name} Pruning Trajectories",
            fontsize=14, fontweight='bold', y=1.02
        )

        for i, task in enumerate(tasks):
            ax = axes[i]
            stats_df = load_all_runs(model_name, task)
            ymin = get_task_ymin(task)

            if stats_df is not None and 'Heads Pruned' in stats_df.columns:
                x = stats_df['Heads Pruned']

                for method_col, m_meta in METHODS_META.items():
                    mean_key = f"{method_col}_mean"
                    std_key = f"{method_col}_std"

                    if mean_key in stats_df.columns:
                        mean_acc = stats_df[mean_key]
                        std_acc = stats_df[std_key]

                        # Plot Mean line
                        ax.plot(
                            x, mean_acc,
                            label=f"{m_meta['label']}",
                            linestyle=m_meta['style'],
                            color=m_meta['color'],
                            linewidth=2.0
                        )

                        # Plot Standard Deviation shadow band
                        ax.fill_between(
                            x,
                            mean_acc - std_acc,
                            mean_acc + std_acc,
                            color=m_meta['color'],
                            alpha=0.20,
                            edgecolor='none'
                        )

                task_title = TASK_DISPLAY_NAMES.get(task, task.upper())
                ax.set_title(f"{task_title}", fontsize=12, fontweight='bold')
                ax.set_xlabel("Number of Pruned Heads", fontsize=11)
                ax.set_ylabel("Accuracy", fontsize=11)
                ax.legend(loc="lower left", fontsize=9.5)
                ax.set_ylim([ymin, 1.01])
            else:
                ax.set_title(f"{task.upper()} (Pending Data)", fontsize=12)
                ax.text(
                    0.5, 0.5,
                    f"No evaluation logs found for\n{model_name}_{task}",
                    ha='center', va='center', transform=ax.transAxes, color='firebrick', fontsize=11
                )
                ax.set_ylim([ymin, 1.01])

        plt.tight_layout()
        save_path_png = os.path.join(output_dir, f"{model_name.lower()}_pruning_trajectories.png")
        save_path_pdf = os.path.join(output_dir, f"{model_name.lower()}_pruning_trajectories.pdf")
        plt.savefig(save_path_png, dpi=300)
        plt.savefig(save_path_pdf)
        print(f"Saved: {save_path_png}")
        print(f"Saved: {save_path_pdf}")

        if show:
            plt.show()
        else:
            plt.close()


def plot_master_overview(output_dir: str, show: bool = False):
    """
    Generates a master 4x3 grid plot showing all 4 models across their respective GLUE tasks
    with SD shadow bands and a flattened bottom legend.
    """
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    fig.suptitle(
        "GLUE Benchmark Pruning Overview",
        fontsize=16, fontweight='bold', y=0.995
    )

    models_list = list(MODELS_META.keys())
    legend_handles, legend_labels = None, None

    for r, model_name in enumerate(models_list):
        meta = MODELS_META[model_name]
        tasks = meta["tasks"]
        display_name = meta["display_name"]

        for c in range(3):
            ax = axes[r, c]
            if c < len(tasks):
                task = tasks[c]
                stats_df = load_all_runs(model_name, task)
                task_title = TASK_DISPLAY_NAMES.get(task, task.upper())
                ymin = get_task_ymin(task)

                if stats_df is not None and 'Heads Pruned' in stats_df.columns:
                    x = stats_df['Heads Pruned']

                    for method_col, m_meta in METHODS_META.items():
                        mean_key = f"{method_col}_mean"
                        std_key = f"{method_col}_std"

                        if mean_key in stats_df.columns:
                            mean_acc = stats_df[mean_key]
                            std_acc = stats_df[std_key]

                            ax.plot(
                                x, mean_acc,
                                label=m_meta['label'],
                                linestyle=m_meta['style'],
                                color=m_meta['color'],
                                linewidth=1.8
                            )

                            ax.fill_between(
                                x,
                                mean_acc - std_acc,
                                mean_acc + std_acc,
                                color=m_meta['color'],
                                alpha=0.18,
                                edgecolor='none'
                            )

                    ax.set_title(f"{display_name} - {task_title}", fontsize=11, fontweight='bold')
                    ax.set_xlabel("Pruned Heads", fontsize=9.5)
                    ax.set_ylabel("Accuracy", fontsize=9.5)
                    ax.set_ylim([ymin, 1.01])

                    if legend_handles is None:
                        legend_handles, legend_labels = ax.get_legend_handles_labels()
            else:
                ax.axis('off')

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            fontsize=12,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            bbox_to_anchor=(0.5, -0.01)
        )

    plt.tight_layout(rect=[0, 0.025, 1, 0.98])
    save_path_png = os.path.join(output_dir, "glue_master_overview.png")
    save_path_pdf = os.path.join(output_dir, "glue_master_overview.pdf")
    plt.savefig(save_path_png, dpi=300)
    plt.savefig(save_path_pdf)
    print(f"Saved: {save_path_png}")
    print(f"Saved: {save_path_pdf}")

    if show:
        plt.show()
    else:
        plt.close()


def generate_summary_report(output_dir: str):
    """
    Generates a consolidated summary CSV report across pruning milestones (0%, 25%, 50%, 75%)
    with Mean +/- SD metrics.
    """
    summary_rows = []

    for model_name, meta in MODELS_META.items():
        max_heads = meta["max_heads"]
        if model_name == "ALBERT":
            # ALBERT has 12 tied heads
            target_checkpoints = {
                "0% (Baseline)": 0,
                "25% Pruned": 3 * 12,
                "50% Pruned": 6 * 12,
                "75% Pruned": 9 * 12
            }
        else:
            target_checkpoints = {
                "0% (Baseline)": 0,
                "25% Pruned": int(max_heads * 0.25),
                "50% Pruned": int(max_heads * 0.50),
                "75% Pruned": int(max_heads * 0.75)
            }

        for task in meta["tasks"]:
            stats_df = load_all_runs(model_name, task)
            if stats_df is None or 'Heads Pruned' not in stats_df.columns:
                continue

            num_runs = int(stats_df['num_runs'].iloc[0])

            for label, target_val in target_checkpoints.items():
                idx = (stats_df['Heads Pruned'] - target_val).abs().idxmin()
                actual_pruned = int(stats_df.loc[idx, 'Heads Pruned'])

                row = {
                    "Architecture": meta["display_name"],
                    "GLUE Task": task.upper(),
                    "Prune Ratio": label,
                    "Actual Heads Pruned": actual_pruned,
                    "Num Runs/Seeds": num_runs
                }

                for method_col, m_meta in METHODS_META.items():
                    m_label = m_meta["label"]
                    mean_key = f"{method_col}_mean"
                    std_key = f"{method_col}_std"

                    if mean_key in stats_df.columns:
                        mean_v = stats_df.loc[idx, mean_key]
                        std_v = stats_df.loc[idx, std_key]
                        row[f"{m_label} Mean"] = round(mean_v, 4)
                        row[f"{m_label} SD"] = round(std_v, 4)
                        row[f"{m_label} (Mean +/- SD)"] = f"{mean_v:.4f} ± {std_v:.4f}"

                summary_rows.append(row)

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        csv_path = os.path.join(RESULTS_DIR, "summary_pruning_report.csv")
        df_summary.to_csv(csv_path, index=False)
        print(f"\nSummary table successfully exported to: {csv_path}")
    else:
        print("No evaluation logs found to generate summary report.")


def main():
    parser = argparse.ArgumentParser(description="GLUE Pruning Benchmark Plotting Tool with SD Shadows")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save output figures")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved PNG figures")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")

    args = parser.parse_args()

    if args.dpi:
        plt.rcParams['figure.dpi'] = args.dpi

    output_dir = args.save_dir if args.save_dir else os.path.join(BASE_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(GLUE_DIR, exist_ok=True)

    print("=" * 60)
    print("Generating GLUE Benchmark Plots (Individual Plots, Master Grid & Consolidated Report)...")
    print(f"Reading data from: {GLUE_DIR}")
    print(f"Saving figures to: {output_dir}")
    print("=" * 60)

    print("\n[1/3] Generating Individual Per-Architecture Trajectory Plots...")
    plot_per_architecture_trajectories(output_dir, show=args.show)

    print("\n[2/3] Generating Master Overview Grid Plot...")
    plot_master_overview(output_dir, show=args.show)

    print("\n[3/3] Generating Consolidated Summary Performance Report...")
    generate_summary_report(output_dir)

    print("\nAll plotting tasks completed successfully!")


if __name__ == "__main__":
    main()
