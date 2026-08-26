"""
plot_glue_random.py

Plots Greedy-Gnorm vs. Random Pruning across GLUE benchmark tasks:
- Random Pruning is visualized with box plots and red outlier dots.
- Greedy-Gnorm is visualized as a solid trajectory line with +/- 1 SD uncertainty band.
- No-pruning unpruned accuracy is shown as a reference baseline.

Data Sources:
- Greedy-Gnorm: experiments_results/glue/{model}_{task}_benchmark_seed_{seed}.csv
- Random Pruning: experiments_results/glue/AE/{model}_{task}_ae_benchmark_seed_{seed}.csv

Usage:
    python plot_glue_random.py [--save-dir figures] [--dpi 300] [--stride 6] [--show]
"""

import os
import glob
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['legend.framealpha'] = 0.90
plt.rcParams['savefig.bbox'] = 'tight'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLUE_DIR = os.path.join(BASE_DIR, "experiments_results", "glue")
AE_DIR = os.path.join(GLUE_DIR, "AE")

MODELS_META = {
    "BERT": {
        "tasks": ["sst2", "mnli", "qnli"],
        "max_heads": 144,
        "display_name": "BERT"
    },
    "ROBERTA": {
        "tasks": ["sst2", "mnli", "qnli"],
        "max_heads": 144,
        "display_name": "RoBERTa"
    },
    "XLM_ROBERTA": {
        "tasks": ["sst2", "mnli", "qqp"],
        "max_heads": 144,
        "display_name": "XLM-RoBERTa"
    },
    "ALBERT": {
        "tasks": ["sst2", "mnli", "qnli"],
        "max_heads": 12,
        "display_name": "ALBERT"
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

SEEDS = [111, 222, 333, 555]


def get_task_ymin(task: str) -> float:
    return TASK_YMIN.get(task.lower(), 0.35)


def load_task_data(model_name: str, task: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Loads Greedy-Gnorm and Random Pruning benchmark data for all seeds.
    Returns:
        (gnorm_stats_df, random_all_seeds_df)
    """
    gnorm_runs = []
    random_runs = []

    for seed in SEEDS:
        gnorm_file = os.path.join(GLUE_DIR, f"{model_name}_{task}_benchmark_seed_{seed}.csv")
        ae_file = os.path.join(AE_DIR, f"{model_name}_{task}_ae_benchmark_seed_{seed}.csv")

        if not os.path.exists(gnorm_file) or not os.path.exists(ae_file):
            continue

        df_gnorm = pd.read_csv(gnorm_file)
        df_ae = pd.read_csv(ae_file)

        if 'Accuracy_Gnorm' in df_gnorm.columns:
            gnorm_runs.append(df_gnorm[['Heads Pruned', 'Accuracy_Gnorm']])

        if 'Accuracy_Random' in df_ae.columns:
            random_runs.append(df_ae[['Heads Pruned', 'Accuracy_Random']])

    if not gnorm_runs or not random_runs:
        return None

    # Aggregate Greedy-Gnorm stats
    df_gnorm_concat = pd.concat(gnorm_runs, ignore_index=True)
    gnorm_stats = df_gnorm_concat.groupby('Heads Pruned')['Accuracy_Gnorm'].agg(['mean', 'std']).reset_index()

    # Concatenate all Random runs preserving individual samples per head count
    df_random_concat = pd.concat(random_runs, ignore_index=True)

    return gnorm_stats, df_random_concat


def plot_master_box(output_dir: str, stride: int = 6, dpi: int = 300, show: bool = False):
    """
    Generates 4x3 Master Overview grid comparing Greedy-Gnorm curve vs. Random Pruning boxplots.
    """
    fig, axes = plt.subplots(4, 3, figsize=(16, 14.5), sharey=False)
    fig.suptitle(
        "Greedy-Gnorm vs. Random Pruning",
        fontsize=16, fontweight='bold', y=0.995
    )

    models_list = list(MODELS_META.keys())

    # Flier props: explicit red dots for outliers
    flierprops = dict(
        marker='o',
        markerfacecolor='#d62728',
        markeredgecolor='#d62728',
        markersize=4.0,
        alpha=0.85,
        zorder=4
    )
    boxprops = dict(
        facecolor='#e2e8f0',
        edgecolor='#475569',
        linewidth=1.0,
        zorder=2
    )
    whiskerprops = dict(color='#475569', linewidth=1.0, zorder=2)
    capprops = dict(color='#475569', linewidth=1.0, zorder=2)
    medianprops = dict(color='#0f172a', linewidth=1.3, zorder=3)

    for r, model_name in enumerate(models_list):
        meta = MODELS_META[model_name]
        tasks = meta["tasks"]
        display_name = meta["display_name"]
        is_albert = (model_name == "ALBERT")

        for c in range(3):
            ax = axes[r, c]
            if c < len(tasks):
                task = tasks[c]
                task_title = TASK_DISPLAY_NAMES.get(task, task.upper())
                ymin = get_task_ymin(task)

                data = load_task_data(model_name, task)
                if data is not None:
                    gnorm_stats, df_random = data

                    # 1. Prepare Box Plot Data for Random Pruning
                    all_heads = sorted(df_random['Heads Pruned'].unique())

                    if is_albert:
                        selected_heads = all_heads
                        box_width = 4.0
                    else:
                        # Apply stride to avoid crowded overlapping boxes
                        step_stride = max(1, stride)
                        selected_heads = [h for h in all_heads if h % step_stride == 0 or h == all_heads[-1]]
                        # Ensure deduplicated & sorted
                        selected_heads = sorted(list(set(selected_heads)))
                        box_width = max(1.5, step_stride * 0.55)

                    box_data = [
                        df_random[df_random['Heads Pruned'] == h]['Accuracy_Random'].dropna().values
                        for h in selected_heads
                    ]

                    # Filter out empty entries
                    valid_indices = [i for i, d in enumerate(box_data) if len(d) > 0]
                    plot_positions = [selected_heads[i] for i in valid_indices]
                    plot_data = [box_data[i] for i in valid_indices]

                    # Plot Boxplots
                    ax.boxplot(
                        plot_data,
                        positions=plot_positions,
                        widths=box_width,
                        patch_artist=True,
                        showmeans=False,
                        flierprops=flierprops,
                        boxprops=boxprops,
                        whiskerprops=whiskerprops,
                        capprops=capprops,
                        medianprops=medianprops,
                        manage_ticks=False,
                        zorder=2
                    )

                    # 2. Plot Greedy-Gnorm Solid Line & Band
                    x_gnorm = gnorm_stats['Heads Pruned']
                    mean_gnorm = gnorm_stats['mean']
                    std_gnorm = gnorm_stats['std'].fillna(0)

                    ax.plot(
                        x_gnorm, mean_gnorm,
                        label='Greedy-Gnorm',
                        color='#1f77b4',
                        linewidth=2.2,
                        zorder=5
                    )
                    ax.fill_between(
                        x_gnorm,
                        mean_gnorm - std_gnorm,
                        mean_gnorm + std_gnorm,
                        color='#1f77b4',
                        alpha=0.18,
                        edgecolor='none',
                        zorder=3
                    )

                    ax.set_title(f"{display_name} - {task_title}", fontsize=11.5, fontweight='bold')
                    ax.set_xlabel("Pruned Heads", fontsize=10)
                    ax.set_ylabel("Accuracy", fontsize=10)
                    ax.set_ylim([ymin, 1.01])
                    ax.set_xlim([-box_width, max(all_heads) + box_width])
            else:
                ax.axis('off')

    # Custom Clean Unified Legend
    legend_handles = [
        mlines.Line2D([], [], color='#1f77b4', linewidth=2.2, label='Greedy-Gnorm'),
        mpatches.Patch(facecolor='#e2e8f0', edgecolor='#475569', label='Random Pruning'),
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor='#d62728', markeredgecolor='#d62728', markersize=6, label='Random Outliers')
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=11.5,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        columnspacing=2.0,
        bbox_to_anchor=(0.5, -0.01)
    )

    plt.tight_layout(rect=[0, 0.035, 1, 0.98])

    os.makedirs(output_dir, exist_ok=True)
    save_path_png = os.path.join(output_dir, "glue_gnorm_vs_random_master.png")
    save_path_pdf = os.path.join(output_dir, "glue_gnorm_vs_random_master.pdf")

    plt.savefig(save_path_png, dpi=dpi)
    plt.savefig(save_path_pdf)
    print(f"Saved: {save_path_png}")
    print(f"Saved: {save_path_pdf}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_individual_boxes(output_dir: str, stride: int = 6, dpi: int = 300, show: bool = False):
    """
    Generates separate 1x3 horizontal panels for each model architecture.
    """
    os.makedirs(output_dir, exist_ok=True)
    models_list = list(MODELS_META.keys())

    flierprops = dict(
        marker='o',
        markerfacecolor='#d62728',
        markeredgecolor='#d62728',
        markersize=4.5,
        alpha=0.85,
        zorder=4
    )
    boxprops = dict(
        facecolor='#e2e8f0',
        edgecolor='#475569',
        linewidth=1.1,
        zorder=2
    )
    whiskerprops = dict(color='#475569', linewidth=1.1, zorder=2)
    capprops = dict(color='#475569', linewidth=1.1, zorder=2)
    medianprops = dict(color='#0f172a', linewidth=1.4, zorder=3)

    for model_name in models_list:
        meta = MODELS_META[model_name]
        tasks = meta["tasks"]
        display_name = meta["display_name"]
        is_albert = (model_name == "ALBERT")

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
        fig.suptitle(f"{display_name}: Greedy-Gnorm vs. Random Pruning", fontsize=14, fontweight='bold', y=1.02)

        for c, task in enumerate(tasks):
            ax = axes[c]
            task_title = TASK_DISPLAY_NAMES.get(task, task.upper())
            ymin = get_task_ymin(task)

            data = load_task_data(model_name, task)
            if data is not None:
                gnorm_stats, df_random = data
                all_heads = sorted(df_random['Heads Pruned'].unique())

                if is_albert:
                    selected_heads = all_heads
                    box_width = 4.0
                else:
                    step_stride = max(1, stride)
                    selected_heads = [h for h in all_heads if h % step_stride == 0 or h == all_heads[-1]]
                    selected_heads = sorted(list(set(selected_heads)))
                    box_width = max(1.5, step_stride * 0.55)

                box_data = [
                    df_random[df_random['Heads Pruned'] == h]['Accuracy_Random'].dropna().values
                    for h in selected_heads
                ]

                valid_indices = [i for i, d in enumerate(box_data) if len(d) > 0]
                plot_positions = [selected_heads[i] for i in valid_indices]
                plot_data = [box_data[i] for i in valid_indices]

                ax.boxplot(
                    plot_data,
                    positions=plot_positions,
                    widths=box_width,
                    patch_artist=True,
                    showmeans=False,
                    flierprops=flierprops,
                    boxprops=boxprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    medianprops=medianprops,
                    manage_ticks=False,
                    zorder=2
                )

                x_gnorm = gnorm_stats['Heads Pruned']
                mean_gnorm = gnorm_stats['mean']
                std_gnorm = gnorm_stats['std'].fillna(0)

                ax.plot(
                    x_gnorm, mean_gnorm,
                    label='Greedy-Gnorm',
                    color='#1f77b4',
                    linewidth=2.2,
                    zorder=5
                )
                ax.fill_between(
                    x_gnorm,
                    mean_gnorm - std_gnorm,
                    mean_gnorm + std_gnorm,
                    color='#1f77b4',
                    alpha=0.18,
                    edgecolor='none',
                    zorder=3
                )

                ax.set_title(f"{task_title}", fontsize=12, fontweight='bold')
                ax.set_xlabel("Pruned Heads", fontsize=10.5)
                ax.set_ylabel("Accuracy", fontsize=10.5)
                ax.set_ylim([ymin, 1.01])
                ax.set_xlim([-box_width, max(all_heads) + box_width])

        legend_handles = [
            mlines.Line2D([], [], color='#1f77b4', linewidth=2.2, label='Greedy-Gnorm'),
            mpatches.Patch(facecolor='#e2e8f0', edgecolor='#475569', label='Random Pruning'),
            mlines.Line2D([], [], marker='o', color='w', markerfacecolor='#d62728', markeredgecolor='#d62728', markersize=6, label='Random Outliers')
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            fontsize=11,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            columnspacing=2.0,
            bbox_to_anchor=(0.5, -0.06)
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
        save_png = os.path.join(output_dir, f"{model_name.lower()}_gnorm_vs_random_boxplots.png")
        save_pdf = os.path.join(output_dir, f"{model_name.lower()}_gnorm_vs_random_boxplots.pdf")

        plt.savefig(save_png, dpi=dpi)
        plt.savefig(save_pdf)
        print(f"Saved: {save_png}")
        print(f"Saved: {save_pdf}")

        if show:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot GLUE: Greedy-Gnorm vs. Random Pruning Boxplots with Red Outlier Dots")
    parser.add_argument("--save-dir", type=str, default="figures", help="Directory to save output figures (default: figures)")
    parser.add_argument("--stride", type=int, default=6, help="Step stride for boxplots on 144-head models (default: 6)")
    parser.add_argument("--individual", action="store_true", default=False, help="Also generate individual per-model plots")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output PNG (default: 300)")
    parser.add_argument("--show", action="store_true", default=False, help="Display the plot window")
    args = parser.parse_args()

    plot_master_box(output_dir=args.save_dir, stride=args.stride, dpi=args.dpi, show=args.show)
    if args.individual:
        plot_individual_boxes(output_dir=args.save_dir, stride=args.stride, dpi=args.dpi, show=args.show)


if __name__ == "__main__":
    main()
