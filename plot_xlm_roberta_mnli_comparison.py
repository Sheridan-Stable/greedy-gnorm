"""
Plot comparing all forward head pruning methods on XLM-RoBERTa for MNLI:
1. Greedy-Gnorm
2. Taylor Importance
3. Self-Attention Attribution
4. Attention Entropy (AE - Static)

Output generated in --save-dir (default: figures):
- xlm_roberta_mnli_inc_static_seed_555.png / .pdf

Formatting matches plot_glue.py and plot_ggnorm_variants.py with bottom legend.

Usage:
    python plot_xlm_roberta_mnli_comparison.py [--seed 555] [--save-dir figures] [--dpi 300]
"""

import os
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
    },
    "Accuracy_AE": {
        "label": "Attention Entropy",
        "style": "-",
        "color": "#d62728"
    }
}


def load_data(seed: int):
    std_bench_path = os.path.join(GLUE_DIR, f"XLM_ROBERTA_mnli_benchmark_seed_{seed}.csv")
    ae_bench_path = os.path.join(GLUE_DIR, f"XLM_ROBERTA_mnli_ae_static_benchmark_seed_{seed}.csv")

    if not os.path.exists(std_bench_path):
        raise FileNotFoundError(f"Missing standard benchmark CSV: {std_bench_path}")
    if not os.path.exists(ae_bench_path):
        raise FileNotFoundError(f"Missing AE benchmark CSV: {ae_bench_path}")

    df_std = pd.read_csv(std_bench_path)
    df_ae = pd.read_csv(ae_bench_path)
    return df_std, df_ae


def plot_xlm_roberta_mnli(seed: int, output_dir: str, dpi: int = 300, show: bool = False):
    df_std, df_ae = load_data(seed)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_title("XLM-RoBERTa - MNLI", fontsize=13, fontweight='bold', pad=10)

    x = df_std['Heads Pruned'].values

    # Plot each method
    for method_col, meta in METHODS_META.items():
        df = df_std if method_col in df_std.columns else df_ae
        if method_col in df.columns:
            y = df[method_col].values
            ax.plot(
                x, y,
                label=meta["label"],
                linestyle=meta["style"],
                color=meta["color"],
                linewidth=2.0
            )

    ax.set_xlabel("Pruned Heads", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xlim([-2, 146])
    ax.set_ylim([0.26, 1.01])

    # Clean centered 4-column legend on the bottom
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=4,
            fontsize=9.8,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            columnspacing=1.3,
            handletextpad=0.5,
            bbox_to_anchor=(0.5, -0.015)
        )

    plt.tight_layout(rect=[0, 0.045, 1, 0.98])

    save_path_png = os.path.join(output_dir, f"xlm_roberta_mnli_inc_static_seed_{seed}.png")
    save_path_pdf = os.path.join(output_dir, f"xlm_roberta_mnli_inc_static_seed_{seed}.pdf")

    plt.savefig(save_path_png, dpi=dpi)
    plt.savefig(save_path_pdf)
    print(f"Saved: {save_path_png}")
    print(f"Saved: {save_path_pdf}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot XLM-RoBERTa MNLI pruning trajectories including static AE")
    parser.add_argument("--seed", type=int, default=555, help="Seed to plot (default: 555)")
    parser.add_argument("--save-dir", type=str, default=os.path.join(BASE_DIR, "figures"), help="Directory to save figures")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures (default: 300)")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    plot_xlm_roberta_mnli(seed=args.seed, output_dir=args.save_dir, dpi=args.dpi, show=args.show)


if __name__ == "__main__":
    main()
