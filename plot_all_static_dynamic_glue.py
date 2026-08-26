"""
Master Grid Overview Plot comparing all Dynamic and Static Pruning Methods across GLUE:
1. Greedy-Gnorm
2. Taylor Importance (Dynamic)
3. Taylor Importance (Static)
4. Self-Attention Attribution (Dynamic)
5. Self-Attention Attribution (Static)

Usage:
    python plot_all_static_dynamic_glue.py [--save-dir figures] [--dpi 300]
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
plt.rcParams['lines.linewidth'] = 1.8
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['legend.framealpha'] = 0.85
plt.rcParams['savefig.bbox'] = 'tight'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DYNAMIC_DIR = os.path.join(BASE_DIR, "experiments_results", "glue")
STATIC_DIR = os.path.join(BASE_DIR, "static-glue", "experiments_results")

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

METHODS_META = {
    "Accuracy_Gnorm": {
        "label": "Greedy-Gnorm",
        "style": "-",
        "color": "#1f77b4"  # Blue
    },
    "Accuracy_Taylor": {
        "label": "Taylor Importance\n(Dynamic)",
        "style": "-",
        "color": "#ff7f0e"  # Orange
    },
    "Accuracy_Static_Taylor": {
        "label": "Taylor Importance\n(Static)",
        "style": "--",
        "color": "#d62728"  # Crimson Red
    },
    "Accuracy_Attr": {
        "label": "Self-Attention Attribution\n(Dynamic, $m=10$)",
        "style": "-",
        "color": "#2ca02c"  # Green
    },
    "Accuracy_Static_AttAttr": {
        "label": "Self-Attention Attribution\n(Static, $m=20$)",
        "style": "--",
        "color": "#9467bd"  # Purple
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


def load_combined_stats(model_name: str, task_name: str):
    prefix = "XLM_ROBERTA" if model_name == "XLM_ROBERTA" else model_name.upper()
    t_lower = task_name.lower()
    
    seeds = [111, 222, 333, 555]
    all_runs = []
    
    for s in seeds:
        dyn_file = os.path.join(DYNAMIC_DIR, f"{prefix}_{t_lower}_benchmark_seed_{s}.csv")
        stat_file = os.path.join(STATIC_DIR, f"{prefix}_{t_lower}_static_benchmark_seed_{s}.csv")
        
        if not os.path.exists(dyn_file) or not os.path.exists(stat_file):
            continue
            
        df_dyn = pd.read_csv(dyn_file)
        df_stat = pd.read_csv(stat_file)
        
        merged = pd.DataFrame()
        merged['Heads Pruned'] = df_dyn['Heads Pruned']
        merged['seed'] = s
        
        # Dynamic columns
        if 'Accuracy_Gnorm' in df_dyn.columns:
            merged['Accuracy_Gnorm'] = df_dyn['Accuracy_Gnorm']
        if 'Accuracy_Taylor' in df_dyn.columns:
            merged['Accuracy_Taylor'] = df_dyn['Accuracy_Taylor']
        if 'Accuracy_Attr' in df_dyn.columns:
            merged['Accuracy_Attr'] = df_dyn['Accuracy_Attr']
            
        # Static columns
        if 'Accuracy_Static_Taylor' in df_stat.columns:
            merged['Accuracy_Static_Taylor'] = df_stat['Accuracy_Static_Taylor']
        if 'Accuracy_Static_AttAttr' in df_stat.columns:
            merged['Accuracy_Static_AttAttr'] = df_stat['Accuracy_Static_AttAttr']
            
        all_runs.append(merged)
        
    if not all_runs:
        return None
        
    df_concat = pd.concat(all_runs, ignore_index=True)
    
    # Compute mean and std per step
    agg_funcs = {}
    for col in METHODS_META.keys():
        if col in df_concat.columns:
            agg_funcs[col] = ['mean', 'std']
            
    stats = df_concat.groupby('Heads Pruned').agg(agg_funcs)
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    stats = stats.reset_index()
    return stats


def plot_master(output_dir: str, dpi: int = 300, show: bool = False):
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharey=False)
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
                stats_df = load_combined_stats(model_name, task)
                task_title = TASK_DISPLAY_NAMES.get(task, task.upper())
                ymin = get_task_ymin(task)

                if stats_df is not None and 'Heads Pruned' in stats_df.columns:
                    x = stats_df['Heads Pruned']

                    for method_col, m_meta in METHODS_META.items():
                        mean_key = f"{method_col}_mean"
                        std_key = f"{method_col}_std"

                        if mean_key in stats_df.columns:
                            mean_acc = stats_df[mean_key]
                            std_acc = stats_df[std_key].fillna(0)

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
                                alpha=0.12,
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
            fontsize=10.0,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            columnspacing=1.3,
            bbox_to_anchor=(0.5, -0.012)
        )

    plt.tight_layout(rect=[0, 0.045, 1, 0.98])
    
    os.makedirs(output_dir, exist_ok=True)
    save_path_png = os.path.join(output_dir, "glue_static_dynamic_master_overview.png")
    save_path_pdf = os.path.join(output_dir, "glue_static_dynamic_master_overview.pdf")
    
    plt.savefig(save_path_png, dpi=dpi)
    plt.savefig(save_path_pdf)
    print(f"Saved: {save_path_png}")
    print(f"Saved: {save_path_pdf}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot GLUE Master Overview with all Static and Dynamic Methods")
    parser.add_argument("--save-dir", type=str, default=os.path.join(BASE_DIR, "figures"), help="Directory to save figures")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures (default: 300)")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = parser.parse_args()

    plot_master(output_dir=args.save_dir, dpi=args.dpi, show=args.show)


if __name__ == "__main__":
    main()
