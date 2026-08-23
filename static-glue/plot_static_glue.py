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
STATIC_RESULTS_DIR = os.path.join(BASE_DIR, "experiments_results")
DYNAMIC_RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "experiments_results", "glue")

MODELS_META = {
    "BERT": {
        "tasks": ["sst2", "mnli", "qnli"],
        "max_heads": 144,
        "color": "#1f77b4",
        "display_name": "BERT"
    },
    "ROBERTA": {
        "tasks": ["sst2", "mnli", "qnli"],
        "max_heads": 144,
        "color": "#2ca02c",
        "display_name": "RoBERTa"
    },
    "XLM_ROBERTA": {
        "tasks": ["sst2", "mnli", "qqp"],
        "max_heads": 144,
        "color": "#9467bd",
        "display_name": "XLM-RoBERTa"
    },
    "ALBERT": {
        "tasks": ["sst2", "mnli", "qnli"],
        "max_heads": 12,
        "color": "#d62728",
        "display_name": "ALBERT"
    }
}

METHODS_META = {
    "Accuracy_Gnorm": {
        "label": "Greedy-Gnorm",
        "style": "-",
        "color": "#1f77b4"
    },
    "Accuracy_Static_Taylor": {
        "label": "Taylor Importance",
        "style": "-",
        "color": "#ff7f0e"
    },
    "Accuracy_Static_AttAttr": {
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

def load_combined_runs(model_name: str, task_name: str, static_dir: str, dynamic_dir: str):
    combined_df = pd.DataFrame()
    
    # 1. Load Dynamic Greedy-Gnorm Runs
    dynamic_files = sorted(glob.glob(os.path.join(dynamic_dir, f"{model_name}_{task_name}_benchmark_seed_*.csv")))
    if dynamic_files:
        dfs_dyn = [pd.read_csv(f) for f in dynamic_files]
        if 'Heads Pruned' in dfs_dyn[0].columns:
            combined_df['Heads Pruned'] = dfs_dyn[0]['Heads Pruned']
            for col in ['Accuracy_Gnorm']:
                arrays = [df[col].values for df in dfs_dyn if col in df.columns]
                if arrays:
                    stacked = np.vstack(arrays)
                    combined_df[f"{col}_mean"] = np.mean(stacked, axis=0)
                    combined_df[f"{col}_std"] = np.std(stacked, axis=0)

    # 2. Load Static Runs (Taylor and Attribution)
    static_files = sorted(glob.glob(os.path.join(static_dir, f"{model_name}_{task_name}_static_benchmark_seed_*.csv")))
    if static_files:
        dfs_stat = [pd.read_csv(f) for f in static_files]
        if 'Heads Pruned' not in combined_df.columns and 'Heads Pruned' in dfs_stat[0].columns:
            combined_df['Heads Pruned'] = dfs_stat[0]['Heads Pruned']
            
        for col in ['Accuracy_Static_Taylor', 'Accuracy_Static_AttAttr']:
            arrays = [df[col].values for df in dfs_stat if col in df.columns]
            if arrays:
                stacked = np.vstack(arrays)
                combined_df[f"{col}_mean"] = np.mean(stacked, axis=0)
                combined_df[f"{col}_std"] = np.std(stacked, axis=0)

    return combined_df

def plot_static_master(static_dir: str, dynamic_dir: str, output_path: str):
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
                stats_df = load_combined_runs(model_name, task, static_dir, dynamic_dir)
                task_title = TASK_DISPLAY_NAMES.get(task, task.upper())
                ymin = get_task_ymin(task)

                if stats_df is not None and 'Heads Pruned' in stats_df.columns and len(stats_df.columns) > 1:
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
                    ax.set_title(f"{display_name} - {task_title}", fontsize=11, fontweight='bold')
                    ax.text(
                        0.5, 0.5,
                        f"No evaluation logs found for\n{model_name}_{task}",
                        ha='center', va='center', transform=ax.transAxes,
                        color='firebrick', fontsize=11
                    )
                    ax.set_ylim([ymin, 1.01])
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
    save_path_png = os.path.splitext(output_path)[0] + ".png"
    save_path_pdf = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(save_path_png, dpi=300)
    plt.savefig(save_path_pdf)
    plt.close()
    print(f"Saved PNG: {save_path_png}")
    print(f"Saved PDF: {save_path_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Plot GLUE Master Overview PNG & PDF")
    parser.add_argument("--static-dir", type=str, default=STATIC_RESULTS_DIR, help="Path to static experiments results")
    parser.add_argument("--dynamic-dir", type=str, default=DYNAMIC_RESULTS_DIR, help="Path to dynamic experiments results")
    parser.add_argument("--output", type=str, default=os.path.join(BASE_DIR, "static_glue_master_overview"), help="Base output path (without extension)")
    args, _ = parser.parse_known_args()

    plot_static_master(args.static_dir, args.dynamic_dir, args.output)

if __name__ == "__main__":
    main()
