import os
import glob
import csv
import math
from collections import defaultdict
import pandas as pd

def compute_mean_std(values):
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean_val = sum(values) / n
    if n <= 1:
        std_val = 0.0
    else:
        variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        std_val = math.sqrt(variance)
    return mean_val, std_val

def generate_glue_results_df(results_dir="experiments_results/glue"):
    files = sorted(glob.glob(os.path.join(results_dir, "*_benchmark_seed_*.csv")))
    bench_files = [f for f in files if "static" not in f and "timing" not in f and "ae" not in f]
    ae_files = sorted(glob.glob(os.path.join(results_dir, "AE", "*_ae_benchmark_seed_*.csv")))
    
    # Store: data[model][task][ratio][method] = list of floats
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for f in bench_files:
        fname = os.path.basename(f)
        parts = fname.split("_")
        
        if fname.startswith("XLM_ROBERTA"):
            model = "XLM-RoBERTa"
            task = parts[2].lower()
        else:
            model = parts[0]
            if model.upper() == "BERT": model = "BERT"
            elif model.upper() == "ROBERTA": model = "RoBERTa"
            elif model.upper() == "ALBERT": model = "ALBERT"
            task = parts[1].lower()
            
        with open(f, "r") as fp:
            reader = list(csv.reader(fp))
            header = reader[0]
            rows = reader[1:]
            
        max_steps = len(rows) - 1
        step_map = {
            "0%": 0,
            "25%": int(round(0.25 * max_steps)),
            "50%": int(round(0.50 * max_steps)),
            "75%": int(round(0.75 * max_steps))
        }
        
        col_gnorm = header.index("Accuracy_Gnorm")
        col_taylor = header.index("Accuracy_Taylor")
        col_attr = header.index("Accuracy_Attr")
        
        for ratio, step_idx in step_map.items():
            if step_idx < len(rows):
                r = rows[step_idx]
                data[model][task][ratio]["gnorm"].append(float(r[col_gnorm]))
                data[model][task][ratio]["taylor"].append(float(r[col_taylor]))
                data[model][task][ratio]["attr"].append(float(r[col_attr]))

    for f in ae_files:
        fname = os.path.basename(f)
        parts = fname.split("_")
        
        if fname.startswith("XLM_ROBERTA"):
            model = "XLM-RoBERTa"
            task = parts[2].lower()
        else:
            model = parts[0]
            if model.upper() == "BERT": model = "BERT"
            elif model.upper() == "ROBERTA": model = "RoBERTa"
            elif model.upper() == "ALBERT": model = "ALBERT"
            task = parts[1].lower()
            
        with open(f, "r") as fp:
            reader = list(csv.reader(fp))
            header = reader[0]
            rows = reader[1:]
            
        max_steps = len(rows) - 1
        step_map = {
            "0%": 0,
            "25%": int(round(0.25 * max_steps)),
            "50%": int(round(0.50 * max_steps)),
            "75%": int(round(0.75 * max_steps))
        }
        
        col_ae = header.index("Accuracy_AE")
        for ratio, step_idx in step_map.items():
            if step_idx < len(rows):
                r = rows[step_idx]
                data[model][task][ratio]["ae"].append(float(r[col_ae]))

    model_tasks = [
        ("BERT", ["sst2", "mnli", "qnli"]),
        ("RoBERTa", ["sst2", "mnli", "qnli"]),
        ("XLM-RoBERTa", ["sst2", "mnli", "qqp"]),
        ("ALBERT", ["sst2", "mnli", "qnli"])
    ]
    
    task_display_names = {
        "sst2": "SST-2",
        "mnli": "MNLI",
        "qnli": "QNLI",
        "qqp": "QQP"
    }

    records = []
    for model_name, tasks in model_tasks:
        for task_key in tasks:
            t_name = task_display_names[task_key]
            for ratio in ["0%", "25%", "50%", "75%"]:
                g_m, g_s = compute_mean_std(data[model_name][task_key][ratio]["gnorm"])
                t_m, t_s = compute_mean_std(data[model_name][task_key][ratio]["taylor"])
                a_m, a_s = compute_mean_std(data[model_name][task_key][ratio]["attr"])
                ae_m, ae_s = compute_mean_std(data[model_name][task_key][ratio]["ae"])
                
                records.append({
                    "Model": model_name,
                    "Task": t_name,
                    "Prune Ratio": ratio,
                    "Greedy-Gnorm": f"{g_m:.4f} ± {g_s:.4f}",
                    "Taylor Importance": f"{t_m:.4f} ± {t_s:.4f}",
                    "Self-Attention Attribution": f"{a_m:.4f} ± {a_s:.4f}",
                    "Attention Entropy": f"{ae_m:.4f} ± {ae_s:.4f}"
                })
                
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = generate_glue_results_df()
    print("=" * 110)
    print("GLUE DYNAMIC PRUNING RESULTS (MEAN ± STD)")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)
