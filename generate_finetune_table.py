import os
import glob
import csv
import pandas as pd

def generate_finetune_df(finetune_dir="experiments_results/finetune", seed="999"):
    models = [
        ("BERT", "BERT", 0.9243119266055045),
        ("RoBERTa", "ROBERTA", 0.9403669724770642),
        ("XLM-RoBERTa", "XLM_ROBERTA", 0.9220183486238532)
    ]
    
    milestones = [
        (72, "50.0%"),
        (108, "75.0%"),
        (120, "83.3%")
    ]
    
    size_csv_path = os.path.join(finetune_dir, "model_size_summary.csv")
    if not os.path.exists(size_csv_path):
        from measure_model_sizes import measure_all_model_sizes
        measure_all_model_sizes(size_csv_path)
        
    size_data = {}
    with open(size_csv_path, 'r') as fp_size:
        for row_s in csv.DictReader(fp_size):
            m_name = row_s['Model']
            h_pruned = int(row_s['Heads_Pruned'])
            if m_name not in size_data:
                size_data[m_name] = {}
            size_data[m_name][h_pruned] = {
                'size_mb': float(row_s['Size_MB']),
                'red_pct': float(row_s['Size_Reduction_Pct'])
            }
    
    records = []
    for display_name, file_prefix, dense_acc in models:
        dense_str = f"{dense_acc * 100:.2f}%"
        dense_size_mb = size_data.get(display_name, {}).get(0, {}).get('size_mb', 0.0)
        dense_size_str = f"{dense_size_mb:.2f} MB"
        
        for heads, ratio_str in milestones:
            summary_pattern = os.path.join(finetune_dir, f"{file_prefix}_sst2_gnorm_finetune_summary_seed_{seed}_{heads}.csv")
            matches = glob.glob(summary_pattern)
            
            if not matches:
                summary_pattern = os.path.join(finetune_dir, f"{file_prefix}_sst2_gnorm_finetune_summary_seed_*_{heads}.csv")
                matches = glob.glob(summary_pattern)
                
            if matches:
                with open(matches[0], 'r') as fp:
                    reader = list(csv.reader(fp))
                    header = reader[0]
                    row = reader[1]
                    data = dict(zip(header, row))
                    
                acc_before = float(data['Acc_Before_FT']) * 100.0
                acc_after = float(data['Acc_After_FT']) * 100.0
                gain = acc_after - acc_before
            else:
                acc_before = 0.0
                acc_after = 0.0
                gain = 0.0
                
            pruned_info = size_data.get(display_name, {}).get(heads, {'size_mb': 0.0, 'red_pct': 0.0})
            pruned_mb = pruned_info['size_mb']
            size_red_pct = pruned_info['red_pct']
            
            records.append({
                "Model": display_name,
                "Heads Pruned": heads,
                "Prune Ratio": ratio_str,
                "Dense Acc": dense_str,
                "Pre-FT Acc": f"{acc_before:.2f}%",
                "Post-FT Acc": f"{acc_after:.2f}%",
                "Recovery Gain": f"+{gain:.2f}%" if gain >= 0 else f"{gain:.2f}%",
                "Dense Size": dense_size_str,
                "Pruned Size": f"{pruned_mb:.2f} MB",
                "Size Reduction": f"-{size_red_pct:.2f}%"
            })
            
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = generate_finetune_df()
    print("=" * 110)
    print("POST-PRUNING FINE-TUNING RESULTS (SST-2)")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)
