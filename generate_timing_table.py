import os
import csv
import pandas as pd

def generate_timing_df(
    main_timing_csv="experiments_results/glue/XLM_ROBERTA_mnli_timing_seed_555.csv",
    ae_timing_csv="experiments_results/glue/XLM_ROBERTA_mnli_ae_static_timing_seed_555.csv"
):
    if not os.path.exists(main_timing_csv):
        raise FileNotFoundError(f"Cannot find main timing CSV: {main_timing_csv}")
    if not os.path.exists(ae_timing_csv):
        raise FileNotFoundError(f"Cannot find AE timing CSV: {ae_timing_csv}")

    with open(main_timing_csv, 'r') as fp:
        r_main = list(csv.reader(fp))
    with open(ae_timing_csv, 'r') as fp:
        r_ae = list(csv.reader(fp))

    rows_main = {int(r[0]): r for r in r_main[1:] if r[0] != 'Total'}
    tot_main = [r for r in r_main if r[0] == 'Total'][0]

    rows_ae = {int(r[0]): r for r in r_ae[1:] if r[0] != 'Total'}
    tot_ae = [r for r in r_ae if r[0] == 'Total'][0]

    intervals = [1, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144]
    
    avg_gnorm = sum(float(rows_main[s][2]) for s in range(1, 145)) / 144.0
    avg_taylor = sum(float(rows_main[s][3]) for s in range(1, 145)) / 144.0
    avg_attr = sum(float(rows_main[s][4]) for s in range(1, 145)) / 144.0
    avg_ae = sum(float(rows_ae[s][2]) for s in range(1, 145)) / 144.0

    tot_gnorm = float(tot_main[2])
    tot_taylor = float(tot_main[3])
    tot_attr = float(tot_main[4])
    tot_ae = float(tot_ae[2])

    records = []
    for s in intervals:
        prune_ratio = (s / 144.0) * 100.0
        t_gnorm = float(rows_main[s][2])
        t_taylor = float(rows_main[s][3])
        t_attr = float(rows_main[s][4])
        t_ae = float(rows_ae[s][2])

        records.append({
            "Heads Pruned": str(s),
            "Prune Ratio": f"{prune_ratio:5.1f}%",
            "Greedy-Gnorm (s)": f"{t_gnorm:.2f}",
            "Taylor Importance (s)": f"{t_taylor:.2f}",
            "Self-Attention Attribution (s)": f"{t_attr:.2f}",
            "Attention Entropy (s)": f"{t_ae:.2f}"
        })

    records.append({
        "Heads Pruned": "Average/Step",
        "Prune Ratio": "-",
        "Greedy-Gnorm (s)": f"{avg_gnorm:.2f}",
        "Taylor Importance (s)": f"{avg_taylor:.2f}",
        "Self-Attention Attribution (s)": f"{avg_attr:.2f}",
        "Attention Entropy (s)": f"{avg_ae:.2f}"
    })
    
    records.append({
        "Heads Pruned": "Total Time",
        "Prune Ratio": "100.0%",
        "Greedy-Gnorm (s)": f"{tot_gnorm:.1f}s ({tot_gnorm/60:.1f}m)",
        "Taylor Importance (s)": f"{tot_taylor:.1f}s ({tot_taylor/60:.1f}m)",
        "Self-Attention Attribution (s)": f"{tot_attr:.1f}s ({tot_attr/60:.1f}m)",
        "Attention Entropy (s)": f"{tot_ae:.1f}s ({tot_ae/60:.1f}m)"
    })

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = generate_timing_df()
    print("=" * 110)
    print("XLM-ROBERTA MNLI WALL-CLOCK TIMING COMPARISON")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)
