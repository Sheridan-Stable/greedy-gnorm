import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import gc
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.modules.linear
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from textpruner import TransformerPruner

# =========================================================================
# PyTorch Replication Safe Linear Patch
# =========================================================================

def _parse_torch_version(v_str):
    clean_v = v_str.split("+")[0].split("a")[0].split("b")[0].split("rc")[0]
    return tuple(int(x) for x in clean_v.split(".")[:3])

if _parse_torch_version(torch.__version__) > (2, 0, 1):
    if not getattr(F, "_is_monkeypatched", False):
        _orig_linear = F.linear

        def replication_safe_linear(input, weight, bias=None):
            if weight.size(1) == 0:
                out_shape = list(input.shape)
                out_shape[-1] = weight.size(0)
                zero_out = torch.zeros(out_shape, device=input.device, dtype=input.dtype)
                if bias is not None:
                    zero_out = zero_out + bias
                return zero_out
            return _orig_linear(input, weight, bias)

        F.linear = replication_safe_linear
        torch.nn.modules.linear.F.linear = replication_safe_linear
        F._is_monkeypatched = True


# =========================================================================
# Utility Functions
# =========================================================================

def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_datapoints(dataset_column, num_samples=1000, seed=555):
    total = len(dataset_column)
    n = min(num_samples, total)
    rng = random.Random(seed)
    indices = rng.sample(range(total), n)
    return [dataset_column[i] for i in indices]


def sample_datapoints_pair(col1, col2, num_samples=1000, seed=555):
    total = len(col1)
    n = min(num_samples, total)
    rng = random.Random(seed)
    indices = rng.sample(range(total), n)
    return [col1[i] for i in indices], [col2[i] for i in indices]


def set_seed(seed=555):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_new_head_mask_basedonAE(head_mask_previous, AE_matrix):
    head_mask = head_mask_previous.clone()
    selected_matrix = head_mask * AE_matrix.to(head_mask.device)
    argmax_idx = torch.argmax(selected_matrix).item()
    layer = argmax_idx // selected_matrix.size(1)
    head = argmax_idx % selected_matrix.size(1)
    head_mask[layer][head] = 0
    return head_mask


def get_new_head_mask_basedonAE_inverse(head_mask_previous, AE_matrix):
    head_mask = head_mask_previous.clone()
    selected_matrix = AE_matrix.to(head_mask.device).clone()
    selected_matrix[head_mask == 0] = float('inf')
    argmin_idx = torch.argmin(selected_matrix).item()
    layer = argmin_idx // selected_matrix.size(1)
    head = argmin_idx % selected_matrix.size(1)
    head_mask[layer][head] = 0
    return head_mask


# =========================================================================
# Attention Entropy (AE) Score Calculation
# =========================================================================

def compute_raw_ae_matrix(ds, model, tokenizer, is_pair=False, key1_name="sentence", key2_name=None,
                          batch_size=128, device="cuda", num_samples=1000, seed=555, eps=1e-7):
    """
    Computes epsilon-rectified Attention Entropy (AE) matrix:
        H(A) = - 1/S * sum_{i,j} (A_ij + eps) * log(A_ij + eps)
    Averaged across calibration samples.
    """
    model.to(device)
    model.eval()

    entropy_accum = torch.zeros(12, 12, device=device)

    if not is_pair:
        all_k1 = ds["train"][key1_name]
        n_samples = min(num_samples, len(all_k1))
        all_k1 = sample_datapoints(all_k1, n_samples, seed=seed)
        all_k2 = None
    else:
        all_k1 = ds["train"][key1_name]
        all_k2 = ds["train"][key2_name]
        n_samples = min(num_samples, len(all_k1))
        all_k1, all_k2 = sample_datapoints_pair(all_k1, all_k2, n_samples, seed=seed)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b1 = all_k1[i:i + batch_size]
            if not is_pair:
                inputs = tokenizer(b1, return_tensors="pt", padding=True, truncation=True)
            else:
                b2 = all_k2[i:i + batch_size]
                inputs = tokenizer(b1, b2, return_tensors="pt", padding=True, truncation=True)

            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            attns = outputs.attentions  # tuple of 12 layer tensors: [batch, num_heads, S, S]

            for layer_idx in range(12):
                A = attns[layer_idx]
                if A.size(1) == 0:
                    continue
                token_entropy = (-(A + eps).log() * (A + eps)).sum(dim=-1).mean(dim=-1)
                entropy_accum[layer_idx] += token_entropy.sum(dim=0)

            del inputs, outputs, attns
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    mean_entropy = entropy_accum.cpu() / n_samples
    return mean_entropy


# =========================================================================
# Accuracy Evaluator
# =========================================================================

def evaluate_accuracy(ds, model, tokenizer, val_split="validation", is_pair=False,
                      key1_name="sentence", key2_name=None, label_name="label",
                      label_map=None, device="cuda", batch_size=128):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    all_k1 = ds[val_split][key1_name]
    all_labels = ds[val_split][label_name]
    all_k2 = ds[val_split][key2_name] if is_pair else None
    val_size = len(all_labels)

    if label_map is not None:
        label_map_tensor = torch.tensor(label_map, device=device)
    else:
        label_map_tensor = None

    with torch.no_grad():
        for i in range(0, val_size, batch_size):
            b1 = all_k1[i:i + batch_size]
            b_labels = all_labels[i:i + batch_size]

            if not is_pair:
                inputs = tokenizer(b1, return_tensors="pt", padding=True, truncation=True)
            else:
                b2 = all_k2[i:i + batch_size]
                inputs = tokenizer(b1, b2, return_tensors="pt", padding=True, truncation=True)

            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

            if label_map_tensor is not None:
                preds = label_map_tensor[preds]

            correct += (preds == torch.tensor(b_labels, device=device)).sum().item()
            total += len(b_labels)

            del inputs, outputs

    return correct / total if total > 0 else 0.0


# =========================================================================
# Pruning Benchmark Pipeline
# =========================================================================

def run_ae_pruning_experiment(task_name, checkpoint, dataset_name, dataset_subset,
                              val_split, is_pair, key1_name, key2_name, label_name, label_map=None,
                              device="cuda", batch_size=128, num_samples=1000, seed=555,
                              run_ae=True, run_random=True, run_inverse=True,
                              output_dir="experiments_results/glue/AE"):
    model_name = "BERT"
    total_steps = 144

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"STARTING STATIC AE EXPERIMENT: BERT on {task_name.upper()}")
    print(f"Checkpoint: {checkpoint}")
    methods_str = []
    if run_ae: methods_str.append("Static AE")
    if run_random: methods_str.append("Random")
    if run_inverse: methods_str.append("Inverse AE")
    print(f"Active Pruning Methods: {', '.join(methods_str)} | Total Steps: {total_steps}")
    if label_map is not None:
        print(f"Using Task Label Map: {label_map}")
    print("=" * 60 + "\n")

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    ds = load_dataset(dataset_name, dataset_subset)

    # 1. Evaluate baseline accuracy
    print("Loading fresh model for baseline evaluation & upfront AE computation...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, output_attentions=True)
    initial_acc = evaluate_accuracy(ds, model, tokenizer, val_split=val_split, is_pair=is_pair,
                                    key1_name=key1_name, key2_name=key2_name, label_name=label_name,
                                    label_map=label_map, device=device, batch_size=batch_size)
    print(f"Unpruned Baseline Accuracy: {initial_acc:.4f}")

    # 2. Compute upfront AE matrix (Static 1-shot calculation in memory)
    print(f"Computing Attention Entropy matrix upfront on {num_samples} calibration samples...")
    t_ae_start = sync_time()
    raw_ae_matrix = compute_raw_ae_matrix(ds, model, tokenizer, is_pair=is_pair,
                                          key1_name=key1_name, key2_name=key2_name,
                                          batch_size=batch_size, device=device,
                                          num_samples=num_samples, seed=seed)
    ae_calc_time = sync_time() - t_ae_start
    print(f"Upfront Attention Entropy computed in {ae_calc_time:.2f}s.")

    del model
    clear_memory()

    # Static AE pruning execution helper
    def execute_ae_pruning_run():
        print("\n--- Running Static Pruning: ATTENTION ENTROPY (AE) ---")
        model_run = AutoModelForSequenceClassification.from_pretrained(checkpoint, output_attentions=True)
        head_mask = torch.ones(12, 12)
        pruner = TransformerPruner(model_run)

        accs = [initial_acc]
        step_times = []
        total_start = sync_time()

        for step in range(1, total_steps + 1):
            s_start = sync_time()

            head_mask = get_new_head_mask_basedonAE(head_mask, raw_ae_matrix)
            pruner.prune(head_mask=head_mask, save_model=False)

            acc = evaluate_accuracy(ds, model_run, tokenizer, val_split=val_split, is_pair=is_pair,
                                    key1_name=key1_name, key2_name=key2_name, label_name=label_name,
                                    label_map=label_map, device=device, batch_size=batch_size)
            accs.append(acc)

            s_dur = sync_time() - s_start
            step_times.append(s_dur)

            if step % 20 == 0 or step == total_steps:
                print(f"[AE] Pruned {step:3d}/{total_steps} heads | Acc: {acc:.4f} | Step Time: {s_dur:.2f}s")

        total_dur = sync_time() - total_start
        print(f"[AE] Completed in {total_dur:.2f}s.")

        del model_run, pruner
        clear_memory()
        return accs, step_times, total_dur

    # Random pruning execution helper
    def execute_random_pruning_run():
        print("\n--- Running Baseline: RANDOM PRUNING ---")
        model_run = AutoModelForSequenceClassification.from_pretrained(checkpoint, output_attentions=True)
        head_mask = torch.ones(12, 12)
        pruner = TransformerPruner(model_run)

        rng = random.Random(seed)
        random_order = rng.sample(range(144), 144)

        accs = [initial_acc]
        step_times = []
        total_start = sync_time()

        for step in range(1, total_steps + 1):
            s_start = sync_time()
            head_idx = random_order[step - 1]
            layer = head_idx // 12
            head = head_idx % 12
            head_mask[layer][head] = 0

            pruner.prune(head_mask=head_mask, save_model=False)

            acc = evaluate_accuracy(ds, model_run, tokenizer, val_split=val_split, is_pair=is_pair,
                                    key1_name=key1_name, key2_name=key2_name, label_name=label_name,
                                    label_map=label_map, device=device, batch_size=batch_size)
            accs.append(acc)

            s_dur = sync_time() - s_start
            step_times.append(s_dur)

            if step % 20 == 0 or step == total_steps:
                print(f"[RANDOM] Pruned {step:3d}/{total_steps} heads | Acc: {acc:.4f} | Step Time: {s_dur:.2f}s")

        total_dur = sync_time() - total_start
        print(f"[RANDOM] Completed in {total_dur:.2f}s.")

        del model_run, pruner
        clear_memory()
        return accs, step_times, total_dur

    # Inverse AE pruning execution helper
    def execute_inverse_ae_pruning_run():
        print("\n--- Running Static Pruning: INVERSE ATTENTION ENTROPY (AE Inverse) ---")
        model_run = AutoModelForSequenceClassification.from_pretrained(checkpoint, output_attentions=True)
        head_mask = torch.ones(12, 12)
        pruner = TransformerPruner(model_run)

        accs = [initial_acc]
        step_times = []
        total_start = sync_time()

        for step in range(1, total_steps + 1):
            s_start = sync_time()

            head_mask = get_new_head_mask_basedonAE_inverse(head_mask, raw_ae_matrix)
            pruner.prune(head_mask=head_mask, save_model=False)

            acc = evaluate_accuracy(ds, model_run, tokenizer, val_split=val_split, is_pair=is_pair,
                                    key1_name=key1_name, key2_name=key2_name, label_name=label_name,
                                    label_map=label_map, device=device, batch_size=batch_size)
            accs.append(acc)

            s_dur = sync_time() - s_start
            step_times.append(s_dur)

            if step % 20 == 0 or step == total_steps:
                print(f"[AE-Inverse] Pruned {step:3d}/{total_steps} heads | Acc: {acc:.4f} | Step Time: {s_dur:.2f}s")

        total_dur = sync_time() - total_start
        print(f"[AE-Inverse] Completed in {total_dur:.2f}s.")

        del model_run, pruner
        clear_memory()
        return accs, step_times, total_dur

    # 3. Run Standard Static AE Pruning
    if run_ae:
        accs_ae, times_ae, tot_time_ae = execute_ae_pruning_run()
    else:
        accs_ae, times_ae, tot_time_ae = None, None, None

    # 4. Run Random Pruning Baseline
    if run_random:
        accs_rand, times_rand, tot_time_rand = execute_random_pruning_run()
    else:
        accs_rand, times_rand, tot_time_rand = None, None, None

    # 5. Run Inverse AE Pruning
    if run_inverse:
        accs_inv, times_inv, tot_time_inv = execute_inverse_ae_pruning_run()
    else:
        accs_inv, times_inv, tot_time_inv = None, None, None

    # 6. Save/Append Benchmark and Timing DataFrames
    heads_col = list(range(total_steps + 1))
    benchmark_csv = os.path.join(output_dir, f"BERT_{task_name}_ae_benchmark_seed_{seed}.csv")

    if os.path.exists(benchmark_csv):
        benchmark_df = pd.read_csv(benchmark_csv)
    else:
        benchmark_df = pd.DataFrame({"Heads Pruned": heads_col})

    if accs_ae is not None:
        benchmark_df["Accuracy_AE"] = accs_ae
    if accs_rand is not None:
        benchmark_df["Accuracy_Random"] = accs_rand
    if accs_inv is not None:
        benchmark_df["Accuracy_AE_Inverse"] = accs_inv

    benchmark_df.to_csv(benchmark_csv, index=False)
    print(f"\nSaved benchmark results to: {benchmark_csv}")

    timing_csv = os.path.join(output_dir, f"BERT_{task_name}_ae_timing_seed_{seed}.csv")
    if os.path.exists(timing_csv):
        timing_df = pd.read_csv(timing_csv)
    else:
        timing_df = pd.DataFrame({
            "Step": list(range(1, total_steps + 1)),
            "Heads Pruned": list(range(1, total_steps + 1))
        })

    if times_ae is not None:
        timing_df["Time_AE_sec"] = times_ae
    if times_rand is not None:
        timing_df["Time_Random_sec"] = times_rand
    if times_inv is not None:
        timing_df["Time_AE_Inverse_sec"] = times_inv

    timing_df.to_csv(timing_csv, index=False)
    print(f"Saved timing results to: {timing_csv}")

    print("\n" + "=" * 60)
    print(f"FINISHED: BERT on {task_name.upper()}")
    print("=" * 60 + "\n")


# =========================================================================
# Tasks Configuration
# =========================================================================

TASKS = {
    "sst2": {
        "checkpoint": "textattack/bert-base-uncased-SST-2",
        "dataset_name": "glue",
        "dataset_subset": "sst2",
        "val_split": "validation",
        "is_pair": False,
        "key1_name": "sentence",
        "key2_name": None,
        "label_name": "label",
        "label_map": None
    },
    "mnli": {
        "checkpoint": "textattack/bert-base-uncased-MNLI",
        "dataset_name": "glue",
        "dataset_subset": "mnli",
        "val_split": "validation_matched",
        "is_pair": True,
        "key1_name": "premise",
        "key2_name": "hypothesis",
        "label_name": "label",
        "label_map": [2, 0, 1]
    },
    "qnli": {
        "checkpoint": "textattack/bert-base-uncased-QNLI",
        "dataset_name": "glue",
        "dataset_subset": "qnli",
        "val_split": "validation",
        "is_pair": True,
        "key1_name": "question",
        "key2_name": "sentence",
        "label_name": "label",
        "label_map": None
    }
}


def main():
    parser = argparse.ArgumentParser(description="BERT Attention Entropy (AE) & Inverse AE Pruning Benchmark")
    parser.add_argument("--task", type=str, default="all", choices=["all", "sst2", "mnli", "qnli"], help="Task to evaluate (default: all)")
    parser.add_argument("--seed", type=int, default=555, help="Random seed (default: 555)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of calibration samples (default: 1000)")
    parser.add_argument("--no_ae", action="store_true", default=False, help="Disable standard AE pruning")
    parser.add_argument("--no_random", action="store_true", default=False, help="Disable Random pruning baseline")
    parser.add_argument("--no_inverse", action="store_true", default=False, help="Disable Inverse AE pruning")
    parser.add_argument("--output_dir", type=str, default="experiments_results/glue/AE", help="Output directory for benchmark CSVs")
    args = parser.parse_args()

    seed = args.seed
    print(f"Using random seed: {seed}")
    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Execution Device: {device}")

    selected_tasks = [args.task.lower()] if args.task != "all" else list(TASKS.keys())
    run_ae = not args.no_ae
    run_random = not args.no_random
    run_inv = not args.no_inverse

    for t in selected_tasks:
        cfg = TASKS[t]
        run_ae_pruning_experiment(
            task_name=t,
            checkpoint=cfg["checkpoint"],
            dataset_name=cfg["dataset_name"],
            dataset_subset=cfg["dataset_subset"],
            val_split=cfg["val_split"],
            is_pair=cfg["is_pair"],
            key1_name=cfg["key1_name"],
            key2_name=cfg["key2_name"],
            label_name=cfg["label_name"],
            label_map=cfg.get("label_map"),
            device=device,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            seed=seed,
            run_ae=run_ae,
            run_random=run_random,
            run_inverse=run_inv,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
