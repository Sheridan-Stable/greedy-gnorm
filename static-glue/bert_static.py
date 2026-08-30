import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch
import torch.nn.functional as F
import torch.nn.modules.linear
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from textpruner import TransformerPruner
import pandas as pd
import traceback
import argparse
import random
import time
import gc

def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

# Apply torch==2.0.1 monkeypatch for linear layer replication
def _parse_torch_version(v_str):
    clean_v = v_str.split('+')[0].split('a')[0].split('b')[0].split('rc')[0]
    return tuple(int(x) for x in clean_v.split('.')[:3])

if _parse_torch_version(torch.__version__) > (2, 0, 1):
    if not getattr(F, "_is_monkeypatched", False):
        _orig_linear = F.linear

        def replication_safe_linear(input, weight, bias=None):
            if weight.size(1) == 0:
                out_shape = list(input.shape)
                out_shape[-1] = weight.size(0)
                zero_out = torch.zeros(out_shape, device=input.device, dtype=input.dtype)
                if bias is not None:
                    zero_out += bias
                return zero_out
            return _orig_linear(input, weight, bias)

        F.linear = replication_safe_linear
        torch.nn.modules.linear.F.linear = replication_safe_linear
        F._is_monkeypatched = True

def expand_weights_to_768x768(weight_matrix, active_heads_mask):
    head_size = 64
    num_heads = 12
    device = weight_matrix.device
    expanded_matrix = torch.zeros((768, 768), device=device)
    active_idx = 0
    tensors_to_concat = []
    
    for i in range(num_heads):
        if active_heads_mask[i] == 1:
            start_row = active_idx * head_size
            end_row = start_row + head_size
            extracted_tensor = weight_matrix[start_row:end_row, :]
            tensors_to_concat.append(extracted_tensor)
            active_idx += 1
        else:
            zero_tensor = torch.zeros((head_size, 768), device=device)
            tensors_to_concat.append(zero_tensor)
            
    expanded_tensor = torch.cat(tensors_to_concat, dim=0)
    return expanded_tensor

def sample_datapoints(dataset_column, num_samples=1000, seed=42):
    total = len(dataset_column)
    n = min(num_samples, total)
    rng = random.Random(seed)
    indices = rng.sample(range(total), n)
    return [dataset_column[i] for i in indices]

def sample_datapoints_pair(col1, col2, num_samples=1000, seed=42):
    total = len(col1)
    n = min(num_samples, total)
    rng = random.Random(seed)
    indices = rng.sample(range(total), n)
    return [col1[i] for i in indices], [col2[i] for i in indices]

def run_static_pruning(ds, model, tokenizer, importance_scores, task_name, device, method_name="Static"):
    """
    Executes one-shot static head pruning:
    - Sorts all 144 heads by importance_scores (ascending = least important first)
    - Sequentially prunes 1 head per step for 144 steps and records accuracy
    """
    head_list = []
    for layer in range(12):
        for head in range(12):
            head_list.append((importance_scores[layer][head].item(), layer, head))
            
    head_list.sort(key=lambda x: x[0])  # Ascending: prune smallest score first

    head_mask = torch.ones(12, 12)
    accs = []
    step_times = []

    initial_acc = evaluate_task_acc(ds, model, tokenizer, task_name, device)
    accs.append(initial_acc)
    print(f"[{method_name} - {task_name}] Step 0/144 | Initial Baseline Acc: {initial_acc:.4f}")

    total_start = sync_time()
    for step, (score_val, l_idx, h_idx) in enumerate(head_list, 1):
        step_start = sync_time()
        head_mask[l_idx][h_idx] = 0

        pruner = TransformerPruner(model)
        pruner.prune(head_mask=head_mask, save_model=False)

        acc = evaluate_task_acc(ds, model, tokenizer, task_name, device)
        accs.append(acc)
        step_end = sync_time()
        step_duration = step_end - step_start
        step_times.append(step_duration)

        rem_in_layer = int(head_mask[l_idx].sum().item())
        alert_str = " -> [ALERT: LAYER WIPED OUT (0 heads)]" if rem_in_layer == 0 else ""
        print(f"[{method_name} - {task_name}] Step {step:3d}/144 | Pruned: (L{l_idx:2d}, H{h_idx:2d}) | Score: {score_val:.4e} | Rem in L{l_idx:2d}: {rem_in_layer}/12 | Acc: {acc:.4f} ({step_duration:.2f}s){alert_str}")

    total_end = sync_time()
    total_pruning_time = total_end - total_start
    print(f"[{method_name} - {task_name}] Total pruning time: {total_pruning_time:.2f}s")
    return accs, step_times, total_pruning_time

def evaluate_task_acc(ds, model, tokenizer, task_name, device, batch_size=128):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    if task_name == "sst2":
        split = "validation"
        all_key1 = ds[split]['sentence']
        all_labels = ds[split]['label']
        with torch.no_grad():
            for i in range(0, len(all_key1), batch_size):
                b_key1 = all_key1[i:i+batch_size]
                b_labels = all_labels[i:i+batch_size]
                inputs = tokenizer(b_key1, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == torch.tensor(b_labels, device=device)).sum().item()
                total += len(b_labels)

    elif task_name == "mnli":
        split = "validation_matched"
        all_key1 = ds[split]['premise']
        all_key2 = ds[split]['hypothesis']
        all_labels = ds[split]['label']
        label_map = torch.tensor([2, 0, 1], device=device)
        with torch.no_grad():
            for i in range(0, len(all_key1), batch_size):
                b_key1 = all_key1[i:i+batch_size]
                b_key2 = all_key2[i:i+batch_size]
                b_labels = all_labels[i:i+batch_size]
                inputs = tokenizer(b_key1, b_key2, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                preds = label_map[torch.argmax(outputs.logits, dim=-1)]
                correct += (preds == torch.tensor(b_labels, device=device)).sum().item()
                total += len(b_labels)

    elif task_name == "qnli":
        split = "validation"
        all_key1 = ds[split]['question']
        all_key2 = ds[split]['sentence']
        all_labels = ds[split]['label']
        with torch.no_grad():
            for i in range(0, len(all_key1), batch_size):
                b_key1 = all_key1[i:i+batch_size]
                b_key2 = all_key2[i:i+batch_size]
                b_labels = all_labels[i:i+batch_size]
                inputs = tokenizer(b_key1, b_key2, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == torch.tensor(b_labels, device=device)).sum().item()
                total += len(b_labels)

    return correct / total if total > 0 else 0.0

def compute_static_taylor(ds, model, tokenizer, task_name, device, seed=42, batch_size=128):
    model.to(device)
    model.train()
    scores = torch.zeros(12, 12, device=device)

    if task_name == "sst2":
        all_k1 = sample_datapoints(ds['train']['sentence'], 1000, seed=seed)
        for i in range(0, len(all_k1), batch_size):
            b1 = all_k1[i:i+batch_size]
            inputs = {k: v.to(device) for k, v in tokenizer(b1, return_tensors='pt', padding=True, truncation=True).items()}
            outputs = model(**inputs)
            loss = torch.norm(outputs.logits)
            attns = outputs.attentions
            for a in attns: a.retain_grad()
            model.zero_grad()
            loss.backward()
            for layer in range(12):
                A, A_grad = attns[layer], attns[layer].grad
                if A_grad is None: continue
                inner = A * A_grad
                scores[layer] += torch.abs(inner.sum(dim=(2, 3))).sum(dim=0).detach()
            del outputs, loss, attns
            model.zero_grad()
    else:
        k1_name = 'premise' if task_name == 'mnli' else 'question'
        k2_name = 'hypothesis' if task_name == 'mnli' else 'sentence'
        all_k1, all_k2 = sample_datapoints_pair(ds['train'][k1_name], ds['train'][k2_name], 1000, seed=seed)
        for i in range(0, len(all_k1), batch_size):
            b1, b2 = all_k1[i:i+batch_size], all_k2[i:i+batch_size]
            inputs = {k: v.to(device) for k, v in tokenizer(b1, b2, return_tensors='pt', padding=True, truncation=True).items()}
            outputs = model(**inputs)
            loss = torch.norm(outputs.logits)
            attns = outputs.attentions
            for a in attns: a.retain_grad()
            model.zero_grad()
            loss.backward()
            for layer in range(12):
                A, A_grad = attns[layer], attns[layer].grad
                if A_grad is None: continue
                inner = A * A_grad
                scores[layer] += torch.abs(inner.sum(dim=(2, 3))).sum(dim=0).detach()
            del outputs, loss, attns
            model.zero_grad()

    return scores.cpu() / 1000

def compute_static_attattr(ds, model, tokenizer, task_name, device, seed=42, batch_size=128, m=20):
    model.to(device)
    model.train()
    scores = torch.zeros(12, 12, device=device)

    is_pair = (task_name != "sst2")
    k1_name = 'sentence' if not is_pair else ('premise' if task_name == 'mnli' else 'question')
    k2_name = None if not is_pair else ('hypothesis' if task_name == 'mnli' else 'sentence')

    if not is_pair:
        all_k1 = sample_datapoints(ds['train'][k1_name], 1000, seed=seed)
    else:
        all_k1, all_k2 = sample_datapoints_pair(ds['train'][k1_name], ds['train'][k2_name], 1000, seed=seed)

    for i in range(0, len(all_k1), batch_size):
        b1 = all_k1[i:i+batch_size]
        if not is_pair:
            inputs = tokenizer(b1, return_tensors='pt', padding=True, truncation=True)
        else:
            b2 = all_k2[i:i+batch_size]
            inputs = tokenizer(b1, b2, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        cur_len = len(b1)
        seq_len = inputs['input_ids'].size(1)

        batch_accum = {layer: torch.zeros(cur_len, 12, seq_len, seq_len, device=device) for layer in range(12)}

        for k in range(1, m + 1):
            alpha_mask = torch.tensor([k / m], dtype=torch.float32, device=device)
            outputs = model(**inputs, head_mask=alpha_mask)
            loss = torch.norm(outputs.logits)
            attns = outputs.attentions
            for a in attns: a.retain_grad()
            model.zero_grad()
            loss.backward()
            for layer in range(12):
                A, A_grad = attns[layer], attns[layer].grad
                if A_grad is None: continue
                batch_accum[layer] += (A * A_grad).detach() / m
            del outputs, loss, attns
            model.zero_grad()

        for layer in range(12):
            sample_max = batch_accum[layer].flatten(-2, -1).max(dim=-1).values
            scores[layer] += sample_max.sum(dim=0)
        del batch_accum
        torch.cuda.empty_cache()

    return torch.abs(scores.cpu() / 1000)

def run_bert_static(seed=42, output_dir="static-glue/experiments_results"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n==================== Running BERT Static GLUE Benchmark (Seed {seed}) ====================")

    tasks = [
        ("sst2", "textattack/bert-base-uncased-SST-2", "glue", "sst2"),
        ("mnli", "textattack/bert-base-uncased-MNLI", "glue", "mnli"),
        ("qnli", "textattack/bert-base-uncased-QNLI", "glue", "qnli")
    ]

    for task_name, model_ckpt, ds_name, ds_config in tasks:
        try:
            print(f"\n>>> Starting Task: BERT on {task_name.upper()}...")
            tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
            ds = load_dataset(ds_name, ds_config)

            # 1. Static Taylor
            model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, output_attentions=True)
            print(f"[BERT-{task_name.upper()}] Computing Static Taylor Scores...")
            scores_taylor = compute_static_taylor(ds, model, tokenizer, task_name, device, seed=seed)
            accs_taylor, times_taylor, tot_taylor = run_static_pruning(ds, model, tokenizer, scores_taylor, task_name, device, "Static_Taylor")
            del model; torch.cuda.empty_cache(); gc.collect()

            # 2. Static AttAttr
            model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, output_attentions=True)
            print(f"[BERT-{task_name.upper()}] Computing Static AttAttr Scores...")
            scores_attr = compute_static_attattr(ds, model, tokenizer, task_name, device, seed=seed)
            accs_attr, times_attr, tot_attr = run_static_pruning(ds, model, tokenizer, scores_attr, task_name, device, "Static_AttAttr")
            del model; torch.cuda.empty_cache(); gc.collect()

            # Save benchmark CSV
            heads = list(range(145))
            df = pd.DataFrame({
                'Heads Pruned': heads,
                'Accuracy_Static_Taylor': accs_taylor,
                'Accuracy_Static_AttAttr': accs_attr
            })
            bench_file = os.path.join(output_dir, f"BERT_{task_name}_static_benchmark_seed_{seed}.csv")
            df.to_csv(bench_file, index=False)
            print(f"Saved: {bench_file}")

            # Save timing CSV
            steps = list(range(1, 145))
            df_time = pd.DataFrame({
                'Step': steps,
                'Heads Pruned': steps,
                'Time_Static_Taylor_sec': times_taylor,
                'Time_Static_AttAttr_sec': times_attr
            })
            df_tot = pd.DataFrame({
                'Step': ['Total'],
                'Heads Pruned': ['All'],
                'Time_Static_Taylor_sec': [tot_taylor],
                'Time_Static_AttAttr_sec': [tot_attr]
            })
            df_time = pd.concat([df_time, df_tot], ignore_index=True)
            time_file = os.path.join(output_dir, f"BERT_{task_name}_static_timing_seed_{seed}.csv")
            df_time.to_csv(time_file, index=False)
            print(f"Saved: {time_file}")

        except Exception as e:
            print(f"ERROR processing BERT on {task_name.upper()}: {e}")
            traceback.print_exc()
            print("Skipping to next task...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT Static Pruning Benchmark")
    parser.add_argument("--seed", type=int, default=555, help="Random seed (default: 555)")
    args = parser.parse_args()
    run_bert_static(seed=args.seed)

