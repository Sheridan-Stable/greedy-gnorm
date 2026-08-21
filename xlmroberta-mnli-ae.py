import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from textpruner import TransformerPruner
import pandas as pd
from tqdm import tqdm
import gc
import torch.nn.functional as F
import torch.nn.modules.linear
import argparse
import random
import time

def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

parser = argparse.ArgumentParser(description="XLM-RoBERTa MNLI Attention Entropy (AE) Pruning Benchmark with Timing")
parser.add_argument("seed_pos", type=int, nargs="?", default=None, help="Random seed (positional argument)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for data sampling")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation and AE computation")
parser.add_argument("--num_samples", type=int, default=1000, help="Number of training samples used to compute AE")
parser.add_argument("--eps", type=float, default=1e-7, help="Epsilon for rectified entropy calculation")
parser.add_argument("--dynamic", action="store_true", default=False, help="Recompute AE scores dynamically at each step (default: False, runs static AE like the notebook)")
parser.add_argument("--run_inverse", action="store_true", default=True, help="Also run Inverse-AE pruning")
args, _ = parser.parse_known_args()

SEED = args.seed_pos if args.seed_pos is not None else args.seed
print(f"Using random seed: {SEED}")
print(f"Pruning mode: {'Dynamic AE' if args.dynamic else 'Static AE (Notebook default)'}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def sample_datapoints_pair(col1, col2, num_samples=1000, seed=SEED):
    total = len(col1)
    n = min(num_samples, total)
    rng = random.Random(seed)
    indices = rng.sample(range(total), n)
    return [col1[i] for i in indices], [col2[i] for i in indices]

os.makedirs("experiments_results/glue", exist_ok=True)

# Apply torch==2.0.1 monkeypatch for newer versions
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

def get_new_head_mask_basedonscore(head_mask_previous, scores):
    head_mask = head_mask_previous.clone()
    num_layers = head_mask.shape[0]
    
    min_score = float('inf')
    min_pos = (-1, -1)
    
    for layer in range(num_layers):
        for head in range(12):
            if head_mask[layer][head] == 1:
                if scores[layer][head] < min_score:
                    min_score = scores[layer][head]
                    min_pos = (layer, head)
                    
    if min_pos != (-1, -1):
        head_mask[min_pos[0]][min_pos[1]] = 0
        
    return head_mask

# ==========================================
# Task: MNLI
# ==========================================
print("\n" + "="*50)
print("Loading XLM-RoBERTa and MNLI Dataset...")
print("="*50 + "\n")

tokenizer = AutoTokenizer.from_pretrained("hiewbt/xlm-roberta-base-mnli")
model = AutoModelForSequenceClassification.from_pretrained("hiewbt/xlm-roberta-base-mnli", output_attentions=True)
ds = load_dataset("glue", "mnli")

def get_acc(ds, model, tokenizer, size=len(ds['validation_matched']), device=device, batch_size=args.batch_size):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_key1 = ds['validation_matched']['premise']
    all_key2 = ds['validation_matched']['hypothesis']
    all_labels = ds['validation_matched']['label']
    
    with torch.no_grad():
        for i in range(0, size, batch_size):
            b_key1 = all_key1[i:i+batch_size]
            b_key2 = all_key2[i:i+batch_size]
            b_labels = all_labels[i:i+batch_size]
            inputs = tokenizer(b_key1, b_key2, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            # Directly apply the mapping
            label_map = torch.tensor([0, 1, 2], device=device)
            preds = label_map[preds]
            
            correct += (preds == torch.tensor(b_labels).to(device)).sum().item()
            total += len(b_labels)
                        
    return correct / total

def get_ae_scores(ds, model, tokenizer, pruned_heads, batch_size=args.batch_size, device=device, eps=args.eps, inverse=False):
    """
    Computes epsilon-rectified Attention Entropy (AE) for active heads.
    - Standard AE (inverse=False): Prunes heads with highest entropy first.
    - Inverse AE (inverse=True): Prunes heads with lowest entropy first.
    """
    model.to(device)
    model.eval()
    entropy_accum = torch.zeros(12, 12, device=device)
    
    all_key1 = ds['train']['premise']
    all_key2 = ds['train']['hypothesis']

    # Sample num_samples for fast estimation
    num_samples = min(args.num_samples, len(all_key1))
    all_key1, all_key2 = sample_datapoints_pair(all_key1, all_key2, num_samples, seed=SEED)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            b_key1 = all_key1[i:i+batch_size]
            b_key2 = all_key2[i:i+batch_size]
            inputs = tokenizer(b_key1, b_key2, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            attns = outputs.attentions  # tuple of 12 tensors, each [batch, num_active_heads, seq_len, seq_len]

            for layer in range(12):
                A = attns[layer]
                if A.size(1) == 0:
                    continue
                
                # Epsilon-rectified entropy: - sum((A + eps) * log(A + eps)) averaged over query tokens
                # Shape: [batch, num_active_heads, seq_len] -> mean over query seq_len -> [batch, num_active_heads]
                token_entropy = (-(A + eps).log() * (A + eps)).sum(dim=-1).mean(dim=-1)
                batch_entropy = token_entropy.sum(dim=0)  # sum over batch -> [num_active_heads]
                
                # Map active heads back to the 12x12 grid
                pl = pruned_heads[layer]
                if isinstance(pl, torch.Tensor):
                    pl = pl.cpu().tolist()
                
                active_idx = 0
                for h_idx, keep in enumerate(pl):
                    if keep == 1:
                        entropy_accum[layer, h_idx] += batch_entropy[active_idx]
                        active_idx += 1

            del outputs, attns
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    mean_entropy = entropy_accum.cpu() / num_samples

    # get_new_head_mask_basedonscore selects the head with the minimum score.
    # For standard AE, highest entropy pruned first -> score = -mean_entropy.
    # For inverse AE, lowest entropy pruned first -> score = mean_entropy.
    if not inverse:
        scores = -mean_entropy
    else:
        scores = mean_entropy

    # Mask out already-pruned heads with inf
    for layer in range(12):
        pl = pruned_heads[layer]
        if isinstance(pl, torch.Tensor):
            pl = pl.cpu()
        else:
            pl = torch.tensor(pl).cpu()
        for head in range(12):
            if pl[head] == 0:
                scores[layer][head] = float('inf')

    return scores

def run_pruning(ds, model, tokenizer, method='ae', dynamic=args.dynamic):
    head_mask = torch.ones(12, 12)
    accs = []
    step_times = []
    initial_acc = get_acc(ds, model, tokenizer)
    accs.append(initial_acc)
    mode_str = "Dynamic" if dynamic else "Static"
    print(f"[{method.upper()} ({mode_str})] pruned heads: 0 acc: {initial_acc:.4f}")
    
    total_start = sync_time()
    
    # In static mode (notebook default), compute the AE matrix once upfront on the unpruned model
    if not dynamic:
        score_start = sync_time()
        if method == 'ae':
            static_scores = get_ae_scores(ds, model, tokenizer, pruned_heads=head_mask, inverse=False)
        elif method == 'ae_inverse':
            static_scores = get_ae_scores(ds, model, tokenizer, pruned_heads=head_mask, inverse=True)
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        score_time = sync_time() - score_start
        print(f"[{method.upper()} (Static)] Upfront AE score computation time: {score_time:.2f}s")
    else:
        score_time = 0.0

    # 1-by-1 pruning for 144 steps
    for step in range(1, 145):
        step_start = sync_time()
        
        if dynamic:
            if method == 'ae':
                scores = get_ae_scores(ds, model, tokenizer, pruned_heads=head_mask, inverse=False)
            elif method == 'ae_inverse':
                scores = get_ae_scores(ds, model, tokenizer, pruned_heads=head_mask, inverse=True)
            else:
                raise ValueError(f"Unknown pruning method: {method}")
        else:
            scores = static_scores.clone()
            
        head_mask = get_new_head_mask_basedonscore(head_mask, scores)
            
        pruner = TransformerPruner(model)
        pruner.prune(head_mask=head_mask, save_model=False)
        
        acc = get_acc(ds, model, tokenizer)
        accs.append(acc)
        step_end = sync_time()
        
        # In static mode, include the one-time scoring time in step 1 duration
        step_duration = (step_end - step_start) + (score_time if (step == 1 and not dynamic) else 0.0)
        step_times.append(step_duration)
        print(f"[{method.upper()} ({mode_str})] pruned heads: {step} acc: {acc:.4f} (step time: {step_duration:.2f}s)")
        
    total_end = sync_time()
    total_pruning_time = total_end - total_start
    print(f"[{method.upper()} ({mode_str})] Total pruning time: {total_pruning_time:.2f}s")
    return accs, step_times, total_pruning_time

# 1. Run Standard Attention Entropy (AE) Pruning
mode_str = "Dynamic" if args.dynamic else "Static"
print("\n" + "="*50)
print(f"Running XLM-RoBERTa MNLI Attention Entropy ({mode_str} AE) Pruning...")
print("="*50 + "\n")
accs_ae, times_ae, total_ae = run_pruning(ds, model, tokenizer, method='ae', dynamic=args.dynamic)

# 2. Run Inverse Attention Entropy Pruning (if enabled)
accs_ae_inv, times_ae_inv, total_ae_inv = None, None, None
if args.run_inverse:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "="*50)
    print(f"Running XLM-RoBERTa MNLI Inverse Attention Entropy ({mode_str} AE Inverse) Pruning...")
    print("="*50 + "\n")
    model = AutoModelForSequenceClassification.from_pretrained("hiewbt/xlm-roberta-base-mnli", output_attentions=True)
    accs_ae_inv, times_ae_inv, total_ae_inv = run_pruning(ds, model, tokenizer, method='ae_inverse', dynamic=args.dynamic)

mode_suffix = "dynamic" if args.dynamic else "static"

# Save Benchmark Accuracy CSV
heads_pruned = list(range(145))
bench_dict = {
    'Heads Pruned': heads_pruned,
    'Accuracy_AE': accs_ae
}
if accs_ae_inv is not None:
    bench_dict['Accuracy_AE_Inverse'] = accs_ae_inv

df_bench = pd.DataFrame(bench_dict)
bench_csv_path = f"experiments_results/glue/XLM_ROBERTA_mnli_ae_{mode_suffix}_benchmark_seed_{SEED}.csv"
df_bench.to_csv(bench_csv_path, index=False)
print(f"\nSaved benchmark accuracy to: {bench_csv_path}")
print(df_bench)

# Save Timing CSV
steps = list(range(1, 145))
timing_dict = {
    "Step": steps,
    "Heads Pruned": steps,
    "Time_AE_sec": times_ae
}
total_row_dict = {
    "Step": ["Total"],
    "Heads Pruned": ["All"],
    "Time_AE_sec": [total_ae]
}

if times_ae_inv is not None:
    timing_dict["Time_AE_Inverse_sec"] = times_ae_inv
    total_row_dict["Time_AE_Inverse_sec"] = [total_ae_inv]

df_timing = pd.DataFrame(timing_dict)
df_total = pd.DataFrame(total_row_dict)
df_timing = pd.concat([df_timing, df_total], ignore_index=True)

timing_csv_path = f"experiments_results/glue/XLM_ROBERTA_mnli_ae_{mode_suffix}_timing_seed_{SEED}.csv"
df_timing.to_csv(timing_csv_path, index=False)
print(f"\nSaved timing to: {timing_csv_path}")
print(df_timing)

del model, tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("\n" + "="*50)
print("Completed XLM-RoBERTa MNLI AE Pruning Benchmark.")
print("="*50 + "\n")
