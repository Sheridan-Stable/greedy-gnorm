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


parser = argparse.ArgumentParser(description="ALBERT GLUE Benchmark with Head Pruning")
parser.add_argument("--seed", type=int, default=555, help="Random seed for data sampling")
args = parser.parse_args()

SEED = args.seed
print(f"Using random seed: {SEED}")

def sample_datapoints(dataset_column, num_samples=1000, seed=SEED):
    total = len(dataset_column)
    n = min(num_samples, total)
    rng = random.Random(seed)
    indices = rng.sample(range(total), n)
    return [dataset_column[i] for i in indices]

def sample_datapoints_pair(col1, col2, num_samples=1000, seed=SEED):
    total = len(col1)
    n = min(num_samples, total)
    rng = random.Random(seed)
    indices = rng.sample(range(total), n)
    return [col1[i] for i in indices], [col2[i] for i in indices]

os.makedirs("experiments_results", exist_ok=True)

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

def expand_weights_to_768x768(tensor, pruned_heads):
    tensors_to_concat = []
    start = 0
    for head_idx, keep in enumerate(pruned_heads):
        if keep == 1:
            tensors_to_concat.append(tensor[start:start+64, :])
            start += 64
        else:
            tensors_to_concat.append(torch.zeros(64, tensor.size(1), device=tensor.device))
    expanded_tensor = torch.cat(tensors_to_concat, dim=0)
    return expanded_tensor

def get_new_head_mask_basedonscore(head_mask_previous, scores):
    head_mask = head_mask_previous.clone()
    num_layers = head_mask.shape[0]
    num_heads = head_mask.shape[1]
    
    min_score = float("inf")
    min_pos = (-1, -1)
    
    for layer in range(num_layers):
        for head in range(num_heads):
            if head_mask[layer][head] == 1:
                if scores[layer][head] < min_score:
                    min_score = scores[layer][head]
                    min_pos = (layer, head)
                    
    if min_pos != (-1, -1):
        head_mask[min_pos[0]][min_pos[1]] = 0
        
    return head_mask

print("\n" + "="*50)
print("Completed setup and helper functions. Starting Task: SST2...")
print("="*50 + "\n")
#next#
# ==========================================
# Task: SST2 (ALBERT)
# ==========================================
print("Loading SST2...")
tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-SST-2")
model = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-SST-2", output_attentions=True)
ds = load_dataset("glue", "sst2")

def get_acc(ds, model, tokenizer, size=len(ds["validation"]), device="cuda", batch_size=64):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_key1 = ds["validation"]["sentence"]
    all_labels = ds["validation"]["label"]
    
    with torch.no_grad():
        for i in range(0, size, batch_size):
            b_key1 = all_key1[i:i+batch_size]
            b_labels = all_labels[i:i+batch_size]
            inputs = tokenizer(b_key1, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == torch.tensor(b_labels).to(device)).sum().item()
            total += len(b_labels)
            
    return correct / total

def get_gnorm_scores(ds, model, tokenizer, pruned_heads, batch_size=64, device="cuda"):
    model.to(device)
    model.train()
    norms_Q = torch.zeros(1, 12, device=device)
    norms_K = torch.zeros(1, 12, device=device)
    norms_V = torch.zeros(1, 12, device=device)
    
    all_k1 = ds["train"]["sentence"]
    num_samples = min(1000, len(all_k1))
    all_k1 = sample_datapoints(all_k1, num_samples, seed=SEED)

    attention_layer = model.albert.encoder.albert_layer_groups[0].albert_layers[0].attention
    heads_Q_weight = attention_layer.query.weight
    heads_K_weight = attention_layer.key.weight
    heads_V_weight = attention_layer.value.weight

    pl = pruned_heads[0]
    if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()

    for i in range(0, num_samples, batch_size):
        b1 = all_k1[i:i+batch_size]
        inputs = tokenizer(b1, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        loss = torch.norm(outputs.logits)
        model.zero_grad()
        loss.backward()

        GQ = expand_weights_to_768x768(heads_Q_weight.grad, pl)
        GK = expand_weights_to_768x768(heads_K_weight.grad, pl)
        GV = expand_weights_to_768x768(heads_V_weight.grad, pl)

        reshaped_Q = GQ.view(12, 64, 768)
        reshaped_K = GK.view(12, 64, 768)
        reshaped_V = GV.view(12, 64, 768)

        norms_Q[0] += torch.norm(reshaped_Q, p=2, dim=(1, 2)).detach()
        norms_K[0] += torch.norm(reshaped_K, p=2, dim=(1, 2)).detach()
        norms_V[0] += torch.norm(reshaped_V, p=2, dim=(1, 2)).detach()

        model.zero_grad()

    norms_Q = norms_Q.cpu() / num_samples
    norms_K = norms_K.cpu() / num_samples
    norms_V = norms_V.cpu() / num_samples
    scores = norms_Q * norms_K * norms_V

    for head in range(12):
        if pl[head] == 0:
            scores[0][head] = float("inf")

    return scores

def get_taylor_scores(ds, model, tokenizer, pruned_heads, batch_size=64, device="cuda"):
    model.to(device)
    model.train()
    scores = torch.zeros(1, 12, device=device)
    
    all_k1 = ds["train"]["sentence"]
    num_samples = min(1000, len(all_k1))
    all_k1 = sample_datapoints(all_k1, num_samples, seed=SEED)

    for i in range(0, num_samples, batch_size):
        b1 = all_k1[i:i+batch_size]
        inputs = tokenizer(b1, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        alpha_mask = torch.tensor([1.0], dtype=torch.float32, device=device)
        outputs = model(**inputs, head_mask=alpha_mask)
        loss = torch.norm(outputs.logits)

        attns = outputs.attentions
        for a in attns:
            a.retain_grad()

        model.zero_grad()
        loss.backward()

        temp_scores = torch.zeros(1, 12, device=device)
        for layer in range(12):
            A = attns[layer]
            A_grad = A.grad
            if A_grad is None or A.size(1) == 0:
                continue

            inner = A * A_grad
            batch_scores = torch.abs(inner.sum(dim=(2, 3))).sum(dim=0).detach()

            exp_scores = torch.zeros(12, device=device)
            active_idx = 0
            pl = pruned_heads[0]
            if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()

            for h_idx, keep in enumerate(pl):
                if keep == 1:
                    exp_scores[h_idx] = batch_scores[active_idx]
                    active_idx += 1
            temp_scores[0] += exp_scores

        scores += temp_scores
        del outputs, loss, attns
        torch.cuda.empty_cache()

    scores = scores.cpu() / num_samples

    pl = pruned_heads[0]
    if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()
    for head in range(12):
        if pl[head] == 0:
            scores[0][head] = float("inf")

    return scores

def get_attr_scores(ds, model, tokenizer, pruned_heads, batch_size=64, device="cuda", m=10):
    model.to(device)
    model.train()
    scores = torch.zeros(1, 12, device=device)
    
    all_k1 = ds["train"]["sentence"]
    num_samples = min(1000, len(all_k1))
    all_k1 = sample_datapoints(all_k1, num_samples, seed=SEED)

    for i in range(0, num_samples, batch_size):
        b1 = all_k1[i:i+batch_size]
        inputs = tokenizer(b1, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        current_batch_len = len(b1)
        seq_len = inputs['input_ids'].size(1)

        batch_path_accum = {
            layer: torch.zeros(current_batch_len, 12, seq_len, seq_len, device=device)
            for layer in range(12)
        }

        for k in range(1, m + 1):
            alpha = k / m
            alpha_mask = torch.tensor([alpha], dtype=torch.float32, device=device)

            outputs = model(**inputs, head_mask=alpha_mask)
            loss = torch.norm(outputs.logits)

            attns = outputs.attentions
            for a in attns:
                a.retain_grad()

            model.zero_grad()
            loss.backward()

            for layer in range(12):
                A = attns[layer]
                A_grad = A.grad
                if A_grad is None or A.size(1) == 0:
                    continue

                inner = (A * A_grad).detach()
                exp_inner = torch.zeros(current_batch_len, 12, seq_len, seq_len, device=device)
                active_idx = 0
                pl = pruned_heads[0]
                if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()

                for h_idx, keep in enumerate(pl):
                    if keep == 1:
                        exp_inner[:, h_idx, :, :] = inner[:, active_idx, :, :]
                        active_idx += 1

                batch_path_accum[layer] += exp_inner / m

            del outputs, loss, attns
            model.zero_grad()

        for layer in range(12):
            sample_max = batch_path_accum[layer].flatten(-2, -1).max(dim=-1).values
            scores[0] += sample_max.sum(dim=0)

        del batch_path_accum
        torch.cuda.empty_cache()

    scores = torch.abs(scores.cpu() / num_samples)

    pl = pruned_heads[0]
    if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()
    for head in range(12):
        if pl[head] == 0:
            scores[0][head] = float("inf")

    return scores

def run_pruning(ds, model, tokenizer, method="gnorm"):
    head_mask = torch.ones(1, 12)
    accs = []
    step_times = []
    initial_acc = get_acc(ds, model, tokenizer)
    accs.append(initial_acc)
    print(f"[{method}] pruned heads: 0 acc: {initial_acc}")
    
    total_start = sync_time()
    for i in range(12):
        step_start = sync_time()
        if method == "gnorm":
            scores = get_gnorm_scores(ds, model, tokenizer, pruned_heads=head_mask)
        elif method == "taylor":
            scores = get_taylor_scores(ds, model, tokenizer, pruned_heads=head_mask)
        else:
            scores = get_attr_scores(ds, model, tokenizer, pruned_heads=head_mask)
            
        head_mask = get_new_head_mask_basedonscore(head_mask, scores)
            
        pruner = TransformerPruner(model)
        pruner.prune(head_mask=head_mask, save_model=False)
        
        acc = get_acc(ds, model, tokenizer)
        accs.append(acc)
        step_end = sync_time()
        step_duration = step_end - step_start
        step_times.append(step_duration)
        print(f"[{method}] pruned heads: {(i+1)*12} acc: {acc} (step time: {step_duration:.2f}s)")
        
    total_end = sync_time()
    total_pruning_time = total_end - total_start
    print(f"[{method}] Total pruning time: {total_pruning_time:.2f}s")
    return accs, step_times, total_pruning_time

# Run Greedy Gnorm
print("Running Gnorm...")
accs_gnorm, times_gnorm, total_gnorm = run_pruning(ds, model, tokenizer, method="gnorm")

# Reload model for Taylor
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()
tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-SST-2")
model = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-SST-2", output_attentions=True)

print("Running Taylor...")
accs_taylor, times_taylor, total_taylor = run_pruning(ds, model, tokenizer, method="taylor")

# Reload model for Attribution
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()
tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-SST-2")
model = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-SST-2", output_attentions=True)

print("Running Attribution...")
accs_attr, times_attr, total_attr = run_pruning(ds, model, tokenizer, method="attr")

heads_pruned = [i * 12 for i in range(13)]
df = pd.DataFrame({"Heads Pruned": heads_pruned, "Accuracy_Gnorm": accs_gnorm, "Accuracy_Taylor": accs_taylor, "Accuracy_Attr": accs_attr})
df.to_csv(f"experiments_results/glue/ALBERT_sst2_benchmark_seed_{SEED}.csv", index=False)
print(df)

steps = list(range(1, 13))
df_timing = pd.DataFrame({
    "Step": steps,
    "Heads Pruned": [i * 12 for i in steps],
    "Time_Gnorm_sec": times_gnorm,
    "Time_Taylor_sec": times_taylor,
    "Time_Attr_sec": times_attr
})
df_total = pd.DataFrame({
    "Step": ["Total"],
    "Heads Pruned": ["All"],
    "Time_Gnorm_sec": [total_gnorm],
    "Time_Taylor_sec": [total_taylor],
    "Time_Attr_sec": [total_attr]
})
df_timing = pd.concat([df_timing, df_total], ignore_index=True)
df_timing.to_csv(f"experiments_results/glue/ALBERT_sst2_timing_seed_{SEED}.csv", index=False)
print(df_timing)

del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

print("\n" + "="*50)
print("Completed Task: SST2 benchmark. Starting Task: MNLI...")
print("="*50 + "\n")
#next#



# ==========================================
# Task: MNLI (ALBERT)
# ==========================================
print("Loading MNLI...")
tokenizer = AutoTokenizer.from_pretrained("Alireza1044/albert-base-v2-mnli")
model = AutoModelForSequenceClassification.from_pretrained("Alireza1044/albert-base-v2-mnli", output_attentions=True)
ds = load_dataset("glue", "mnli")

def get_acc(ds, model, tokenizer, size=len(ds["validation_matched"]), device="cuda", batch_size=64):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_key1 = ds["validation_matched"]["premise"]
    all_key2 = ds["validation_matched"]["hypothesis"]
    all_labels = ds["validation_matched"]["label"]
    
    with torch.no_grad():
        for i in range(0, size, batch_size):
            b_key1 = all_key1[i:i+batch_size]
            b_key2 = all_key2[i:i+batch_size]
            b_labels = all_labels[i:i+batch_size]
            inputs = tokenizer(b_key1, b_key2, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == torch.tensor(b_labels).to(device)).sum().item()
            total += len(b_labels)
            
    return correct / total

def get_gnorm_scores(ds, model, tokenizer, pruned_heads, batch_size=64, device="cuda"):
    model.to(device)
    model.train()
    norms_Q = torch.zeros(1, 12, device=device)
    norms_K = torch.zeros(1, 12, device=device)
    norms_V = torch.zeros(1, 12, device=device)
    
    all_k1 = ds["train"]["premise"]
    all_k2 = ds["train"]["hypothesis"]
    num_samples = min(1000, len(all_k1))
    all_k1, all_k2 = sample_datapoints_pair(all_k1, all_k2, num_samples, seed=SEED)

    attention_layer = model.albert.encoder.albert_layer_groups[0].albert_layers[0].attention
    heads_Q_weight = attention_layer.query.weight
    heads_K_weight = attention_layer.key.weight
    heads_V_weight = attention_layer.value.weight

    pl = pruned_heads[0]
    if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()

    for i in range(0, num_samples, batch_size):
        b1 = all_k1[i:i+batch_size]
        b2 = all_k2[i:i+batch_size]
        inputs = tokenizer(b1, b2, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        loss = torch.norm(outputs.logits)
        model.zero_grad()
        loss.backward()

        GQ = expand_weights_to_768x768(heads_Q_weight.grad, pl)
        GK = expand_weights_to_768x768(heads_K_weight.grad, pl)
        GV = expand_weights_to_768x768(heads_V_weight.grad, pl)

        reshaped_Q = GQ.view(12, 64, 768)
        reshaped_K = GK.view(12, 64, 768)
        reshaped_V = GV.view(12, 64, 768)

        norms_Q[0] += torch.norm(reshaped_Q, p=2, dim=(1, 2)).detach()
        norms_K[0] += torch.norm(reshaped_K, p=2, dim=(1, 2)).detach()
        norms_V[0] += torch.norm(reshaped_V, p=2, dim=(1, 2)).detach()

        model.zero_grad()

    norms_Q = norms_Q.cpu() / num_samples
    norms_K = norms_K.cpu() / num_samples
    norms_V = norms_V.cpu() / num_samples
    scores = norms_Q * norms_K * norms_V

    for head in range(12):
        if pl[head] == 0:
            scores[0][head] = float("inf")

    return scores

def get_taylor_scores(ds, model, tokenizer, pruned_heads, batch_size=64, device="cuda"):
    model.to(device)
    model.train()
    scores = torch.zeros(1, 12, device=device)
    
    all_k1 = ds["train"]["premise"]
    all_k2 = ds["train"]["hypothesis"]
    num_samples = min(1000, len(all_k1))
    all_k1, all_k2 = sample_datapoints_pair(all_k1, all_k2, num_samples, seed=SEED)

    for i in range(0, num_samples, batch_size):
        b1 = all_k1[i:i+batch_size]
        b2 = all_k2[i:i+batch_size]
        inputs = tokenizer(b1, b2, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        alpha_mask = torch.tensor([1.0], dtype=torch.float32, device=device)
        outputs = model(**inputs, head_mask=alpha_mask)
        loss = torch.norm(outputs.logits)

        attns = outputs.attentions
        for a in attns:
            a.retain_grad()

        model.zero_grad()
        loss.backward()

        temp_scores = torch.zeros(1, 12, device=device)
        for layer in range(12):
            A = attns[layer]
            A_grad = A.grad
            if A_grad is None or A.size(1) == 0:
                continue

            inner = A * A_grad
            batch_scores = torch.abs(inner.sum(dim=(2, 3))).sum(dim=0).detach()

            exp_scores = torch.zeros(12, device=device)
            active_idx = 0
            pl = pruned_heads[0]
            if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()

            for h_idx, keep in enumerate(pl):
                if keep == 1:
                    exp_scores[h_idx] = batch_scores[active_idx]
                    active_idx += 1
            temp_scores[0] += exp_scores

        scores += temp_scores
        del outputs, loss, attns
        torch.cuda.empty_cache()

    scores = scores.cpu() / num_samples

    pl = pruned_heads[0]
    if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()
    for head in range(12):
        if pl[head] == 0:
            scores[0][head] = float("inf")

    return scores

def get_attr_scores(ds, model, tokenizer, pruned_heads, batch_size=64, device="cuda", m=10):
    model.to(device)
    model.train()
    scores = torch.zeros(1, 12, device=device)
    
    all_k1 = ds["train"]["premise"]
    all_k2 = ds["train"]["hypothesis"]
    num_samples = min(1000, len(all_k1))
    all_k1, all_k2 = sample_datapoints_pair(all_k1, all_k2, num_samples, seed=SEED)

    for i in range(0, num_samples, batch_size):
        b1 = all_k1[i:i+batch_size]
        b2 = all_k2[i:i+batch_size]
        inputs = tokenizer(b1, b2, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        current_batch_len = len(b1)
        seq_len = inputs['input_ids'].size(1)

        batch_path_accum = {
            layer: torch.zeros(current_batch_len, 12, seq_len, seq_len, device=device)
            for layer in range(12)
        }

        for k in range(1, m + 1):
            alpha = k / m
            alpha_mask = torch.tensor([alpha], dtype=torch.float32, device=device)

            outputs = model(**inputs, head_mask=alpha_mask)
            loss = torch.norm(outputs.logits)

            attns = outputs.attentions
            for a in attns:
                a.retain_grad()

            model.zero_grad()
            loss.backward()

            for layer in range(12):
                A = attns[layer]
                A_grad = A.grad
                if A_grad is None or A.size(1) == 0:
                    continue

                inner = (A * A_grad).detach()
                exp_inner = torch.zeros(current_batch_len, 12, seq_len, seq_len, device=device)
                active_idx = 0
                pl = pruned_heads[0]
                if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()

                for h_idx, keep in enumerate(pl):
                    if keep == 1:
                        exp_inner[:, h_idx, :, :] = inner[:, active_idx, :, :]
                        active_idx += 1

                batch_path_accum[layer] += exp_inner / m

            del outputs, loss, attns
            model.zero_grad()

        for layer in range(12):
            sample_max = batch_path_accum[layer].flatten(-2, -1).max(dim=-1).values
            scores[0] += sample_max.sum(dim=0)

        del batch_path_accum
        torch.cuda.empty_cache()

    scores = torch.abs(scores.cpu() / num_samples)

    pl = pruned_heads[0]
    if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()
    for head in range(12):
        if pl[head] == 0:
            scores[0][head] = float("inf")

    return scores

def run_pruning(ds, model, tokenizer, method="gnorm"):
    head_mask = torch.ones(1, 12)
    accs = []
    step_times = []
    initial_acc = get_acc(ds, model, tokenizer)
    accs.append(initial_acc)
    print(f"[{method}] pruned heads: 0 acc: {initial_acc}")
    
    total_start = sync_time()
    for i in range(12):
        step_start = sync_time()
        if method == "gnorm":
            scores = get_gnorm_scores(ds, model, tokenizer, pruned_heads=head_mask)
        elif method == "taylor":
            scores = get_taylor_scores(ds, model, tokenizer, pruned_heads=head_mask)
        else:
            scores = get_attr_scores(ds, model, tokenizer, pruned_heads=head_mask)
            
        head_mask = get_new_head_mask_basedonscore(head_mask, scores)
            
        pruner = TransformerPruner(model)
        pruner.prune(head_mask=head_mask, save_model=False)
        
        acc = get_acc(ds, model, tokenizer)
        accs.append(acc)
        step_end = sync_time()
        step_duration = step_end - step_start
        step_times.append(step_duration)
        print(f"[{method}] pruned heads: {(i+1)*12} acc: {acc} (step time: {step_duration:.2f}s)")
        
    total_end = sync_time()
    total_pruning_time = total_end - total_start
    print(f"[{method}] Total pruning time: {total_pruning_time:.2f}s")
    return accs, step_times, total_pruning_time

# Run Greedy Gnorm
print("Running Gnorm...")
accs_gnorm, times_gnorm, total_gnorm = run_pruning(ds, model, tokenizer, method="gnorm")

# Reload model for Taylor
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()
tokenizer = AutoTokenizer.from_pretrained("Alireza1044/albert-base-v2-mnli")
model = AutoModelForSequenceClassification.from_pretrained("Alireza1044/albert-base-v2-mnli", output_attentions=True)

print("Running Taylor...")
accs_taylor, times_taylor, total_taylor = run_pruning(ds, model, tokenizer, method="taylor")

# Reload model for Attribution
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()
tokenizer = AutoTokenizer.from_pretrained("Alireza1044/albert-base-v2-mnli")
model = AutoModelForSequenceClassification.from_pretrained("Alireza1044/albert-base-v2-mnli", output_attentions=True)

print("Running Attribution...")
accs_attr, times_attr, total_attr = run_pruning(ds, model, tokenizer, method="attr")

heads_pruned = [i * 12 for i in range(13)]
df = pd.DataFrame({"Heads Pruned": heads_pruned, "Accuracy_Gnorm": accs_gnorm, "Accuracy_Taylor": accs_taylor, "Accuracy_Attr": accs_attr})
df.to_csv(f"experiments_results/glue/ALBERT_mnli_benchmark_seed_{SEED}.csv", index=False)
print(df)

steps = list(range(1, 13))
df_timing = pd.DataFrame({
    "Step": steps,
    "Heads Pruned": [i * 12 for i in steps],
    "Time_Gnorm_sec": times_gnorm,
    "Time_Taylor_sec": times_taylor,
    "Time_Attr_sec": times_attr
})
df_total = pd.DataFrame({
    "Step": ["Total"],
    "Heads Pruned": ["All"],
    "Time_Gnorm_sec": [total_gnorm],
    "Time_Taylor_sec": [total_taylor],
    "Time_Attr_sec": [total_attr]
})
df_timing = pd.concat([df_timing, df_total], ignore_index=True)
df_timing.to_csv(f"experiments_results/glue/ALBERT_mnli_timing_seed_{SEED}.csv", index=False)
print(df_timing)

del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

print("\n" + "="*50)
print("Completed Task: MNLI benchmark. Starting Task: QNLI...")
print("="*50 + "\n")
#next#



# ==========================================
# Task: QNLI (ALBERT)
# ==========================================
print("Loading QNLI...")
tokenizer = AutoTokenizer.from_pretrained("Alireza1044/albert-base-v2-qnli")
model = AutoModelForSequenceClassification.from_pretrained("Alireza1044/albert-base-v2-qnli", output_attentions=True)
ds = load_dataset("glue", "qnli")

def get_acc(ds, model, tokenizer, size=len(ds["validation"]), device="cuda", batch_size=64):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_key1 = ds["validation"]["question"]
    all_key2 = ds["validation"]["sentence"]
    all_labels = ds["validation"]["label"]
    
    with torch.no_grad():
        for i in range(0, size, batch_size):
            b_key1 = all_key1[i:i+batch_size]
            b_key2 = all_key2[i:i+batch_size]
            b_labels = all_labels[i:i+batch_size]
            inputs = tokenizer(b_key1, b_key2, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == torch.tensor(b_labels).to(device)).sum().item()
            total += len(b_labels)
            
    return correct / total

def get_gnorm_scores(ds, model, tokenizer, pruned_heads, batch_size=64, device="cuda"):
    model.to(device)
    model.train()
    norms_Q = torch.zeros(1, 12, device=device)
    norms_K = torch.zeros(1, 12, device=device)
    norms_V = torch.zeros(1, 12, device=device)
    
    all_k1 = ds["train"]["question"]
    all_k2 = ds["train"]["sentence"]
    num_samples = min(1000, len(all_k1))
    all_k1, all_k2 = sample_datapoints_pair(all_k1, all_k2, num_samples, seed=SEED)

    attention_layer = model.albert.encoder.albert_layer_groups[0].albert_layers[0].attention
    heads_Q_weight = attention_layer.query.weight
    heads_K_weight = attention_layer.key.weight
    heads_V_weight = attention_layer.value.weight

    pl = pruned_heads[0]
    if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()

    for i in range(0, num_samples, batch_size):
        b1 = all_k1[i:i+batch_size]
        b2 = all_k2[i:i+batch_size]
        inputs = tokenizer(b1, b2, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        loss = torch.norm(outputs.logits)
        model.zero_grad()
        loss.backward()

        GQ = expand_weights_to_768x768(heads_Q_weight.grad, pl)
        GK = expand_weights_to_768x768(heads_K_weight.grad, pl)
        GV = expand_weights_to_768x768(heads_V_weight.grad, pl)

        reshaped_Q = GQ.view(12, 64, 768)
        reshaped_K = GK.view(12, 64, 768)
        reshaped_V = GV.view(12, 64, 768)

        norms_Q[0] += torch.norm(reshaped_Q, p=2, dim=(1, 2)).detach()
        norms_K[0] += torch.norm(reshaped_K, p=2, dim=(1, 2)).detach()
        norms_V[0] += torch.norm(reshaped_V, p=2, dim=(1, 2)).detach()

        model.zero_grad()

    norms_Q = norms_Q.cpu() / num_samples
    norms_K = norms_K.cpu() / num_samples
    norms_V = norms_V.cpu() / num_samples
    scores = norms_Q * norms_K * norms_V

    for head in range(12):
        if pl[head] == 0:
            scores[0][head] = float("inf")

    return scores

def get_taylor_scores(ds, model, tokenizer, pruned_heads, batch_size=64, device="cuda"):
    model.to(device)
    model.train()
    scores = torch.zeros(1, 12, device=device)
    
    all_k1 = ds["train"]["question"]
    all_k2 = ds["train"]["sentence"]
    num_samples = min(1000, len(all_k1))
    all_k1, all_k2 = sample_datapoints_pair(all_k1, all_k2, num_samples, seed=SEED)

    for i in range(0, num_samples, batch_size):
        b1 = all_k1[i:i+batch_size]
        b2 = all_k2[i:i+batch_size]
        inputs = tokenizer(b1, b2, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        alpha_mask = torch.tensor([1.0], dtype=torch.float32, device=device)
        outputs = model(**inputs, head_mask=alpha_mask)
        loss = torch.norm(outputs.logits)

        attns = outputs.attentions
        for a in attns:
            a.retain_grad()

        model.zero_grad()
        loss.backward()

        temp_scores = torch.zeros(1, 12, device=device)
        for layer in range(12):
            A = attns[layer]
            A_grad = A.grad
            if A_grad is None or A.size(1) == 0:
                continue

            inner = A * A_grad
            batch_scores = torch.abs(inner.sum(dim=(2, 3))).sum(dim=0).detach()

            exp_scores = torch.zeros(12, device=device)
            active_idx = 0
            pl = pruned_heads[0]
            if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()

            for h_idx, keep in enumerate(pl):
                if keep == 1:
                    exp_scores[h_idx] = batch_scores[active_idx]
                    active_idx += 1
            temp_scores[0] += exp_scores

        scores += temp_scores
        del outputs, loss, attns
        torch.cuda.empty_cache()

    scores = scores.cpu() / num_samples

    pl = pruned_heads[0]
    if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()
    for head in range(12):
        if pl[head] == 0:
            scores[0][head] = float("inf")

    return scores

def get_attr_scores(ds, model, tokenizer, pruned_heads, batch_size=64, device="cuda", m=10):
    model.to(device)
    model.train()
    scores = torch.zeros(1, 12, device=device)
    
    all_k1 = ds["train"]["question"]
    all_k2 = ds["train"]["sentence"]
    num_samples = min(1000, len(all_k1))
    all_k1, all_k2 = sample_datapoints_pair(all_k1, all_k2, num_samples, seed=SEED)

    for i in range(0, num_samples, batch_size):
        b1 = all_k1[i:i+batch_size]
        b2 = all_k2[i:i+batch_size]
        inputs = tokenizer(b1, b2, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        current_batch_len = len(b1)
        seq_len = inputs['input_ids'].size(1)

        batch_path_accum = {
            layer: torch.zeros(current_batch_len, 12, seq_len, seq_len, device=device)
            for layer in range(12)
        }

        for k in range(1, m + 1):
            alpha = k / m
            alpha_mask = torch.tensor([alpha], dtype=torch.float32, device=device)

            outputs = model(**inputs, head_mask=alpha_mask)
            loss = torch.norm(outputs.logits)

            attns = outputs.attentions
            for a in attns:
                a.retain_grad()

            model.zero_grad()
            loss.backward()

            for layer in range(12):
                A = attns[layer]
                A_grad = A.grad
                if A_grad is None or A.size(1) == 0:
                    continue

                inner = (A * A_grad).detach()
                exp_inner = torch.zeros(current_batch_len, 12, seq_len, seq_len, device=device)
                active_idx = 0
                pl = pruned_heads[0]
                if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()

                for h_idx, keep in enumerate(pl):
                    if keep == 1:
                        exp_inner[:, h_idx, :, :] = inner[:, active_idx, :, :]
                        active_idx += 1

                batch_path_accum[layer] += exp_inner / m

            del outputs, loss, attns
            model.zero_grad()

        for layer in range(12):
            sample_max = batch_path_accum[layer].flatten(-2, -1).max(dim=-1).values
            scores[0] += sample_max.sum(dim=0)

        del batch_path_accum
        torch.cuda.empty_cache()

    scores = torch.abs(scores.cpu() / num_samples)

    pl = pruned_heads[0]
    if isinstance(pl, torch.Tensor): pl = pl.cpu().tolist()
    for head in range(12):
        if pl[head] == 0:
            scores[0][head] = float("inf")

    return scores

def run_pruning(ds, model, tokenizer, method="gnorm"):
    head_mask = torch.ones(1, 12)
    accs = []
    step_times = []
    initial_acc = get_acc(ds, model, tokenizer)
    accs.append(initial_acc)
    print(f"[{method}] pruned heads: 0 acc: {initial_acc}")
    
    total_start = sync_time()
    for i in range(12):
        step_start = sync_time()
        if method == "gnorm":
            scores = get_gnorm_scores(ds, model, tokenizer, pruned_heads=head_mask)
        elif method == "taylor":
            scores = get_taylor_scores(ds, model, tokenizer, pruned_heads=head_mask)
        else:
            scores = get_attr_scores(ds, model, tokenizer, pruned_heads=head_mask)
            
        head_mask = get_new_head_mask_basedonscore(head_mask, scores)
            
        pruner = TransformerPruner(model)
        pruner.prune(head_mask=head_mask, save_model=False)
        
        acc = get_acc(ds, model, tokenizer)
        accs.append(acc)
        step_end = sync_time()
        step_duration = step_end - step_start
        step_times.append(step_duration)
        print(f"[{method}] pruned heads: {(i+1)*12} acc: {acc} (step time: {step_duration:.2f}s)")
        
    total_end = sync_time()
    total_pruning_time = total_end - total_start
    print(f"[{method}] Total pruning time: {total_pruning_time:.2f}s")
    return accs, step_times, total_pruning_time

# Run Greedy Gnorm
print("Running Gnorm...")
accs_gnorm, times_gnorm, total_gnorm = run_pruning(ds, model, tokenizer, method="gnorm")

# Reload model for Taylor
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()
tokenizer = AutoTokenizer.from_pretrained("Alireza1044/albert-base-v2-qnli")
model = AutoModelForSequenceClassification.from_pretrained("Alireza1044/albert-base-v2-qnli", output_attentions=True)

print("Running Taylor...")
accs_taylor, times_taylor, total_taylor = run_pruning(ds, model, tokenizer, method="taylor")

# Reload model for Attribution
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()
tokenizer = AutoTokenizer.from_pretrained("Alireza1044/albert-base-v2-qnli")
model = AutoModelForSequenceClassification.from_pretrained("Alireza1044/albert-base-v2-qnli", output_attentions=True)

print("Running Attribution...")
accs_attr, times_attr, total_attr = run_pruning(ds, model, tokenizer, method="attr")

heads_pruned = [i * 12 for i in range(13)]
df = pd.DataFrame({"Heads Pruned": heads_pruned, "Accuracy_Gnorm": accs_gnorm, "Accuracy_Taylor": accs_taylor, "Accuracy_Attr": accs_attr})
df.to_csv(f"experiments_results/glue/ALBERT_qnli_benchmark_seed_{SEED}.csv", index=False)
print(df)

steps = list(range(1, 13))
df_timing = pd.DataFrame({
    "Step": steps,
    "Heads Pruned": [i * 12 for i in steps],
    "Time_Gnorm_sec": times_gnorm,
    "Time_Taylor_sec": times_taylor,
    "Time_Attr_sec": times_attr
})
df_total = pd.DataFrame({
    "Step": ["Total"],
    "Heads Pruned": ["All"],
    "Time_Gnorm_sec": [total_gnorm],
    "Time_Taylor_sec": [total_taylor],
    "Time_Attr_sec": [total_attr]
})
df_timing = pd.concat([df_timing, df_total], ignore_index=True)
df_timing.to_csv(f"experiments_results/glue/ALBERT_qnli_timing_seed_{SEED}.csv", index=False)
print(df_timing)

del model, tokenizer
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
gc.collect()

print("\n" + "="*50)
print("Completed Task: QNLI benchmark. All tasks completed successfully!")
print("="*50 + "\n")

