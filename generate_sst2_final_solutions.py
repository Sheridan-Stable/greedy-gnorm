"""
generate_sst2_final_solutions.py

Iteratively prunes attention heads dynamically using Greedy-Gnorm and TransformerPruner
on SST-2 for BERT, ALBERT, RoBERTa, and XLM-RoBERTa, and plots the final 2x2 mask figure:
- figures/allfinalsolutions_sst2.pdf
- figures/allfinalsolutions_sst2.png
- figures/masks_sst2.pt (cached masks)

Usage:
    python generate_sst2_final_solutions.py [--prune-ratio 0.75] [--albert-prune-ratio 0.25] [--num-samples 1000] [--seed 555]
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import argparse
import random
import gc
import time
import torch
import torch.nn.functional as F
import torch.nn.modules.linear
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from textpruner import TransformerPruner


def clear_memory():
    """Explicitly garbage collects and empties CUDA / MPS device memory caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# Monkeypatch for linear layer replication
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


def sample_datapoints(dataset_column, num_samples=1000, seed=555):
    total = len(dataset_column)
    n = min(num_samples, total)
    rng = random.Random(seed)
    indices = rng.sample(range(total), n)
    return [dataset_column[i] for i in indices]


def set_seed(seed=555):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_new_head_mask_basedonscore(head_mask_previous, scores):
    """Finds the minimum score head among active heads (value=1) and sets it to 0."""
    head_mask = head_mask_previous.clone()
    num_layers = head_mask.shape[0]

    min_score = float("inf")
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


def get_new_head_mask_basedonscore_albert(head_mask_previous, scores):
    """Finds the minimum score head among ALBERT's 12 shared active heads and sets it to 0."""
    head_mask = head_mask_previous.clone()

    min_score = float("inf")
    min_pos = -1

    for head in range(12):
        if head_mask[0][head] == 1:
            if scores[head] < min_score:
                min_score = scores[head]
                min_pos = head

    if min_pos != -1:
        head_mask[0][min_pos] = 0

    return head_mask


# =========================================================================
# Gnorm Score Calculators (Per Step)
# =========================================================================

def get_gnorm_scores_bert(ds, model, tokenizer, pruned_heads, batch_size=128, device="cuda", num_samples=1000, seed=555):
    model.to(device)
    model.train()
    norms_Q = torch.zeros(12, 12).to(device)
    norms_K = torch.zeros(12, 12).to(device)
    norms_V = torch.zeros(12, 12).to(device)

    all_key1 = ds["train"]["sentence"]
    num_samples = min(num_samples, len(all_key1))
    all_key1 = sample_datapoints(all_key1, num_samples, seed=seed)

    for layer in range(12):
        attention_layer = model.bert.encoder.layer[layer].attention.self
        heads_Q_weight = attention_layer.query.weight
        heads_K_weight = attention_layer.key.weight
        heads_V_weight = attention_layer.value.weight

        for i in range(0, num_samples, batch_size):
            b_key1 = all_key1[i:i + batch_size]
            inputs = tokenizer(b_key1, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            loss = torch.norm(outputs.logits)
            model.zero_grad()
            loss.backward()

            if heads_Q_weight.grad is None or heads_Q_weight.size(1) == 0:
                model.zero_grad()
                continue

            GQ = expand_weights_to_768x768(heads_Q_weight.grad, pruned_heads[layer])
            GK = expand_weights_to_768x768(heads_K_weight.grad, pruned_heads[layer])
            GV = expand_weights_to_768x768(heads_V_weight.grad, pruned_heads[layer])

            tensor_Q = GQ.view(12, 64, 768)
            tensor_K = GK.view(12, 64, 768)
            tensor_V = GV.view(12, 64, 768)

            norms_Q[layer] += torch.norm(tensor_Q, p=2, dim=(1, 2))
            norms_K[layer] += torch.norm(tensor_K, p=2, dim=(1, 2))
            norms_V[layer] += torch.norm(tensor_V, p=2, dim=(1, 2))

        model.zero_grad()

    norms_Q = norms_Q.cpu() / num_samples
    norms_K = norms_K.cpu() / num_samples
    norms_V = norms_V.cpu() / num_samples

    norms = norms_Q * norms_K * norms_V

    # Set score of already pruned heads to infinity so they are not picked again
    for layer in range(12):
        mask = pruned_heads[layer].cpu() if isinstance(pruned_heads[layer], torch.Tensor) else torch.tensor(pruned_heads[layer]).cpu()
        for head in range(12):
            if mask[head] == 0:
                norms[layer][head] = float("inf")

    return norms


def get_gnorm_scores_albert(ds, model, tokenizer, pruned_heads, batch_size=64, device="cuda", num_samples=1000, seed=555):
    model.to(device)
    model.train()
    norms_Q = torch.zeros(1, 12, device=device)
    norms_K = torch.zeros(1, 12, device=device)
    norms_V = torch.zeros(1, 12, device=device)

    all_k1 = ds["train"]["sentence"]
    num_samples = min(num_samples, len(all_k1))
    all_k1 = sample_datapoints(all_k1, num_samples, seed=seed)

    attention_layer = model.albert.encoder.albert_layer_groups[0].albert_layers[0].attention
    heads_Q_weight = attention_layer.query.weight
    heads_K_weight = attention_layer.key.weight
    heads_V_weight = attention_layer.value.weight

    pl = pruned_heads[0]
    if isinstance(pl, torch.Tensor):
        pl = pl.cpu().tolist()

    for i in range(0, num_samples, batch_size):
        b1 = all_k1[i:i + batch_size]
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

    return scores[0]


def get_gnorm_scores_roberta(ds, model, tokenizer, pruned_heads, batch_size=128, device="cuda", num_samples=1000, seed=555):
    model.to(device)
    model.train()
    norms_Q = torch.zeros(12, 12).to(device)
    norms_K = torch.zeros(12, 12).to(device)
    norms_V = torch.zeros(12, 12).to(device)

    all_key1 = ds["train"]["sentence"]
    num_samples = min(num_samples, len(all_key1))
    all_key1 = sample_datapoints(all_key1, num_samples, seed=seed)

    for layer in range(12):
        attention_layer = model.roberta.encoder.layer[layer].attention.self
        heads_Q_weight = attention_layer.query.weight
        heads_K_weight = attention_layer.key.weight
        heads_V_weight = attention_layer.value.weight

        for i in range(0, num_samples, batch_size):
            b_key1 = all_key1[i:i + batch_size]
            inputs = tokenizer(b_key1, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            loss = torch.norm(outputs.logits)
            model.zero_grad()
            loss.backward()

            if heads_Q_weight.grad is None or heads_Q_weight.size(1) == 0:
                model.zero_grad()
                continue

            GQ = expand_weights_to_768x768(heads_Q_weight.grad, pruned_heads[layer])
            GK = expand_weights_to_768x768(heads_K_weight.grad, pruned_heads[layer])
            GV = expand_weights_to_768x768(heads_V_weight.grad, pruned_heads[layer])

            tensor_Q = GQ.view(12, 64, 768)
            tensor_K = GK.view(12, 64, 768)
            tensor_V = GV.view(12, 64, 768)

            norms_Q[layer] += torch.norm(tensor_Q, p=2, dim=(1, 2))
            norms_K[layer] += torch.norm(tensor_K, p=2, dim=(1, 2))
            norms_V[layer] += torch.norm(tensor_V, p=2, dim=(1, 2))

        model.zero_grad()

    norms_Q = norms_Q.cpu() / num_samples
    norms_K = norms_K.cpu() / num_samples
    norms_V = norms_V.cpu() / num_samples

    norms = norms_Q * norms_K * norms_V

    for layer in range(12):
        mask = pruned_heads[layer].cpu() if isinstance(pruned_heads[layer], torch.Tensor) else torch.tensor(pruned_heads[layer]).cpu()
        for head in range(12):
            if mask[head] == 0:
                norms[layer][head] = float("inf")

    return norms


def get_gnorm_scores_xlmroberta(ds, model, tokenizer, pruned_heads, batch_size=128, device="cuda", num_samples=1000, seed=555):
    model.to(device)
    model.train()
    norms_Q = torch.zeros(12, 12).to(device)
    norms_K = torch.zeros(12, 12).to(device)
    norms_V = torch.zeros(12, 12).to(device)

    all_key1 = ds["train"]["sentence"]
    num_samples = min(num_samples, len(all_key1))
    all_key1 = sample_datapoints(all_key1, num_samples, seed=seed)

    for layer in range(12):
        attention_layer = model.roberta.encoder.layer[layer].attention.self
        heads_Q_weight = attention_layer.query.weight
        heads_K_weight = attention_layer.key.weight
        heads_V_weight = attention_layer.value.weight

        for i in range(0, num_samples, batch_size):
            b_key1 = all_key1[i:i + batch_size]
            inputs = tokenizer(b_key1, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            loss = torch.norm(outputs.logits)
            model.zero_grad()
            loss.backward()

            if heads_Q_weight.grad is None or heads_Q_weight.size(1) == 0:
                model.zero_grad()
                continue

            GQ = expand_weights_to_768x768(heads_Q_weight.grad, pruned_heads[layer])
            GK = expand_weights_to_768x768(heads_K_weight.grad, pruned_heads[layer])
            GV = expand_weights_to_768x768(heads_V_weight.grad, pruned_heads[layer])

            tensor_Q = GQ.view(12, 64, 768)
            tensor_K = GK.view(12, 64, 768)
            tensor_V = GV.view(12, 64, 768)

            norms_Q[layer] += torch.norm(tensor_Q, p=2, dim=(1, 2))
            norms_K[layer] += torch.norm(tensor_K, p=2, dim=(1, 2))
            norms_V[layer] += torch.norm(tensor_V, p=2, dim=(1, 2))

        model.zero_grad()

    norms_Q = norms_Q.cpu() / num_samples
    norms_K = norms_K.cpu() / num_samples
    norms_V = norms_V.cpu() / num_samples

    norms = norms_Q * norms_K * norms_V

    for layer in range(12):
        mask = pruned_heads[layer].cpu() if isinstance(pruned_heads[layer], torch.Tensor) else torch.tensor(pruned_heads[layer]).cpu()
        for head in range(12):
            if mask[head] == 0:
                norms[layer][head] = float("inf")

    return norms


# =========================================================================
# Dynamic Iterative Pruning Loops (Algorithm 1)
# =========================================================================

def run_dynamic_pruning_bert(ds, model, tokenizer, target_pruned=108, batch_size=128, device="cuda", num_samples=1000, seed=555):
    head_mask = torch.ones(12, 12)
    print(f"Pruning BERT: target = {target_pruned}/144 heads...")
    for step in range(1, target_pruned + 1):
        scores = get_gnorm_scores_bert(ds, model, tokenizer, pruned_heads=head_mask, batch_size=batch_size, device=device, num_samples=num_samples, seed=seed)
        head_mask = get_new_head_mask_basedonscore(head_mask, scores)
        pruner = TransformerPruner(model)
        pruner.prune(head_mask=head_mask, save_model=False)
        del pruner, scores
        if step % 20 == 0 or step == target_pruned:
            print(f" -> Pruned {step:3d}/{target_pruned} heads (retained {int(head_mask.sum().item())}/144)")
            clear_memory()
    clear_memory()
    return head_mask.cpu().numpy()


def run_dynamic_pruning_albert(ds, model, tokenizer, target_pruned=3, batch_size=64, device="cuda", num_samples=1000, seed=555):
    head_mask = torch.ones(1, 12)
    print(f"Pruning ALBERT: target = {target_pruned}/12 shared heads...")
    for step in range(1, target_pruned + 1):
        scores = get_gnorm_scores_albert(ds, model, tokenizer, pruned_heads=head_mask, batch_size=batch_size, device=device, num_samples=num_samples, seed=seed)
        head_mask = get_new_head_mask_basedonscore_albert(head_mask, scores)
        pruner = TransformerPruner(model)
        pruner.prune(head_mask=head_mask, save_model=False)
        del pruner, scores
        print(f" -> Pruned {step:2d}/{target_pruned} heads (retained {int(head_mask.sum().item())}/12)")
    clear_memory()
    return np.tile(head_mask[0].cpu().numpy(), (12, 1))


def run_dynamic_pruning_roberta(ds, model, tokenizer, target_pruned=108, batch_size=128, device="cuda", num_samples=1000, seed=555):
    head_mask = torch.ones(12, 12)
    print(f"Pruning RoBERTa: target = {target_pruned}/144 heads...")
    for step in range(1, target_pruned + 1):
        scores = get_gnorm_scores_roberta(ds, model, tokenizer, pruned_heads=head_mask, batch_size=batch_size, device=device, num_samples=num_samples, seed=seed)
        head_mask = get_new_head_mask_basedonscore(head_mask, scores)
        pruner = TransformerPruner(model)
        pruner.prune(head_mask=head_mask, save_model=False)
        del pruner, scores
        if step % 20 == 0 or step == target_pruned:
            print(f" -> Pruned {step:3d}/{target_pruned} heads (retained {int(head_mask.sum().item())}/144)")
            clear_memory()
    clear_memory()
    return head_mask.cpu().numpy()


def run_dynamic_pruning_xlmroberta(ds, model, tokenizer, target_pruned=108, batch_size=128, device="cuda", num_samples=1000, seed=555):
    head_mask = torch.ones(12, 12)
    print(f"Pruning XLM-RoBERTa: target = {target_pruned}/144 heads...")
    for step in range(1, target_pruned + 1):
        scores = get_gnorm_scores_xlmroberta(ds, model, tokenizer, pruned_heads=head_mask, batch_size=batch_size, device=device, num_samples=num_samples, seed=seed)
        head_mask = get_new_head_mask_basedonscore(head_mask, scores)
        pruner = TransformerPruner(model)
        pruner.prune(head_mask=head_mask, save_model=False)
        del pruner, scores
        if step % 20 == 0 or step == target_pruned:
            print(f" -> Pruned {step:3d}/{target_pruned} heads (retained {int(head_mask.sum().item())}/144)")
            clear_memory()
    clear_memory()
    return head_mask.cpu().numpy()


# =========================================================================
# Main Execution Pipeline
# =========================================================================

def extract_or_load_dynamic_masks(prune_ratio=0.75, albert_prune_ratio=0.25, num_samples=1000, seed=555, device=None, force_recompute=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, "masks_sst2.pt")

    masks = {}
    if not force_recompute and os.path.exists(cache_path):
        try:
            try:
                cached_data = torch.load(cache_path, map_location="cpu", weights_only=False)
            except TypeError:
                cached_data = torch.load(cache_path, map_location="cpu")
            if isinstance(cached_data, dict) and "masks" in cached_data:
                masks = cached_data["masks"]
                print(f"Loaded existing checkpoint from {cache_path} with completed models: {list(masks.keys())}")
                if all(k in masks for k in ["BERT", "ALBERT", "RoBERTa", "XLM-RoBERTa"]):
                    print("All 4 model masks already completed. Returning cached masks.")
                    return masks
        except Exception as e:
            print(f"Warning: Could not read existing checkpoint ({e}). Starting fresh.")
            masks = {}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")

    set_seed(seed)
    print("Loading SST-2 dataset...")
    ds = load_dataset("glue", "sst2")

    # Target head count calculation
    target_144 = int(round(144 * prune_ratio))       # 108 heads for 75%
    target_albert = int(round(12 * albert_prune_ratio)) # 3 heads for 25%

    def save_checkpoint():
        torch.save({"masks": masks, "prune_ratio": prune_ratio, "albert_prune_ratio": albert_prune_ratio}, cache_path)
        print(f"Saved checkpoint to: {cache_path} (Completed so far: {list(masks.keys())})")

    # 1. BERT
    if "BERT" in masks and not force_recompute:
        print("\n[1/4] BERT already completed in cache. Skipping.")
    else:
        print(f"\n[1/4] Running dynamic Greedy-Gnorm pruning on BERT (target: {target_144}/144)...")
        tokenizer_bert = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
        model_bert = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2", output_attentions=True)
        masks["BERT"] = run_dynamic_pruning_bert(ds, model_bert, tokenizer_bert, target_pruned=target_144, batch_size=128, device=device, num_samples=num_samples, seed=seed)
        del model_bert, tokenizer_bert
        clear_memory()
        save_checkpoint()

    # 2. ALBERT
    if "ALBERT" in masks and not force_recompute:
        print("\n[2/4] ALBERT already completed in cache. Skipping.")
    else:
        print(f"\n[2/4] Running dynamic Greedy-Gnorm pruning on ALBERT (target: {target_albert}/12)...")
        tokenizer_albert = AutoTokenizer.from_pretrained("textattack/albert-base-v2-SST-2")
        model_albert = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-SST-2", output_attentions=True)
        masks["ALBERT"] = run_dynamic_pruning_albert(ds, model_albert, tokenizer_albert, target_pruned=target_albert, batch_size=64, device=device, num_samples=num_samples, seed=seed)
        del model_albert, tokenizer_albert
        clear_memory()
        save_checkpoint()

    # 3. RoBERTa
    if "RoBERTa" in masks and not force_recompute:
        print("\n[3/4] RoBERTa already completed in cache. Skipping.")
    else:
        print(f"\n[3/4] Running dynamic Greedy-Gnorm pruning on RoBERTa (target: {target_144}/144)...")
        tokenizer_roberta = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
        model_roberta = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2", output_attentions=True)
        masks["RoBERTa"] = run_dynamic_pruning_roberta(ds, model_roberta, tokenizer_roberta, target_pruned=target_144, batch_size=128, device=device, num_samples=num_samples, seed=seed)
        del model_roberta, tokenizer_roberta
        clear_memory()
        save_checkpoint()

    # 4. XLM-RoBERTa
    if "XLM-RoBERTa" in masks and not force_recompute:
        print("\n[4/4] XLM-RoBERTa already completed in cache. Skipping.")
    else:
        print(f"\n[4/4] Running dynamic Greedy-Gnorm pruning on XLM-RoBERTa (target: {target_144}/144)...")
        tokenizer_xlm = AutoTokenizer.from_pretrained("Ibrahim-Alam/finetuning-xlm-roberta-base-on-sst2")
        model_xlm = AutoModelForSequenceClassification.from_pretrained("Ibrahim-Alam/finetuning-xlm-roberta-base-on-sst2", output_attentions=True)
        masks["XLM-RoBERTa"] = run_dynamic_pruning_xlmroberta(ds, model_xlm, tokenizer_xlm, target_pruned=target_144, batch_size=128, device=device, num_samples=num_samples, seed=seed)
        del model_xlm, tokenizer_xlm
        clear_memory()
        save_checkpoint()

    return masks


def plot_all_final_solutions_sst2(masks, output_dir=None, dpi=300):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(output_dir, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(16, 14), dpi=100)
    fig.patch.set_alpha(0.0)

    panels = [
        ("BERT", masks["BERT"], 0, 0),
        ("ALBERT", masks["ALBERT"], 0, 1),
        ("RoBERTa", masks["RoBERTa"], 1, 0),
        ("XLM-RoBERTa", masks["XLM-RoBERTa"], 1, 1),
    ]

    for title, mask, r, c in panels:
        ax = axs[r, c]
        ax.imshow(mask, cmap="Blues", aspect="auto")
        ax.set_title(title, fontsize=20, pad=12, fontweight="bold")
        ax.set_xlabel("Heads", fontsize=16)
        ax.set_ylabel("Layers", fontsize=16)
        ax.set_xticks(range(12))
        ax.set_xticklabels([str(i) for i in range(1, 13)])
        ax.set_yticks(range(12))
        ax.set_yticklabels([str(i) for i in range(1, 13)])
        ax.tick_params(axis="both", labelsize=12)

    pruned_patch = mpatches.Patch(color=plt.cm.Blues(0.2), label="Pruned Head (0)")
    unpruned_patch = mpatches.Patch(color=plt.cm.Blues(1.0), label="Retained Head (1)")
    fig.legend(
        handles=[pruned_patch, unpruned_patch],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        prop={"size": 15},
        framealpha=0.9
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.92, 0.96])

    png_path = os.path.join(output_dir, "allfinalsolutions_sst2.png")
    pdf_path = os.path.join(output_dir, "allfinalsolutions_sst2.pdf")

    fig.savefig(png_path, dpi=dpi, transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(pdf_path, transparent=True)
    plt.close(fig)

    print("\n" + "=" * 60)
    print("SUCCESSFULLY GENERATED SST-2 FINAL PRUNING SOLUTIONS:")
    print(f" -> PDF: {pdf_path}")
    print(f" -> PNG: {png_path}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate SST-2 final pruning mask figure across 4 models using dynamic Greedy-Gnorm.")
    parser.add_argument("--prune-ratio", type=float, default=0.75, help="Pruning ratio for BERT, RoBERTa, XLM-RoBERTa (default: 0.75 = 75%% heads pruned / 108 heads)")
    parser.add_argument("--albert-prune-ratio", type=float, default=0.25, help="Pruning ratio for ALBERT (default: 0.25 = 25%% heads pruned / 3 heads)")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of training samples to estimate Gnorm per step (default: 1000)")
    parser.add_argument("--seed", type=int, default=555, help="Random seed (default: 555)")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory for figures (default: figures)")
    parser.add_argument("--force-recompute", action="store_true", help="Force re-computation of dynamic pruning instead of loading cache")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for exported PNG (default: 300)")
    args = parser.parse_args()

    masks = extract_or_load_dynamic_masks(
        prune_ratio=args.prune_ratio,
        albert_prune_ratio=args.albert_prune_ratio,
        num_samples=args.num_samples,
        seed=args.seed,
        force_recompute=args.force_recompute
    )
    plot_all_final_solutions_sst2(masks, output_dir=args.output_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
