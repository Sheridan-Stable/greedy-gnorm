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

parser = argparse.ArgumentParser(description="BERT GLUE Greedy Gnorm Variants Benchmark")
parser.add_argument("seed_pos", type=int, nargs="?", default=None, help="Random seed (positional argument)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for data sampling")
args, _ = parser.parse_known_args()

SEED = args.seed_pos if args.seed_pos is not None else args.seed
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

os.makedirs("experiments_results/glue", exist_ok=True)

# Monkeypatch for replication-safe linear layer if needed
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
# Task 1: SST2
# ==========================================
sst2_output_path = f"experiments_results/glue/BERT_sst2_ggnorm_variants_seed_{SEED}.csv"
if os.path.exists(sst2_output_path):
    print(f"\nSkipping Task: SST2 because output file already exists: {sst2_output_path}")
else:
    print("\n" + "="*50)
    print("Starting Task: SST2 (BERT Gnorm Variants)...")
    print("="*50 + "\n")

    print("Loading SST2...")
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2", output_attentions=True)
    ds = load_dataset("glue", "sst2")

    def get_acc_sst2(ds, model, tokenizer, size=len(ds['validation']), device='cuda', batch_size=128):
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        all_key1 = ds['validation']['sentence']
        all_labels = ds['validation']['label']
        
        with torch.no_grad():
            for i in range(0, size, batch_size):
                b_key1 = all_key1[i:i+batch_size]
                b_labels = all_labels[i:i+batch_size]
                inputs = tokenizer(b_key1, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == torch.tensor(b_labels).to(device)).sum().item()
                total += len(b_labels)
                
        return correct / total

    def get_gnorm_scores_sst2(ds, model, tokenizer, pruned_heads, variant='product', batch_size=128, device='cuda'):
        model.to(device)
        model.train()
        norms_Q = torch.zeros(12, 12).to(device)
        norms_K = torch.zeros(12, 12).to(device)
        norms_V = torch.zeros(12, 12).to(device)
        
        all_key1 = ds['train']['sentence']

        num_samples = min(1000, len(all_key1))
        all_key1 = sample_datapoints(all_key1, num_samples, seed=SEED)

        for layer in range(12):
            attention_layer = model.bert.encoder.layer[layer].attention.self
            heads_Q_weight = attention_layer.query.weight
            heads_K_weight = attention_layer.key.weight
            heads_V_weight = attention_layer.value.weight
            
            for i in range(0, num_samples, batch_size):
                b_key1 = all_key1[i:i+batch_size]
                inputs = tokenizer(b_key1, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                loss = torch.norm(outputs.logits)
                model.zero_grad()
                loss.backward()
                
                if heads_Q_weight.grad is None or heads_Q_weight.size(1) == 0:
                    model.zero_grad()
                    continue
                    
                GQ  = expand_weights_to_768x768(heads_Q_weight.grad, pruned_heads[layer])
                GK  = expand_weights_to_768x768(heads_K_weight.grad, pruned_heads[layer])
                GV  = expand_weights_to_768x768(heads_V_weight.grad, pruned_heads[layer])
                
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
        
        if variant == 'product':
            norms = norms_Q * norms_K * norms_V
        elif variant == 'sum':
            norms = norms_Q + norms_K + norms_V
        elif variant == 'max':
            norms = torch.maximum(torch.maximum(norms_Q, norms_K), norms_V)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        # Zero out already pruned heads
        for layer in range(12):
            if isinstance(pruned_heads[layer], torch.Tensor):
                mask = pruned_heads[layer].cpu()
            else:
                mask = torch.tensor(pruned_heads[layer]).cpu()
            for head in range(12):
                if mask[head] == 0:
                    norms[layer][head] = float('inf')
                    
        return norms

    def run_pruning_sst2(ds, model, tokenizer, variant='product'):
        head_mask = torch.ones(12, 12)
        accs = []
        initial_acc = get_acc_sst2(ds, model, tokenizer)
        accs.append(initial_acc)
        print(f"[SST2 Gnorm variant: {variant}] pruned heads: 0 acc: {initial_acc}")
        
        for step in range(1, 145):
            scores = get_gnorm_scores_sst2(ds, model, tokenizer, pruned_heads=head_mask, variant=variant)
            head_mask = get_new_head_mask_basedonscore(head_mask, scores)
                
            pruner = TransformerPruner(model)
            pruner.prune(head_mask=head_mask, save_model=False)
            
            acc = get_acc_sst2(ds, model, tokenizer)
            accs.append(acc)
            print(f"[SST2 Gnorm variant: {variant}] pruned heads: {step} acc: {acc}")
            
        return accs

    # Run Greedy Gnorm (Product)
    print("Running SST2 Greedy Gnorm (Product)...")
    accs_product = run_pruning_sst2(ds, model, tokenizer, variant='product')

    # Reload model for Sum
    del model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2", output_attentions=True)
    print("Running SST2 Greedy Gnorm (Sum)...")
    accs_sum = run_pruning_sst2(ds, model, tokenizer, variant='sum')

    # Reload model for Max
    del model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2", output_attentions=True)
    print("Running SST2 Greedy Gnorm (Max)...")
    accs_max = run_pruning_sst2(ds, model, tokenizer, variant='max')

    # Output CSV
    heads_pruned = list(range(145))
    df_acc = pd.DataFrame({
        'Heads Pruned': heads_pruned,
        'Accuracy_Product': accs_product,
        'Accuracy_Sum': accs_sum,
        'Accuracy_Max': accs_max
    })
    df_acc.to_csv(sst2_output_path, index=False)
    print(f"Saved SST2 accuracy variants benchmark to {sst2_output_path}")
    print(df_acc)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


# ==========================================
# Task 2: MNLI
# ==========================================
mnli_output_path = f"experiments_results/glue/BERT_mnli_ggnorm_variants_seed_{SEED}.csv"
if os.path.exists(mnli_output_path):
    print(f"\nSkipping Task: MNLI because output file already exists: {mnli_output_path}")
else:
    print("\n" + "="*50)
    print("Starting Task: MNLI (BERT Gnorm Variants)...")
    print("="*50 + "\n")

    print("Loading MNLI...")
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI")
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI", output_attentions=True)
    ds = load_dataset("glue", "mnli")

    def get_acc_mnli(ds, model, tokenizer, size=len(ds['validation_matched']), device='cuda', batch_size=128):
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
                
                label_map = torch.tensor([2, 0, 1], device=device)
                preds = label_map[preds]
                
                correct += (preds == torch.tensor(b_labels).to(device)).sum().item()
                total += len(b_labels)
                
        return correct / total

    def get_gnorm_scores_mnli(ds, model, tokenizer, pruned_heads, variant='product', batch_size=128, device='cuda'):
        model.to(device)
        model.train()
        norms_Q = torch.zeros(12, 12).to(device)
        norms_K = torch.zeros(12, 12).to(device)
        norms_V = torch.zeros(12, 12).to(device)
        
        all_key1 = ds['train']['premise']
        all_key2 = ds['train']['hypothesis']

        num_samples = min(1000, len(all_key1))
        all_key1, all_key2 = sample_datapoints_pair(all_key1, all_key2, num_samples, seed=SEED)

        for layer in range(12):
            attention_layer = model.bert.encoder.layer[layer].attention.self
            heads_Q_weight = attention_layer.query.weight
            heads_K_weight = attention_layer.key.weight
            heads_V_weight = attention_layer.value.weight
            
            for i in range(0, num_samples, batch_size):
                b_key1 = all_key1[i:i+batch_size]
                b_key2 = all_key2[i:i+batch_size]
                inputs = tokenizer(b_key1, b_key2, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                loss = torch.norm(outputs.logits)
                model.zero_grad()
                loss.backward()
                
                if heads_Q_weight.grad is None or heads_Q_weight.size(1) == 0:
                    model.zero_grad()
                    continue
                    
                GQ  = expand_weights_to_768x768(heads_Q_weight.grad, pruned_heads[layer])
                GK  = expand_weights_to_768x768(heads_K_weight.grad, pruned_heads[layer])
                GV  = expand_weights_to_768x768(heads_V_weight.grad, pruned_heads[layer])
                
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
        
        if variant == 'product':
            norms = norms_Q * norms_K * norms_V
        elif variant == 'sum':
            norms = norms_Q + norms_K + norms_V
        elif variant == 'max':
            norms = torch.maximum(torch.maximum(norms_Q, norms_K), norms_V)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        # Zero out already pruned heads
        for layer in range(12):
            if isinstance(pruned_heads[layer], torch.Tensor):
                mask = pruned_heads[layer].cpu()
            else:
                mask = torch.tensor(pruned_heads[layer]).cpu()
            for head in range(12):
                if mask[head] == 0:
                    norms[layer][head] = float('inf')
                    
        return norms

    def run_pruning_mnli(ds, model, tokenizer, variant='product'):
        head_mask = torch.ones(12, 12)
        accs = []
        initial_acc = get_acc_mnli(ds, model, tokenizer)
        accs.append(initial_acc)
        print(f"[MNLI Gnorm variant: {variant}] pruned heads: 0 acc: {initial_acc}")
        
        for step in range(1, 145):
            scores = get_gnorm_scores_mnli(ds, model, tokenizer, pruned_heads=head_mask, variant=variant)
            head_mask = get_new_head_mask_basedonscore(head_mask, scores)
                
            pruner = TransformerPruner(model)
            pruner.prune(head_mask=head_mask, save_model=False)
            
            acc = get_acc_mnli(ds, model, tokenizer)
            accs.append(acc)
            print(f"[MNLI Gnorm variant: {variant}] pruned heads: {step} acc: {acc}")
            
        return accs

    # Run Greedy Gnorm (Product)
    print("Running MNLI Greedy Gnorm (Product)...")
    accs_product = run_pruning_mnli(ds, model, tokenizer, variant='product')

    # Reload model for Sum
    del model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI", output_attentions=True)
    print("Running MNLI Greedy Gnorm (Sum)...")
    accs_sum = run_pruning_mnli(ds, model, tokenizer, variant='sum')

    # Reload model for Max
    del model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI", output_attentions=True)
    print("Running MNLI Greedy Gnorm (Max)...")
    accs_max = run_pruning_mnli(ds, model, tokenizer, variant='max')

    # Output CSV
    heads_pruned = list(range(145))
    df_acc = pd.DataFrame({
        'Heads Pruned': heads_pruned,
        'Accuracy_Product': accs_product,
        'Accuracy_Sum': accs_sum,
        'Accuracy_Max': accs_max
    })
    df_acc.to_csv(mnli_output_path, index=False)
    print(f"Saved MNLI accuracy variants benchmark to {mnli_output_path}")
    print(df_acc)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
