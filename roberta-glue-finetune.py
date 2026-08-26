import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
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


parser = argparse.ArgumentParser(description="RoBERTa GLUE Pruning + Post-Pruning Fine-Tuning")
parser.add_argument("--seed", type=int, default=555, help="Random seed for data sampling")
parser.add_argument("--prune_steps", type=int, default=108,
                    help="Number of heads to prune before fine-tuning (default: 108 = 75%% of 144)")
parser.add_argument("--finetune_epochs", type=int, default=3, help="Number of fine-tuning epochs after pruning")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for fine-tuning")
parser.add_argument("--finetune_batch_size", type=int, default=32, help="Batch size for fine-tuning")
args = parser.parse_args()

SEED = args.seed
print(f"Using random seed: {SEED}")
print(f"Pruning {args.prune_steps} heads, then fine-tuning for {args.finetune_epochs} epochs at lr={args.lr}")

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

os.makedirs("experiments_results/finetune", exist_ok=True)

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

print("\n" + "="*50)
print("Completed setup and helper functions. Starting Task: SST2 (RoBERTa)...")
print("="*50 + "\n")

# ==========================================
# Task: SST2
# ==========================================
print("Loading SST2...")
tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2", output_attentions=True)
ds = load_dataset("glue", "sst2")

def get_acc(ds, model, tokenizer, size=len(ds['validation']), device='cuda', batch_size=128):
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

def get_gnorm_scores(ds, model, tokenizer, pruned_heads, batch_size=128, device='cuda'):
    model.to(device)
    model.train()
    norms_Q = torch.zeros(12, 12).to(device)
    norms_K = torch.zeros(12, 12).to(device)
    norms_V = torch.zeros(12, 12).to(device)

    all_key1 = ds['train']['sentence']

    # Sample 1000 for fast estimation
    num_samples = min(1000, len(all_key1))
    all_key1 = sample_datapoints(all_key1, num_samples, seed=SEED)

    for layer in range(12):
        attention_layer = model.roberta.encoder.layer[layer].attention.self
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

def finetune(ds, model, tokenizer, epochs=3, lr=2e-5, batch_size=32, device='cuda'):
    """Fine-tune the pruned model on the training split for a fixed number of epochs."""
    model.to(device)

    all_sentences = ds['train']['sentence']
    all_labels    = ds['train']['label']
    total_steps = (len(all_sentences) // batch_size + 1) * epochs

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    rng = random.Random(SEED)

    for epoch in range(epochs):
        model.train()

        # Shuffle training data at the start of every epoch
        indices = list(range(len(all_sentences)))
        rng.shuffle(indices)
        epoch_sentences = [all_sentences[i] for i in indices]
        epoch_labels    = [all_labels[i]    for i in indices]

        total_loss = 0
        num_batches = 0
        for i in range(0, len(epoch_sentences), batch_size):
            b_sentences = epoch_sentences[i:i+batch_size]
            b_labels    = torch.tensor(epoch_labels[i:i+batch_size]).to(device)

            inputs = tokenizer(b_sentences, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs['labels'] = b_labels

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        val_acc  = get_acc(ds, model, tokenizer, device=device)
        print(f"  [finetune] Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.4f} | val acc: {val_acc*100:.2f}%")

def run_pruning_then_finetune(ds, model, tokenizer, prune_steps=108, finetune_epochs=3, lr=2e-5, finetune_batch_size=32):
    head_mask = torch.ones(12, 12)
    accs = []

    initial_acc = get_acc(ds, model, tokenizer)
    accs.append(initial_acc)
    print(f"[gnorm] pruned heads: 0 acc: {initial_acc:.4f}")

    # ---- Greedy-Gnorm pruning phase ----
    for step in range(1, prune_steps + 1):
        scores = get_gnorm_scores(ds, model, tokenizer, pruned_heads=head_mask)
        head_mask = get_new_head_mask_basedonscore(head_mask, scores)

        pruner = TransformerPruner(model)
        pruner.prune(head_mask=head_mask, save_model=False)

        acc = get_acc(ds, model, tokenizer)
        accs.append(acc)
        print(f"[gnorm] pruned heads: {step} acc: {acc:.4f}")

    acc_before_ft = accs[-1]
    print(f"\nAccuracy after pruning {prune_steps} heads (before fine-tuning): {acc_before_ft*100:.2f}%")

    # ---- Post-pruning fine-tuning phase ----
    print(f"\nFine-tuning pruned model for {finetune_epochs} epochs...")
    finetune(ds, model, tokenizer, epochs=finetune_epochs, lr=lr, batch_size=finetune_batch_size)

    acc_after_ft = get_acc(ds, model, tokenizer)
    print(f"\nAccuracy after fine-tuning: {acc_after_ft*100:.2f}%")
    print(f"Recovery gain:              +{(acc_after_ft - acc_before_ft)*100:.2f}pp")
    print(f"Remaining gap vs unpruned:  -{(initial_acc - acc_after_ft)*100:.2f}pp")

    return accs, acc_before_ft, acc_after_ft

# ---- Run ----
print("Running Greedy-Gnorm + Post-Pruning Fine-Tuning on RoBERTa (SST-2)...")
accs, acc_before_ft, acc_after_ft = run_pruning_then_finetune(
    ds, model, tokenizer,
    prune_steps=args.prune_steps,
    finetune_epochs=args.finetune_epochs,
    lr=args.lr,
    finetune_batch_size=args.finetune_batch_size
)

# ---- Save results ----
heads_pruned = list(range(args.prune_steps + 1))
df = pd.DataFrame({'Heads Pruned': heads_pruned, 'Accuracy_Gnorm': accs})
df.to_csv(f"experiments_results/finetune/ROBERTA_sst2_gnorm_finetune_seed_{SEED}_{args.prune_steps}.csv", index=False)

df_summary = pd.DataFrame({
    'Seed':            [SEED],
    'Prune_Steps':     [args.prune_steps],
    'Finetune_Epochs': [args.finetune_epochs],
    'LR':              [args.lr],
    'Acc_Before_FT':   [acc_before_ft],
    'Acc_After_FT':    [acc_after_ft],
    'Recovery_pp':     [acc_after_ft - acc_before_ft],
})
df_summary.to_csv(f"experiments_results/finetune/ROBERTA_sst2_gnorm_finetune_summary_seed_{SEED}_{args.prune_steps}.csv", index=False)
print(df_summary)
print("Done.")
