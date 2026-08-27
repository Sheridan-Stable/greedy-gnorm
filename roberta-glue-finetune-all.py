import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from textpruner import TransformerPruner
import pandas as pd
import gc
import torch.nn.functional as F
import torch.nn.modules.linear
import argparse
import random

parser = argparse.ArgumentParser(description="RoBERTa GLUE Pruning + Post-Pruning Fine-Tuning (all tasks)")
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--prune_steps", type=int, default=108)
parser.add_argument("--finetune_epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--finetune_batch_size", type=int, default=32)
args = parser.parse_args()

SEED = args.seed
os.makedirs("experiments_results/finetune", exist_ok=True)

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

def sample_datapoints(col, num=1000):
    return random.Random(SEED).sample(list(col), min(num, len(col)))

def sample_datapoints_pair(col1, col2, num=1000):
    idx = random.Random(SEED).sample(range(len(col1)), min(num, len(col1)))
    return [col1[i] for i in idx], [col2[i] for i in idx]

def expand_weights_to_768x768(weight_matrix, active_heads_mask):
    head_size = 64
    device = weight_matrix.device
    tensors = []
    active_idx = 0
    for i in range(12):
        if active_heads_mask[i] == 1:
            s = active_idx * head_size
            tensors.append(weight_matrix[s:s+head_size, :])
            active_idx += 1
        else:
            tensors.append(torch.zeros((head_size, 768), device=device))
    return torch.cat(tensors, dim=0)

def get_new_head_mask(head_mask, scores):
    mask = head_mask.clone()
    min_score, min_pos = float('inf'), (-1, -1)
    for layer in range(mask.shape[0]):
        for head in range(12):
            if mask[layer][head] == 1 and scores[layer][head] < min_score:
                min_score = scores[layer][head]
                min_pos = (layer, head)
    if min_pos != (-1, -1):
        mask[min_pos[0]][min_pos[1]] = 0
    return mask

def _gnorm(model_obj, nQ, nK, nV, pruned_heads, layer, w_triples):
    for nrm, w in w_triples:
        g = expand_weights_to_768x768(w.grad, pruned_heads[layer])
        nrm[layer] += g.view(12, 64, 768).norm(p=2, dim=(1, 2))

def finalize_norms(nQ, nK, nV, pruned_heads, num):
    norms = (nQ * nK * nV).cpu() / num
    for layer in range(12):
        mask = pruned_heads[layer] if isinstance(pruned_heads[layer], torch.Tensor) else torch.tensor(pruned_heads[layer])
        for h in range(12):
            if mask[h] == 0:
                norms[layer][h] = float('inf')
    return norms


# ─────────────────────────────────────────────────────────────
# SST-2
# ─────────────────────────────────────────────────────────────
def make_sst2_fns(ds, tokenizer, model, device='cuda'):
    def get_acc():
        model.to(device); model.eval()
        correct = total = 0
        sents  = ds['validation']['sentence']
        labels = ds['validation']['label']
        with torch.no_grad():
            for i in range(0, len(sents), 128):
                inp = tokenizer(sents[i:i+128], return_tensors='pt', padding=True, truncation=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                preds = torch.argmax(model(**inp).logits, dim=-1)
                correct += (preds == torch.tensor(labels[i:i+128]).to(device)).sum().item()
                total += len(labels[i:i+128])
        return correct / total

    def get_gnorm(pruned_heads):
        model.to(device); model.train()
        nQ = torch.zeros(12, 12, device=device)
        nK = torch.zeros(12, 12, device=device)
        nV = torch.zeros(12, 12, device=device)
        sents = sample_datapoints(ds['train']['sentence'])
        num = len(sents)
        for layer in range(12):
            attn = model.roberta.encoder.layer[layer].attention.self
            wQ, wK, wV = attn.query.weight, attn.key.weight, attn.value.weight
            for i in range(0, num, 128):
                inp = tokenizer(sents[i:i+128], return_tensors='pt', padding=True, truncation=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                model(**inp).logits.norm().backward()
                if wQ.grad is None or wQ.size(1) == 0:
                    model.zero_grad(); continue
                for nrm, w in [(nQ, wQ), (nK, wK), (nV, wV)]:
                    nrm[layer] += expand_weights_to_768x768(w.grad, pruned_heads[layer]).view(12, 64, 768).norm(p=2, dim=(1, 2))
                model.zero_grad()
        return finalize_norms(nQ, nK, nV, pruned_heads, num)

    def finetune():
        model.to(device)
        sents  = list(ds['train']['sentence'])
        labels = list(ds['train']['label'])
        total_steps = (len(sents) // args.finetune_batch_size + 1) * args.finetune_epochs
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)
        rng = random.Random(SEED)
        for epoch in range(args.finetune_epochs):
            model.train()
            idx = list(range(len(sents))); rng.shuffle(idx)
            es, el = [sents[i] for i in idx], [labels[i] for i in idx]
            for i in range(0, len(es), args.finetune_batch_size):
                inp = tokenizer(es[i:i+args.finetune_batch_size], return_tensors='pt', padding=True, truncation=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                inp['labels'] = torch.tensor(el[i:i+args.finetune_batch_size]).to(device)
                opt.zero_grad(); model(**inp).loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step()
            print(f"  [FT epoch {epoch+1}] val acc: {get_acc()*100:.2f}%")

    return get_acc, get_gnorm, finetune


# ─────────────────────────────────────────────────────────────
# MNLI
# ─────────────────────────────────────────────────────────────
def make_mnli_fns(ds, tokenizer, model, device='cuda'):
    def get_acc():
        model.to(device); model.eval()
        correct = total = 0
        p = ds['validation_matched']['premise']
        h = ds['validation_matched']['hypothesis']
        l = ds['validation_matched']['label']
        label_map = torch.tensor([2, 0, 1], device=device)
        with torch.no_grad():
            for i in range(0, len(p), 128):
                inp = tokenizer(p[i:i+128], h[i:i+128], return_tensors='pt', padding=True, truncation=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                preds = label_map[torch.argmax(model(**inp).logits, dim=-1)]
                correct += (preds == torch.tensor(l[i:i+128]).to(device)).sum().item()
                total += len(l[i:i+128])
        return correct / total

    def get_gnorm(pruned_heads):
        model.to(device); model.train()
        nQ = torch.zeros(12, 12, device=device)
        nK = torch.zeros(12, 12, device=device)
        nV = torch.zeros(12, 12, device=device)
        s1, s2 = sample_datapoints_pair(ds['train']['premise'], ds['train']['hypothesis'])
        num = len(s1)
        for layer in range(12):
            attn = model.roberta.encoder.layer[layer].attention.self
            wQ, wK, wV = attn.query.weight, attn.key.weight, attn.value.weight
            for i in range(0, num, 128):
                inp = tokenizer(s1[i:i+128], s2[i:i+128], return_tensors='pt', padding=True, truncation=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                model(**inp).logits.norm().backward()
                if wQ.grad is None or wQ.size(1) == 0:
                    model.zero_grad(); continue
                for nrm, w in [(nQ, wQ), (nK, wK), (nV, wV)]:
                    nrm[layer] += expand_weights_to_768x768(w.grad, pruned_heads[layer]).view(12, 64, 768).norm(p=2, dim=(1, 2))
                model.zero_grad()
        return finalize_norms(nQ, nK, nV, pruned_heads, num)

    def finetune():
        model.to(device)
        p = list(ds['train']['premise'])
        h = list(ds['train']['hypothesis'])
        l = list(ds['train']['label'])
        total_steps = (len(p) // args.finetune_batch_size + 1) * args.finetune_epochs
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)
        rng = random.Random(SEED)
        for epoch in range(args.finetune_epochs):
            model.train()
            idx = list(range(len(p))); rng.shuffle(idx)
            ep, eh, el = [p[i] for i in idx], [h[i] for i in idx], [l[i] for i in idx]
            for i in range(0, len(ep), args.finetune_batch_size):
                inp = tokenizer(ep[i:i+args.finetune_batch_size], eh[i:i+args.finetune_batch_size],
                                return_tensors='pt', padding=True, truncation=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                inp['labels'] = torch.tensor(el[i:i+args.finetune_batch_size]).to(device)
                opt.zero_grad(); model(**inp).loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step()
            print(f"  [FT epoch {epoch+1}] val acc: {get_acc()*100:.2f}%")

    return get_acc, get_gnorm, finetune


# ─────────────────────────────────────────────────────────────
# QNLI
# ─────────────────────────────────────────────────────────────
def make_qnli_fns(ds, tokenizer, model, device='cuda'):
    def get_acc():
        model.to(device); model.eval()
        correct = total = 0
        q = ds['validation']['question']
        s = ds['validation']['sentence']
        l = ds['validation']['label']
        with torch.no_grad():
            for i in range(0, len(q), 128):
                inp = tokenizer(q[i:i+128], s[i:i+128], return_tensors='pt', padding=True, truncation=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                preds = torch.argmax(model(**inp).logits, dim=-1)
                correct += (preds == torch.tensor(l[i:i+128]).to(device)).sum().item()
                total += len(l[i:i+128])
        return correct / total

    def get_gnorm(pruned_heads):
        model.to(device); model.train()
        nQ = torch.zeros(12, 12, device=device)
        nK = torch.zeros(12, 12, device=device)
        nV = torch.zeros(12, 12, device=device)
        s1, s2 = sample_datapoints_pair(ds['train']['question'], ds['train']['sentence'])
        num = len(s1)
        for layer in range(12):
            attn = model.roberta.encoder.layer[layer].attention.self
            wQ, wK, wV = attn.query.weight, attn.key.weight, attn.value.weight
            for i in range(0, num, 128):
                inp = tokenizer(s1[i:i+128], s2[i:i+128], return_tensors='pt', padding=True, truncation=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                model(**inp).logits.norm().backward()
                if wQ.grad is None or wQ.size(1) == 0:
                    model.zero_grad(); continue
                for nrm, w in [(nQ, wQ), (nK, wK), (nV, wV)]:
                    nrm[layer] += expand_weights_to_768x768(w.grad, pruned_heads[layer]).view(12, 64, 768).norm(p=2, dim=(1, 2))
                model.zero_grad()
        return finalize_norms(nQ, nK, nV, pruned_heads, num)

    def finetune():
        model.to(device)
        q = list(ds['train']['question'])
        s = list(ds['train']['sentence'])
        l = list(ds['train']['label'])
        total_steps = (len(q) // args.finetune_batch_size + 1) * args.finetune_epochs
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)
        rng = random.Random(SEED)
        for epoch in range(args.finetune_epochs):
            model.train()
            idx = list(range(len(q))); rng.shuffle(idx)
            eq, es, el = [q[i] for i in idx], [s[i] for i in idx], [l[i] for i in idx]
            for i in range(0, len(eq), args.finetune_batch_size):
                inp = tokenizer(eq[i:i+args.finetune_batch_size], es[i:i+args.finetune_batch_size],
                                return_tensors='pt', padding=True, truncation=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                inp['labels'] = torch.tensor(el[i:i+args.finetune_batch_size]).to(device)
                opt.zero_grad(); model(**inp).loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step()
            print(f"  [FT epoch {epoch+1}] val acc: {get_acc()*100:.2f}%")

    return get_acc, get_gnorm, finetune


def run_task(task_name, get_acc, get_gnorm, finetune, model_tag):
    summary_path = f"experiments_results/finetune/{model_tag}_{task_name}_gnorm_finetune_summary_seed_{SEED}_{args.prune_steps}.csv"
    if os.path.exists(summary_path):
        print(f"\n[SKIP] {model_tag} {task_name} seed={SEED} prune_steps={args.prune_steps} already done ({summary_path})")
        return
    print(f"\n{'='*60}\nTask: {task_name}  |  prune_steps={args.prune_steps}\n{'='*60}")
    head_mask = torch.ones(12, 12)
    accs = [get_acc()]
    print(f"[step 0] acc: {accs[0]:.4f}")
    for step in range(1, args.prune_steps + 1):
        scores    = get_gnorm(head_mask)
        head_mask = get_new_head_mask(head_mask, scores)
        pruner    = TransformerPruner(model)
        pruner.prune(head_mask=head_mask, save_model=False)
        acc = get_acc()
        accs.append(acc)
        print(f"[step {step}] acc: {acc:.4f}")
    acc_before = accs[-1]
    print(f"\nBefore FT: {acc_before*100:.2f}%")
    finetune()
    acc_after = get_acc()
    print(f"After  FT: {acc_after*100:.2f}%  (recovery +{(acc_after-acc_before)*100:.2f}pp)")
    df = pd.DataFrame({'Heads_Pruned': list(range(args.prune_steps + 1)), 'Accuracy_Gnorm': accs})
    df.to_csv(f"experiments_results/finetune/{model_tag}_{task_name}_gnorm_finetune_seed_{SEED}_{args.prune_steps}.csv", index=False)
    df_s = pd.DataFrame({
        'Seed': [SEED], 'Task': [task_name], 'Prune_Steps': [args.prune_steps],
        'Finetune_Epochs': [args.finetune_epochs], 'LR': [args.lr],
        'Acc_Before_FT': [acc_before], 'Acc_After_FT': [acc_after],
        'Recovery_pp': [acc_after - acc_before],
    })
    df_s.to_csv(f"experiments_results/finetune/{model_tag}_{task_name}_gnorm_finetune_summary_seed_{SEED}_{args.prune_steps}.csv", index=False)
    print(df_s.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# SST-2
# ─────────────────────────────────────────────────────────────
print("Loading RoBERTa-SST-2...")
tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
model     = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2", output_attentions=True)
ds        = load_dataset("glue", "sst2")
get_acc, get_gnorm, finetune = make_sst2_fns(ds, tokenizer, model)
run_task("sst2", get_acc, get_gnorm, finetune, "ROBERTA")
del model, tokenizer, ds; gc.collect()

# ─────────────────────────────────────────────────────────────
# MNLI
# ─────────────────────────────────────────────────────────────
print("Loading RoBERTa-MNLI...")
tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-MNLI")
model     = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-MNLI", output_attentions=True)
ds        = load_dataset("glue", "mnli")
get_acc, get_gnorm, finetune = make_mnli_fns(ds, tokenizer, model)
run_task("mnli", get_acc, get_gnorm, finetune, "ROBERTA")
del model, tokenizer, ds; gc.collect()

# ─────────────────────────────────────────────────────────────
# QNLI
# ─────────────────────────────────────────────────────────────
print("Loading RoBERTa-QNLI...")
tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-QNLI")
model     = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-QNLI", output_attentions=True)
ds        = load_dataset("glue", "qnli")
get_acc, get_gnorm, finetune = make_qnli_fns(ds, tokenizer, model)
run_task("qnli", get_acc, get_gnorm, finetune, "ROBERTA")
del model, tokenizer, ds; gc.collect()

print("\nAll RoBERTa tasks complete.")
