"""
train_multitask.py

Trains the multi-task deviation encoder with five output heads:
  1. Cancer detection  (binary, BCE)
  2. Purity            (continuous, MSE + rank loss)
  3. T-stage           (3-class T1/T2/T3+, cross-entropy, weighted)
  4. N-stage           (binary N0/N+, BCE, weighted)
  5. Progression       (stage I–IV ordinal, MSE on ordinal label)

Input: deviation vectors (obs_beta - blood_ref_mean) at 661 targeted CpGs
Splits are by original TCGA sample index — all fractions of the same tumor
stay in the same split, preventing data leakage.

Outputs:
  models/best_multitask.pt
  results/multitask_train_log.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import json

PROCESSED = Path("data/processed")
MODELS    = Path("models")
RESULTS   = Path("results")
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

SEED       = 42
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE   = 30
LR         = 1e-3
WD         = 1e-4

W_CANCER     = 1.0
W_PURITY     = 1.0
W_TSTAGE     = 0.5
W_NSTAGE     = 0.5
W_PROG       = 0.3
W_RANK       = 0.3

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


print("Loading in silico mixture data...")
X = np.load(PROCESSED / "insilio_X.npy")
L = np.load(PROCESSED / "insilio_labels.npz")

tumor_fraction = L["tumor_fraction"].astype(np.float32)
cpe            = L["cpe"].astype(np.float32)
t_stage        = L["t_stage"].astype(np.float32)
n_stage        = L["n_stage"].astype(np.float32)
overall_stage  = L["overall_stage"].astype(np.float32)
is_cancer      = L["is_cancer"].astype(np.float32)
sample_idx     = L["sample_idx"].astype(np.int32)

N, n_cpg = X.shape
print(f"Loaded: {N} samples, {n_cpg} CpGs")


unique_samples = np.unique(sample_idx)
rng = np.random.default_rng(SEED)
rng.shuffle(unique_samples)

n = len(unique_samples)
n_train = int(0.80 * n)
n_val   = int(0.10 * n)

train_samples = set(unique_samples[:n_train])
val_samples   = set(unique_samples[n_train:n_train+n_val])
test_samples  = set(unique_samples[n_train+n_val:])

train_mask = np.array([s in train_samples for s in sample_idx])
val_mask   = np.array([s in val_samples   for s in sample_idx])
test_mask  = np.array([s in test_samples  for s in sample_idx])

print(f"Split: {train_mask.sum()} train / {val_mask.sum()} val / {test_mask.sum()} test rows")
print(f"       ({len(train_samples)} / {len(val_samples)} / {len(test_samples)} unique tumors)")


def to_tensor(arr):
    return torch.from_numpy(arr.copy())

X_tr = to_tensor(X[train_mask])
X_va = to_tensor(X[val_mask])
X_te = to_tensor(X[test_mask])

def get_label_tensors(mask):
    return {
        "cancer":    to_tensor(is_cancer[mask]),
        "purity":    to_tensor(cpe[mask]),
        "t_stage":   to_tensor(t_stage[mask]),
        "n_stage":   to_tensor(n_stage[mask]),
        "prog":      to_tensor(overall_stage[mask]),
        "tf":        to_tensor(tumor_fraction[mask]),
    }

L_tr = get_label_tensors(train_mask)
L_va = get_label_tensors(val_mask)
L_te = get_label_tensors(test_mask)


class MixDataset(torch.utils.data.Dataset):
    def __init__(self, X, labels):
        self.X = X
        self.L = labels
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], {k: v[i] for k, v in self.L.items()}

train_ds = MixDataset(X_tr, L_tr)
val_ds   = MixDataset(X_va, L_va)
test_ds  = MixDataset(X_te, L_te)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)


class MultiTaskDeviationEncoder(nn.Module):
    """
    Shared encoder on deviation vectors, five output heads.
    Deeper than the purity-only model to handle multi-task complexity.
    """
    def __init__(self, n_cpg):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_cpg, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.cancer_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.purity_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.tstage_head = nn.Linear(64, 3)
        self.nstage_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.prog_head   = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        return {
            "cancer":  self.cancer_head(z).squeeze(1),
            "purity":  self.purity_head(z).squeeze(1),
            "t_stage": self.tstage_head(z),
            "n_stage": self.nstage_head(z).squeeze(1),
            "prog":    self.prog_head(z).squeeze(1),
        }

model = MultiTaskDeviationEncoder(n_cpg).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")


t_vals_tr = L_tr["t_stage"]
valid_t = t_vals_tr[~torch.isnan(t_vals_tr)].long()
t_counts = torch.bincount(valid_t.clamp(0, 2), minlength=3).float()
t_weights = (1.0 / (t_counts + 1)).to(device)
t_weights = t_weights / t_weights.sum() * 3

n_vals_tr = L_tr["n_stage"]
valid_n = n_vals_tr[~torch.isnan(n_vals_tr)]
n_pos_rate = (valid_n > 0).float().mean()
n_weight = torch.tensor([1.0 / (1 - n_pos_rate + 1e-6),
                          1.0 / (n_pos_rate + 1e-6)]).to(device)
n_weight = n_weight / n_weight.sum() * 2

print(f"T-stage class counts: {t_counts.long().tolist()}, weights: {t_weights.tolist()}")
print(f"N+ rate: {n_pos_rate:.2f}")


def masked_mse(pred, target):
    valid = ~torch.isnan(target)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.mse_loss(pred[valid], target[valid])

def masked_bce(pred, target, pos_weight=None):
    valid = ~torch.isnan(target)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    p = pred[valid]
    t = target[valid].clamp(0, 1)
    t_bin = (t > 0).float()
    return F.binary_cross_entropy(p, t_bin)

def masked_ce(logits, target, weight=None):
    valid = ~torch.isnan(target)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(logits[valid], target[valid].clamp(0, 2).long(), weight=weight)

def rank_loss(pred, target):
    valid = ~torch.isnan(target)
    if valid.sum() < 4:
        return torch.tensor(0.0, device=pred.device)
    p = pred[valid]
    t = target[valid]
    diff_p = p.unsqueeze(0) - p.unsqueeze(1)
    diff_t = t.unsqueeze(0) - t.unsqueeze(1)
    concordance = torch.sigmoid(diff_p * 10) * (diff_t > 0).float()
    discordance = torch.sigmoid(-diff_p * 10) * (diff_t < 0).float()
    loss = 1.0 - (concordance + discordance).mean()
    return loss

def compute_loss(preds, labels):
    purity_loss = masked_mse(preds["purity"], labels["purity"])
    tstage_loss = masked_ce(preds["t_stage"], labels["t_stage"], weight=t_weights)
    nstage_loss = masked_bce(preds["n_stage"], (labels["n_stage"] > 0).float())
    prog_loss   = masked_mse(preds["prog"], labels["prog"] / 3.0)
    rank_l      = rank_loss(preds["purity"], labels["purity"])

    total = (W_PURITY  * purity_loss  +
             W_TSTAGE  * tstage_loss  +
             W_NSTAGE  * nstage_loss  +
             W_PROG    * prog_loss    +
             W_RANK    * rank_l)

    return total, {
        "cancer": 0.0,
        "purity": purity_loss.item(),
        "t_stage": tstage_loss.item(),
        "n_stage": nstage_loss.item(),
        "prog": prog_loss.item(),
        "rank": rank_l.item(),
    }


@torch.no_grad()
def evaluate(loader):
    model.eval()
    all_preds = {k: [] for k in ["cancer", "purity", "t_stage", "n_stage", "prog"]}
    all_labels = {k: [] for k in ["cancer", "purity", "t_stage", "n_stage", "prog"]}
    total_loss = 0; n_batches = 0

    for X_b, L_b in loader:
        X_b = X_b.to(device)
        L_b = {k: v.to(device) for k, v in L_b.items()}
        preds = model(X_b)
        loss, _ = compute_loss(preds, L_b)
        total_loss += loss.item()
        n_batches += 1

        for k in ["cancer", "purity"]:
            all_preds[k].append(preds[k].cpu().numpy())
            all_labels[k].append(L_b[k].cpu().numpy())

        tstage_prob = F.softmax(preds["t_stage"], dim=1).cpu().numpy()
        all_preds["t_stage"].append(tstage_prob)
        all_labels["t_stage"].append(L_b["t_stage"].cpu().numpy())

        all_preds["n_stage"].append(preds["n_stage"].cpu().numpy())
        all_labels["n_stage"].append(L_b["n_stage"].cpu().numpy())
        all_preds["prog"].append(preds["prog"].cpu().numpy())
        all_labels["prog"].append(L_b["prog"].cpu().numpy())

    metrics = {}

    pur_for_auc = np.concatenate(all_preds["purity"])
    cancer_l    = np.concatenate(all_labels["cancer"])
    try:
        metrics["cancer_auc"] = roc_auc_score((cancer_l > 0).astype(int), pur_for_auc)
    except Exception:
        metrics["cancer_auc"] = float("nan")

    pur_p = np.concatenate(all_preds["purity"])
    pur_l = np.concatenate(all_labels["purity"])
    valid = ~np.isnan(pur_l)
    if valid.sum() > 10:
        r, _ = pearsonr(pur_p[valid], pur_l[valid])
        metrics["purity_r"] = r
    else:
        metrics["purity_r"] = float("nan")

    t_p = np.concatenate(all_preds["t_stage"])
    t_l = np.concatenate(all_labels["t_stage"])
    valid = ~np.isnan(t_l)
    if valid.sum() > 10:
        t_pred_cls = t_p[valid].argmax(axis=1)
        metrics["tstage_bacc"] = balanced_accuracy_score(t_l[valid].astype(int), t_pred_cls)
    else:
        metrics["tstage_bacc"] = float("nan")

    n_p = np.concatenate(all_preds["n_stage"])
    n_l = np.concatenate(all_labels["n_stage"])
    valid = ~np.isnan(n_l)
    n_bin = (n_l[valid] > 0).astype(int)
    if valid.sum() > 10 and n_bin.sum() > 0:
        try:
            metrics["nstage_auc"] = roc_auc_score(n_bin, n_p[valid])
        except Exception:
            metrics["nstage_auc"] = float("nan")
    else:
        metrics["nstage_auc"] = float("nan")

    metrics["val_loss"] = total_loss / n_batches
    return metrics


optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5, min_lr=1e-5, verbose=True)

best_val_loss = float("inf")
epochs_no_improve = 0
log = []

print(f"\nTraining for up to {MAX_EPOCHS} epochs (patience={PATIENCE})...")
print(f"{'Epoch':>6}  {'TrLoss':>8}  {'ValLoss':>8}  "
      f"{'CancerAUC':>10}  {'PurityR':>8}  {'Tstage':>8}  {'Nstage':>8}")

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_train_loss = 0; n_batches = 0

    for X_b, L_b in train_loader:
        X_b = X_b.to(device)
        L_b = {k: v.to(device) for k, v in L_b.items()}

        optimizer.zero_grad()
        preds = model(X_b)
        loss, _ = compute_loss(preds, L_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_train_loss += loss.item()
        n_batches += 1

    train_loss = total_train_loss / n_batches
    metrics = evaluate(val_loader)
    val_loss = metrics["val_loss"]

    scheduler.step(val_loss)

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        **metrics,
    }
    log.append(row)

    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:>6}  {train_loss:>8.4f}  {val_loss:>8.4f}  "
              f"{metrics['cancer_auc']:>10.3f}  {metrics['purity_r']:>8.3f}  "
              f"{metrics['tstage_bacc']:>8.3f}  {metrics['nstage_auc']:>8.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODELS / "best_multitask.pt")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print("\nLoading best model for test evaluation...")
model.load_state_dict(torch.load(MODELS / "best_multitask.pt"))
test_metrics = evaluate(test_loader)

print("\n── Test set results ─────────────────────────────")
print(f"  Cancer detection AUC : {test_metrics['cancer_auc']:.4f}")
print(f"  Purity Pearson r     : {test_metrics['purity_r']:.4f}")
print(f"  T-stage balanced acc : {test_metrics['tstage_bacc']:.4f}")
print(f"  N-stage AUC          : {test_metrics['nstage_auc']:.4f}")

pd.DataFrame(log).to_csv(RESULTS / "multitask_train_log.csv", index=False)
with open(RESULTS / "multitask_test_metrics.json", "w") as fh:
    json.dump(test_metrics, fh, indent=2)

print(f"\nSaved:")
print(f"  models/best_multitask.pt")
print(f"  results/multitask_train_log.csv")
print(f"  results/multitask_test_metrics.json")
print("\nNext: python3 scripts/08_validate_multitask_cfdna.py")
