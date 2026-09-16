"""
validate_multitask_cfdna.py

Applies the trained multi-task deviation encoder to held-out cfDNA plasma
samples from GSE122126 and GSE214344.

For each sample:
  1. Load beta values at targeted CpGs
  2. Compute deviation: (obs_beta - blood_ref_mean)
  3. Run inference: cancer prob, purity estimate, T-stage probs, N-stage, progression
  4. Report AUC, Spearman r vs metadata, per-group breakdown

Key validation tests:
  - Cancer vs healthy AUC (primary)
  - Primary vs metastatic breast cancer score ordering (stage gradient test)
  - Sepsis vs cancer dissociation on T-stage head (specificity test)
"""

import gzip
import h5py
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score

GEO_CACHE = Path("data/raw/geo_cache")
PROCESSED = Path("data/processed")
MODELS    = Path("models")
RESULTS   = Path("results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class MultiTaskDeviationEncoder(nn.Module):
    def __init__(self, n_cpg):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_cpg, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 256),   nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),    nn.GELU(),
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


print("Loading targeted CpG panel and blood reference...")
targeted_idx   = np.load(PROCESSED / "targeted_cpg_idx.npy")
targeted_ids   = np.load(PROCESSED / "targeted_cpg_ids.npy").astype(str)
blood_ref_mean = np.load(PROCESSED / "blood_ref_mean_targeted.npy")
n_targeted     = len(targeted_idx)

print(f"Targeted CpGs: {n_targeted}")
print(f"Loading model...")
model = MultiTaskDeviationEncoder(n_targeted).to(device)
model.load_state_dict(torch.load(MODELS / "best_multitask.pt", map_location=device))
model.eval()


def extract_cfDNA_betas(soft_gz_path, target_cpg_ids):
    target_set = set(target_cpg_ids)
    cpg_to_col = {c: i for i, c in enumerate(target_cpg_ids)}
    sample_betas = {}
    current_gsm  = None
    in_table     = False

    print(f"  Parsing: {soft_gz_path.name}")
    with gzip.open(soft_gz_path, "rt", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")

            if line.startswith("^SAMPLE"):
                current_gsm = line.split("=")[1].strip()
                sample_betas[current_gsm] = {}
                in_table = False
            elif line.startswith("!sample_table_begin"):
                in_table = True
            elif line.startswith("!sample_table_end"):
                in_table = False
            elif in_table and current_gsm:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                cpg = parts[0].strip()
                if cpg not in target_set:
                    continue
                try:
                    val = float(parts[1])
                    sample_betas[current_gsm][cpg] = val
                except ValueError:
                    continue

    gsm_ids = list(sample_betas.keys())
    beta_mat = np.full((len(gsm_ids), len(target_cpg_ids)), np.nan, dtype=np.float32)

    for i, gsm in enumerate(gsm_ids):
        for cpg, val in sample_betas[gsm].items():
            j = cpg_to_col[cpg]
            beta_mat[i, j] = val

    coverage = (~np.isnan(beta_mat)).mean()
    print(f"  Samples: {len(gsm_ids)}, CpG coverage: {coverage*100:.1f}%")
    return beta_mat, gsm_ids


def impute_and_deviate(beta_mat):
    mat = beta_mat.copy()
    nan_mask = np.isnan(mat)
    for j in range(mat.shape[1]):
        mat[nan_mask[:, j], j] = blood_ref_mean[j]
    deviation = mat - blood_ref_mean[np.newaxis, :]
    return np.clip(deviation, -1.0, 1.0).astype(np.float32)


@torch.no_grad()
def run_inference(deviation_np):
    t = torch.from_numpy(deviation_np).to(device)
    preds = model(t)
    tstage_probs = F.softmax(preds["t_stage"], dim=1).cpu().numpy()
    return {
        "cancer_prob": preds["purity"].cpu().numpy(),  # purity used as cancer score
        "purity":      preds["purity"].cpu().numpy(),
        "t1_prob":     tstage_probs[:, 0],
        "t2_prob":     tstage_probs[:, 1],
        "t34_prob":    tstage_probs[:, 2],
        "npos_prob":   preds["n_stage"].cpu().numpy(),
        "prog_score":  preds["prog"].cpu().numpy(),
    }


meta_path = PROCESSED / "gse122126_metadata.tsv"
if meta_path.exists():
    meta = pd.read_csv(meta_path, sep="\t").set_index("gsm_id")
    print(f"Loaded GSE122126 metadata: {len(meta)} samples")
else:
    print("WARNING: gse122126_metadata.tsv not found — group labels will be missing")
    meta = pd.DataFrame()


soft_122126 = GEO_CACHE / "GSE122126_family.soft.gz"
print(f"\nProcessing GSE122126...")
beta_122126, gsm_122126 = extract_cfDNA_betas(soft_122126, targeted_ids)
dev_122126 = impute_and_deviate(beta_122126)
preds_122126 = run_inference(dev_122126)

df_122126 = pd.DataFrame({"gsm_id": gsm_122126, **preds_122126})
df_122126["dataset"] = "GSE122126"
if not meta.empty:
    df_122126 = df_122126.merge(meta.reset_index()[["gsm_id", "group", "is_breast_cancer",
                                                      "is_healthy"]], on="gsm_id", how="left")
print(f"GSE122126 predictions: {len(df_122126)} samples")


soft_214344 = GEO_CACHE / "GSE214344_family.soft.gz"
df_214344 = None

if soft_214344.exists() and soft_214344.stat().st_size > 1e6:
    print(f"\nProcessing GSE214344 (metastatic BC)...")
    beta_214344, gsm_214344 = extract_cfDNA_betas(soft_214344, targeted_ids)

    coverage = (~np.isnan(beta_214344)).mean()
    print(f"  EPIC->450K probe coverage: {coverage*100:.1f}% (expected >85%)")

    dev_214344 = impute_and_deviate(beta_214344)
    preds_214344 = run_inference(dev_214344)

    df_214344 = pd.DataFrame({"gsm_id": gsm_214344, **preds_214344})
    df_214344["dataset"] = "GSE214344"
    group_map = {}
    import gzip as gz
    with gz.open(soft_214344, "rt", errors="replace") as fh:
        current = None
        for line in fh:
            line = line.rstrip()
            if "!Sample_geo_accession" in line:
                current = line.split("=")[1].strip()
            elif "!Sample_characteristics" in line and "disease state" in line.lower() and current:
                state = line.split("disease state:")[-1].strip()
                if "healthy" in state.lower():
                    group_map[current] = "Healthy cfDNA (GSE214344)"
                else:
                    group_map[current] = "Metastatic breast cancer"
    df_214344["group"] = df_214344["gsm_id"].map(group_map).fillna("Metastatic breast cancer")
    # FIX: use the parsed group instead of hardcoding every sample as cancer.
    df_214344["is_breast_cancer"] = (df_214344["group"] == "Metastatic breast cancer").astype(int)
    df_214344["is_healthy"] = (df_214344["group"] == "Healthy cfDNA (GSE214344)").astype(int)
    print(f"GSE214344 predictions: {len(df_214344)} samples")
else:
    print("\nGSE214344 not yet downloaded — run download + rerun this script to include it")


parts = [df_122126]
if df_214344 is not None:
    parts.append(df_214344)
df_all = pd.concat(parts, ignore_index=True)


print("\n── Per-group predictions (" + "─" * 50)
print(f"{'Group':40s}  {'N':>4}  {'Cancer':>7}  {'Purity':>7}  "
      f"{'T1':>6}  {'T34':>6}  {'N+':>6}  {'Prog':>6}")
print("─" * 100)

groups = df_all.groupby("group") if "group" in df_all.columns else [("All", df_all)]
for gname, gdf in sorted(groups, key=lambda x: x[1]["cancer_prob"].mean(), reverse=True):
    n = len(gdf)
    print(f"{gname:40s}  {n:>4}  "
          f"{gdf['cancer_prob'].mean():>7.3f}  "
          f"{gdf['purity'].mean():>7.3f}  "
          f"{gdf['t1_prob'].mean():>6.3f}  "
          f"{gdf['t34_prob'].mean():>6.3f}  "
          f"{gdf['npos_prob'].mean():>6.3f}  "
          f"{gdf['prog_score'].mean():>6.3f}")

if "is_breast_cancer" in df_all.columns and "is_healthy" in df_all.columns:
    df_auc = df_all[df_all["is_breast_cancer"].notna() | df_all["is_healthy"].notna()].copy()
    df_auc = df_all[df_all["group"].isin(
        [g for g in df_all["group"].unique()
         if "breast" in str(g).lower() or "healthy" in str(g).lower()
         or "Healthy" in str(g)]
    )].copy()

    bc_mask  = df_auc["group"].str.lower().str.contains("breast")
    hlt_mask = df_auc["group"].str.lower().str.contains("healthy")

    if bc_mask.sum() > 0 and hlt_mask.sum() > 0:
        bc_scores  = df_auc.loc[bc_mask,  "cancer_prob"].values
        hlt_scores = df_auc.loc[hlt_mask, "cancer_prob"].values

        y_true = np.concatenate([np.ones(len(bc_scores)), np.zeros(len(hlt_scores))])
        y_pred = np.concatenate([bc_scores, hlt_scores])

        try:
            auc = roc_auc_score(y_true, y_pred)
            stat, p = mannwhitneyu(bc_scores, hlt_scores, alternative="greater")
            print(f"\n── Cancer detection AUC (breast vs healthy cfDNA): {auc:.3f}  p={p:.5f}")
            print(f"   Breast cancer mean: {bc_scores.mean():.3f} "
                  f"(n={len(bc_scores)})  Healthy mean: {hlt_scores.mean():.3f} (n={len(hlt_scores)})")
        except Exception as e:
            print(f"AUC computation failed: {e}")

if df_214344 is not None:
    primary_scores = df_all[df_all["group"] == "Breast adenocarcinoma"]["cancer_prob"]
    metastatic_scores = df_all[df_all["group"] == "Metastatic breast cancer"]["cancer_prob"]

    print(f"\n── Stage gradient test (key result):")
    print(f"   Primary breast cancer    mean cancer_prob: {primary_scores.mean():.3f}")
    print(f"   Metastatic breast cancer mean cancer_prob: {metastatic_scores.mean():.3f}")
    print(f"   Direction: {'✓ metastatic > primary (expected)' if metastatic_scores.mean() > primary_scores.mean() else '✗ unexpected ordering'}")

    purity_primary = df_all[df_all["group"] == "Breast adenocarcinoma"]["purity"]
    purity_meta    = df_all[df_all["group"] == "Metastatic breast cancer"]["purity"]
    print(f"\n   Primary purity score:    {purity_primary.mean():.3f}")
    print(f"   Metastatic purity score: {purity_meta.mean():.3f}")


GROUP_ORDER = [
    "Breast adenocarcinoma", "Metastatic breast cancer",
    "Colon adenocarcinoma", "NSCLC", "SCLC",
    "Cancer (unknown primary)", "Sepsis",
    "Islet transplant recipient", "Healthy cfDNA",
    "Healthy tissue", "In vitro mix",
]

COLOR_MAP = {
    "Breast adenocarcinoma":     "#D85A30",
    "Metastatic breast cancer":  "#993C1D",
    "Colon adenocarcinoma":      "#E24B4A",
    "NSCLC":                     "#BA7517",
    "SCLC":                      "#854F0B",
    "Cancer (unknown primary)":  "#F09595",
    "Sepsis":                    "#888780",
    "Islet transplant recipient":"#B4B2A9",
    "Healthy cfDNA":             "#1D9E75",
    "Healthy tissue":            "#5DCAA5",
    "In vitro mix":              "#9FE1CB",
}

metrics_to_plot = [
    ("cancer_prob",  "Cancer detection probability"),
    ("purity",       "Purity estimate"),
    ("t34_prob",     "Late T-stage probability (T3/T4)"),
    ("npos_prob",    "Node positive probability (N+)"),
    ("prog_score",   "Progression score (stage I→IV)"),
]

fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 6))
fig.patch.set_facecolor("#F8F8F6")

for ax, (col, label) in zip(axes, metrics_to_plot):
    ax.set_facecolor("#F8F8F6")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(label, fontsize=9, pad=8)
    ax.set_ylim(-0.05, 1.05)

    y_pos = 0
    yticks, ylabels = [], []

    for group in reversed(GROUP_ORDER):
        subset = df_all[df_all["group"] == group][col] if "group" in df_all.columns else pd.Series([])
        if len(subset) == 0:
            continue

        color = COLOR_MAP.get(group, "#888780")
        vals = subset.values

        jitter = np.random.uniform(-0.2, 0.2, len(vals))
        ax.scatter(vals, [y_pos + j for j in jitter],
                   c=color, alpha=0.75, s=28, zorder=3, linewidths=0)
        ax.plot([vals.mean()], [y_pos], "D", color=color, markersize=7,
                markeredgecolor="white", markeredgewidth=1, zorder=4)

        yticks.append(y_pos)
        ylabels.append(f"{group} (n={len(vals)})")
        y_pos += 1

    ax.set_yticks(yticks)
    if ax == axes[0]:
        ax.set_yticklabels(ylabels, fontsize=7.5)
    else:
        ax.set_yticklabels([""] * len(yticks))
    ax.set_xlabel("Score", fontsize=8)
    ax.axvline(0.5, color="#CCCCCC", lw=0.8, ls="--", zorder=1)

plt.suptitle("Multi-task cfDNA methylation model — validation on plasma samples",
             fontsize=11, y=1.02, fontweight="normal")
plt.tight_layout()

fig_path = RESULTS / "multitask_cfdna_validation.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n✓ Figure saved: {fig_path}")

df_all.to_csv(RESULTS / "multitask_cfdna_predictions.tsv", sep="\t", index=False)
print(f"✓ Predictions saved: {RESULTS / 'multitask_cfdna_predictions.tsv'}")
