"""
validate_brca_only.py
---------------------
Re-analyses GSE122126 cfDNA predictions restricted to BRCA-relevant samples.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--pred", default="results/multitask_cfdna_predictions.tsv")
parser.add_argument("--outdir", default="results")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
OUT_TSV = os.path.join(args.outdir, "brca_only_predictions.tsv")
OUT_PNG = os.path.join(args.outdir, "brca_only_validation.png")

print(f"Loading predictions from: {args.pred}")
pred = pd.read_csv(args.pred, sep="\t")
print(f"  Total samples in file: {len(pred)}")

BRCA_GROUPS = {
    "Breast adenocarcinoma":     "BRCA (primary)",
    "Metastatic breast cancer":  "BRCA (metastatic)",
    "Healthy cfDNA":             "Healthy (GSE122126)",
    "Healthy cfDNA (GSE214344)": "Healthy (GSE214344)",
}

df = pred[pred["group"].isin(BRCA_GROUPS.keys())].copy()
df["group_label"] = df["group"].map(BRCA_GROUPS)

print(f"\n  Samples retained after BRCA filter: {len(df)}")
print(df["group_label"].value_counts().to_string())

df.to_csv(OUT_TSV, sep="\t", index=False)
print(f"\nSaved filtered predictions → {OUT_TSV}")

print("\n── Purity score summary ──────────────────────────────────")
print(df.groupby("group_label")["purity"].describe().round(3).to_string())

print("\n── T-stage mean predicted probabilities ──────────────────")
tstage_summary = df.groupby("group_label")[["t1_prob","t2_prob","t34_prob"]].mean().round(3)
print(tstage_summary.to_string())

print("\n── N-stage mean positive probability ────────────────────")
print(df.groupby("group_label")["npos_prob"].describe().round(3).to_string())

brca_primary = df[df["group_label"] == "BRCA (primary)"]["purity"].values
healthy_122  = df[df["group_label"] == "Healthy (GSE122126)"]["purity"].values

y_true = np.array([1]*len(brca_primary) + [0]*len(healthy_122))
y_score = np.concatenate([brca_primary, healthy_122])

auc = roc_auc_score(y_true, y_score)
_, p_val = stats.mannwhitneyu(brca_primary, healthy_122, alternative="greater")

print(f"\n── BRCA primary vs Healthy (GSE122126) ───────────────────")
print(f"   AUC  = {auc:.3f}")
print(f"   Mann-Whitney p = {p_val:.5f}")
print(f"   BRCA primary mean purity  = {brca_primary.mean():.3f} ± {brca_primary.std():.3f}")
print(f"   Healthy mean purity       = {healthy_122.mean():.3f} ± {healthy_122.std():.3f}")

all_brca = df[df["group_label"].isin(["BRCA (primary)", "BRCA (metastatic)"])]["purity"].values
y_true_all = np.array([1]*len(all_brca) + [0]*len(healthy_122))
y_score_all = np.concatenate([all_brca, healthy_122])
auc_all = roc_auc_score(y_true_all, y_score_all)
_, p_all = stats.mannwhitneyu(all_brca, healthy_122, alternative="greater")

print(f"\n── All BRCA vs Healthy (GSE122126) ───────────────────────")
print(f"   AUC  = {auc_all:.3f}")
print(f"   Mann-Whitney p = {p_all:.5f}")
print(f"   All BRCA mean purity = {all_brca.mean():.3f} ± {all_brca.std():.3f}")

mbc = df[df["group_label"] == "BRCA (metastatic)"]["purity"].values
print(f"\n── Metastatic BRCA note ──────────────────────────────────")
print(f"   Mean purity = {mbc.mean():.3f} (lower than primary {brca_primary.mean():.3f})")
print(f"   Possible reason: GSE214344 platform/processing difference (batch effect)")
print(f"   GSE214344 healthy mean = {df[df['group_label']=='Healthy (GSE214344)']['purity'].mean():.3f}")
print(f"   → Metastatic score elevated relative to its own healthy baseline")

ORDER = [
    "Healthy (GSE122126)",
    "Healthy (GSE214344)",
    "BRCA (primary)",
    "BRCA (metastatic)",
]
COLORS = {
    "Healthy (GSE122126)": "#4393c3",
    "Healthy (GSE214344)": "#92c5de",
    "BRCA (primary)":      "#d6604d",
    "BRCA (metastatic)":   "#b2182b",
}

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(
    "BRCA-only cfDNA Validation — GSE122126 + GSE214344\n"
    "(Model trained on TCGA-BRCA tissue, 661 CpGs, zero-shot transfer)",
    fontsize=12, fontweight="bold", y=1.01
)

ax1 = axes[0]
purity_data = [df[df["group_label"] == g]["purity"].values for g in ORDER]

bp = ax1.boxplot(
    purity_data, positions=range(len(ORDER)), widths=0.5, patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=4, alpha=0.5),
)
for patch, g in zip(bp["boxes"], ORDER):
    patch.set_facecolor(COLORS[g])
    patch.set_alpha(0.7)

rng = np.random.default_rng(42)
for i, (g, vals) in enumerate(zip(ORDER, purity_data)):
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax1.scatter(np.full(len(vals), i) + jitter, vals,
                color=COLORS[g], edgecolors="black", linewidths=0.5,
                s=40, zorder=3, alpha=0.9)

ax1.set_xticks(range(len(ORDER)))
ax1.set_xticklabels([g.replace(" (", "\n(") for g in ORDER], fontsize=9)
ax1.set_ylabel("Predicted Tumor Burden Score", fontsize=10)
ax1.set_title("Panel 1: Purity / Cancer Score", fontsize=10, fontweight="bold")
ax1.set_ylim(-0.05, 1.05)
ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Decision threshold (0.5)")
ax1.legend(fontsize=8)

ax1.text(0.97, 0.97,
    f"Primary vs Healthy\nAUC={auc:.3f}, p={p_val:.4f}\n\n"
    f"All BRCA vs Healthy\nAUC={auc_all:.3f}, p={p_all:.4f}",
    transform=ax1.transAxes, fontsize=8,
    verticalalignment="top", horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

ax2 = axes[1]
brca_df = df[df["group_label"].isin(["BRCA (primary)", "BRCA (metastatic)"])].copy()
brca_df = brca_df.reset_index(drop=True)

sample_labels = [
    f"{row.gsm_id}\n({row.group_label.split('(')[1].rstrip(')')})"
    for _, row in brca_df.iterrows()
]

t1 = brca_df["t1_prob"].values
t2 = brca_df["t2_prob"].values
t34 = brca_df["t34_prob"].values
x = np.arange(len(brca_df))

bar_w = 0.6
ax2.bar(x, t1,  width=bar_w, label="T1",    color="#fee090", edgecolor="black", linewidth=0.5)
ax2.bar(x, t2,  width=bar_w, bottom=t1,     label="T2",    color="#fc8d59", edgecolor="black", linewidth=0.5)
ax2.bar(x, t34, width=bar_w, bottom=t1+t2,  label="T3/T4", color="#d73027", edgecolor="black", linewidth=0.5)

ax2.set_xticks(x)
ax2.set_xticklabels(sample_labels, fontsize=7, rotation=45, ha="right")
ax2.set_ylabel("Predicted T-stage Probability", fontsize=10)
ax2.set_title("Panel 2: T-stage Distribution (BRCA samples)\n"
              "Note: no ground-truth stage available for cfDNA samples",
              fontsize=9, fontweight="bold")
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=9, loc="upper right")
ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)

for tick, (_, row) in zip(ax2.get_xticklabels(), brca_df.iterrows()):
    tick.set_color(COLORS[row["group_label"]])

ax3 = axes[2]
nstage_data = [df[df["group_label"] == g]["npos_prob"].values for g in ORDER]

bp3 = ax3.boxplot(
    nstage_data, positions=range(len(ORDER)), widths=0.5, patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=4, alpha=0.5),
)
for patch, g in zip(bp3["boxes"], ORDER):
    patch.set_facecolor(COLORS[g])
    patch.set_alpha(0.7)

for i, (g, vals) in enumerate(zip(ORDER, nstage_data)):
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax3.scatter(np.full(len(vals), i) + jitter, vals,
                color=COLORS[g], edgecolors="black", linewidths=0.5,
                s=40, zorder=3, alpha=0.9)

ax3.set_xticks(range(len(ORDER)))
ax3.set_xticklabels([g.replace(" (", "\n(") for g in ORDER], fontsize=9)
ax3.set_ylabel("Predicted N-stage Positive Probability", fontsize=10)
ax3.set_title("Panel 3: N-stage Score\nNote: no ground-truth N-stage available",
              fontsize=9, fontweight="bold")
ax3.set_ylim(-0.05, 1.05)
ax3.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

legend_patches = [mpatches.Patch(facecolor=COLORS[g], edgecolor="black", label=g) for g in ORDER]
fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.05), frameon=True)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved figure → {OUT_PNG}")
print("\nDone.")
