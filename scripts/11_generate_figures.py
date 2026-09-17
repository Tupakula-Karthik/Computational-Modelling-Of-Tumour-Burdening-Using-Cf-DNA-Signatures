"""
cfDNA Methylation Project — Figure Generation Script
=====================================================
Generates 2 publication-quality figures from your existing results:
  1. fig_burden_ordered.png   — Tumour burden scores by disease group (GSE122126)
  2. fig_roc_cfdna.png        — ROC for zero-shot cfDNA cancer detection

fig_roc_tissue (held-out tissue test ROC) is produced by
10_fig_purity_scatter.py's sibling logic / 07_train_multitask.py's
saved test metrics — not duplicated here, to avoid the filename
collision the original two scripts had.

Usage:
    python3 scripts/11_generate_figures.py

All outputs saved to: results/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc

mpl.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          12,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         150,
})

OUTDIR = 'results'
os.makedirs(OUTDIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE — Predicted tumour burden scores ordered by median (GSE122126)
# ─────────────────────────────────────────────────────────────────────────────
def fig_burden_ordered():
    pred_path = os.path.join(OUTDIR, 'multitask_cfdna_predictions.tsv')
    if not os.path.exists(pred_path):
        print(f'[Burden] Missing: {pred_path}  — SKIPPING')
        return

    df = pd.read_csv(pred_path, sep='\t')
    print(f'[Burden] Columns: {df.columns.tolist()}')

    group_col = 'group'
    score_col = 'purity'

    order = (df.groupby(group_col)[score_col]
               .median()
               .sort_values()
               .index.tolist())

    def get_colour(g):
        gl = g.lower()
        if 'healthy' in gl or 'normal' in gl:    return '#27AE60'
        if 'islet' in gl or 'transplant' in gl:  return '#3498DB'
        if 'breast' in gl:                        return '#C0392B'
        if 'colon' in gl or 'adenocarcinoma' in gl: return '#8E44AD'
        if 'lung' in gl or 'nsclc' in gl or 'sclc' in gl: return '#E67E22'
        if 'sepsis' in gl:                        return '#7F8C8D'
        return '#E74C3C'

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, grp in enumerate(order):
        vals = df[df[group_col] == grp][score_col].dropna().values
        col  = get_colour(grp)
        ax.scatter([i] * len(vals), vals, alpha=0.55, s=32, color=col, zorder=3)
        ax.hlines(np.median(vals), i - 0.38, i + 0.38, lw=2.8, color=col, zorder=4)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([g.replace(' ', '\n') for g in order], fontsize=9)
    ax.set_ylabel('Predicted Tumour Burden Score')
    ax.set_title(
        'Predicted Tumour Burden Scores Ordered by Median\n(GSE122126 — Zero-Shot cfDNA Validation)',
        fontweight='bold')
    ax.set_xlim(-0.65, len(order) - 0.35)
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.5, ls='--', lw=1, color='grey', alpha=0.35)
    plt.tight_layout()
    out = os.path.join(OUTDIR, 'fig_burden_ordered.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Burden] Saved → {out}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE — ROC for zero-shot cfDNA cancer detection (GSE122126)
# FIX: only real cancer patients count as positive, only real healthy
# controls count as negative. Sepsis patients, "cell type reference",
# and "in vitro mix" are excluded entirely — they are not cancer/healthy
# comparisons and inflating the AUC with them is misleading.
# ─────────────────────────────────────────────────────────────────────────────
def fig_roc_cfdna():
    pred_path = os.path.join(OUTDIR, 'multitask_cfdna_predictions.tsv')
    if not os.path.exists(pred_path):
        print(f'[ROC cfDNA] Missing: {pred_path}  — SKIPPING')
        return

    df = pd.read_csv(pred_path, sep='\t')

    CANCER_GROUPS = {
        'Breast adenocarcinoma', 'Metastatic breast cancer',
        'Colon adenocarcinoma', 'NSCLC', 'SCLC',
        'Cancer (unknown primary)',
    }
    HEALTHY_GROUPS = {
        'Healthy cfDNA', 'Healthy cfDNA (GSE214344)',
    }
    # Excluded on purpose: Sepsis, Islet transplant recipient,
    # Healthy tissue, In vitro mix, Cell type reference — none of these
    # are a cancer-vs-healthy-cfDNA comparison.

    mask = df['group'].isin(CANCER_GROUPS | HEALTHY_GROUPS)
    df_eval = df[mask].copy()
    df_eval['is_cancer'] = df_eval['group'].isin(CANCER_GROUPS).astype(int)

    y_true  = df_eval['is_cancer'].values
    y_score = df_eval['purity'].values
    n_cancer  = y_true.sum()
    n_healthy = (y_true == 0).sum()
    print(f'[ROC cfDNA] Cancer n={n_cancer}  Healthy n={n_healthy}  '
          f'(excluded {len(df) - len(df_eval)} non-comparable samples)')

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, lw=2.5, color='#2980B9',
            label=f'Zero-Shot cfDNA ROC\nAUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.45)
    ax.fill_between(fpr, tpr, alpha=0.10, color='#2980B9')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(
        'Receiver Operating Characteristic\nZero-Shot Cancer Detection (GSE122126 Plasma cfDNA)',
        fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.text(0.98, 0.08,
            f'Cancer n={n_cancer}  |  Healthy n={n_healthy}',
            transform=ax.transAxes, ha='right', fontsize=9, color='#555555')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    plt.tight_layout()
    out = os.path.join(OUTDIR, 'fig_roc_cfdna.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[ROC cfDNA] Saved → {out}')


if __name__ == '__main__':
    print('=' * 55)
    print(' cfDNA Figure Generator')
    print('=' * 55)
    fig_burden_ordered()
    fig_roc_cfdna()
    print('=' * 55)
    print('Done. Check results/ for output PNGs.')
