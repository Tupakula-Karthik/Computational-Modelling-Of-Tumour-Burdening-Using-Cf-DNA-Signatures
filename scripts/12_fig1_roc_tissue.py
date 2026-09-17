import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from sklearn.metrics import roc_curve, auc

mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                     'axes.spines.top':False,'axes.spines.right':False})

OUTDIR = 'results'

df = pd.read_csv(os.path.join(OUTDIR, 'multitask_cfdna_predictions.tsv'), sep='\t')

y_true  = (df['is_healthy'] == 0).astype(int).values
y_score = df['cancer_prob'].values

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(fpr, tpr, lw=2, color='#C0392B', label=f'ROC  AUC = {roc_auc:.3f}')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.45)
ax.fill_between(fpr, tpr, alpha=0.08, color='#C0392B')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Zero-Shot Cancer Detection\n(GSE122126 + GSE214344 Plasma cfDNA)', fontweight='bold')
ax.legend(loc='lower right')
ax.set_xlim([0,1]); ax.set_ylim([0,1.01])
plt.tight_layout()
out = os.path.join(OUTDIR, 'fig_roc_tissue.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved → {out}')
