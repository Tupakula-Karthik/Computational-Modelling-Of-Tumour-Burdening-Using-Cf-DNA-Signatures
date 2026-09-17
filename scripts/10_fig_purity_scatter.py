import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import torch

mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False})

OUTDIR = 'results'

X      = np.load('data/processed/insilio_X.npy')
labels = np.load('data/processed/insilio_labels.npz')
cpe        = labels['cpe']
frac       = labels['tumor_fraction']
sample_idx = labels['sample_idx']

# Match the sample-level split used in training (07_train_multitask.py)
unique_samples = np.unique(sample_idx)
rng = np.random.default_rng(42)
rng.shuffle(unique_samples)
n = len(unique_samples)
n_train = int(0.80 * n)
n_val   = int(0.10 * n)
test_samples = set(unique_samples[n_train+n_val:])
test_idx = np.array([i for i, s in enumerate(sample_idx) if s in test_samples])

sys.path.insert(0, 'scripts')
from importlib import import_module
train_mod = import_module('07_train_multitask')
MultiTaskDeviationEncoder = train_mod.MultiTaskDeviationEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = MultiTaskDeviationEncoder(n_cpg=X.shape[1])
model.load_state_dict(torch.load('models/best_multitask.pt', map_location=device))
model.eval().to(device)

with torch.no_grad():
    out = model(torch.tensor(X[test_idx], dtype=torch.float32).to(device))
    y_pred = out["purity"].cpu().numpy().squeeze()

y_true    = cpe[test_idx]
y_frac    = frac[test_idx]
r, _      = pearsonr(y_true, y_pred)
r2        = r2_score(y_true, y_pred)
mae       = np.mean(np.abs(y_true - y_pred))

fracs   = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
labels_ = ['0%', '1%', '2%', '5%', '10%', '20%', '50%', '100%']
cmap    = plt.cm.RdYlGn
colours = [cmap(i / (len(fracs)-1)) for i in range(len(fracs))]

fig, ax = plt.subplots(figsize=(6.5, 6))

for f, lab, col in zip(fracs, labels_, colours):
    mask = np.isclose(y_frac, f, atol=0.001)
    ax.scatter(y_true[mask], y_pred[mask],
               alpha=0.45, s=12, color=col, label=lab, rasterized=True)

ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
ax.set_xlabel('True Purity (CPE)', fontsize=12)
ax.set_ylabel('Predicted Purity', fontsize=12)
ax.set_title('Purity Prediction on Held-Out Test Set\n(In Silico Mixtures, TCGA-BRCA)', fontweight='bold')

ax.text(0.04, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=11,
        color='#C0392B', va='top', fontweight='bold')
ax.text(0.04, 0.89, f'R² = {r2:.3f}', transform=ax.transAxes, fontsize=11,
        color='#C0392B', va='top')
ax.text(0.04, 0.83, f'MAE = {mae:.3f}', transform=ax.transAxes, fontsize=11,
        color='#C0392B', va='top')

leg = ax.legend(title='Tumour Fraction', loc='lower right',
                fontsize=8, title_fontsize=9, markerscale=1.8,
                framealpha=0.9, edgecolor='#cccccc')

ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
out = os.path.join(OUTDIR, 'fig_purity_scatter.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved → {out}')
print(f'r={r:.3f}  R²={r2:.3f}  MAE={mae:.3f}')
