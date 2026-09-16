"""
build_insilio_mixtures.py

Generates synthetic cfDNA training samples by mixing TCGA tumor methylomes
with blood reference methylomes at known tumor fractions.

For each TCGA sample at 8 mixing fractions:
  mixed_beta = f * tumor_beta + (1 - f) * blood_beta

The deviation input (obs - blood_mean) is computed here and stored
directly as the model's input feature, cancelling out the dilution magnitude
and preserving only the pattern relative to the blood baseline.

Outputs:
  data/processed/insilio_X.npy        — (N, n_targeted_cpgs) float32 deviation vectors
  data/processed/insilio_labels.npz   — all label arrays aligned to rows of X
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROCESSED   = Path("data/processed")
LABELS_FILE = Path("data/labels/labels_with_purity.tsv")

MIXING_FRACTIONS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]

print("Loading targeted CpG indices...")
targeted_idx   = np.load(PROCESSED / "targeted_cpg_idx.npy")
blood_ref_mean = np.load(PROCESSED / "blood_ref_mean_targeted.npy")
blood_targeted = np.load(PROCESSED / "blood_reference_targeted.npy")

n_targeted = len(targeted_idx)
n_blood    = blood_targeted.shape[0]
print(f"Targeted CpGs: {n_targeted}")
print(f"Blood samples: {n_blood}")

print("\nLoading TCGA methylation matrix...")
with h5py.File(PROCESSED / "methylation_matrix.h5", "r") as f:
    full_beta  = f["beta"][:]
    sample_ids = f["sample_ids"][:].astype(str)

tcga_targeted = full_beta[:, targeted_idx].astype(np.float32)

print("Imputing TCGA NaNs on targeted panel...")
col_med = np.nanmedian(tcga_targeted, axis=0)
for j in range(n_targeted):
    mask = np.isnan(tcga_targeted[:, j])
    tcga_targeted[mask, j] = col_med[j]

print("Loading labels...")
labels = pd.read_csv(LABELS_FILE, sep="\t")
labels = labels.drop_duplicates(subset="file_uuid", keep="first")
labels = labels.set_index("file_uuid")
labels = labels.reindex(sample_ids)

cpe_vals    = labels["CPE"].values.astype(np.float32)
absolute_v  = labels["ABSOLUTE"].values.astype(np.float32)
t_stage_ord = labels["t_stage_ord"].values.astype(np.float32)
n_stage_ord = labels["n_stage_ord"].values.astype(np.float32)

def encode_overall_stage(stage_str):
    if pd.isna(stage_str):
        return np.nan
    s = str(stage_str).upper()
    if s.startswith("STAGE IV") or s == "IV":
        return 3.0
    if s.startswith("STAGE III") or s.startswith("III"):
        return 2.0
    if s.startswith("STAGE II") or s.startswith("II"):
        return 1.0
    if s.startswith("STAGE I") or s.startswith("I"):
        return 0.0
    return np.nan

overall_stage = labels["stage"].apply(encode_overall_stage).values.astype(np.float32)

n_tcga       = len(sample_ids)
n_fractions  = len(MIXING_FRACTIONS)
total_rows   = n_tcga * n_fractions

print(f"\nGenerating {total_rows} synthetic cfDNA samples "
      f"({n_tcga} samples × {n_fractions} fractions)...")

X_dev = np.zeros((total_rows, n_targeted), dtype=np.float32)

out_tumor_fraction = np.zeros(total_rows, dtype=np.float32)
out_cpe            = np.zeros(total_rows, dtype=np.float32)
out_absolute       = np.zeros(total_rows, dtype=np.float32)
out_t_stage        = np.full(total_rows, np.nan, dtype=np.float32)
out_n_stage        = np.full(total_rows, np.nan, dtype=np.float32)
out_overall_stage  = np.full(total_rows, np.nan, dtype=np.float32)
out_is_cancer      = np.zeros(total_rows, dtype=np.float32)
out_sample_idx     = np.zeros(total_rows, dtype=np.int32)

row = 0
for i in tqdm(range(n_tcga), desc="Tumor samples"):
    tumor_betas = tcga_targeted[i]

    blood_idx = np.random.randint(0, n_blood)
    blood_betas = blood_targeted[blood_idx]

    for f in MIXING_FRACTIONS:
        if f == 0.0:
            deviation = blood_betas - blood_ref_mean
            deviation = np.clip(deviation, -1.0, 1.0)
        else:
            mixed = f * tumor_betas + (1.0 - f) * blood_betas
            deviation = mixed - blood_ref_mean
            deviation = np.clip(deviation, -1.0, 1.0)

        X_dev[row]              = deviation
        out_tumor_fraction[row] = f
        out_cpe[row]            = cpe_vals[i] * f
        out_absolute[row]       = absolute_v[i] * f
        out_t_stage[row]        = t_stage_ord[i]
        out_n_stage[row]        = n_stage_ord[i]
        out_overall_stage[row]  = overall_stage[i]
        out_is_cancer[row]      = 1.0 if f > 0 else 0.0
        out_sample_idx[row]     = i

        row += 1

print(f"\nGenerated {row} total samples")
print(f"Cancer samples (f>0): {out_is_cancer.sum():.0f}")
print(f"Healthy samples (f=0): {(out_is_cancer==0).sum():.0f}")
print(f"Fraction distribution: {dict(zip(MIXING_FRACTIONS, [(out_tumor_fraction==f).sum() for f in MIXING_FRACTIONS]))}")

X_path = PROCESSED / "insilio_X.npy"
L_path = PROCESSED / "insilio_labels.npz"

print(f"\nSaving X matrix: {X_dev.shape}...")
np.save(X_path, X_dev)

print(f"Saving labels...")
np.savez(L_path,
    tumor_fraction = out_tumor_fraction,
    cpe            = out_cpe,
    absolute       = out_absolute,
    t_stage        = out_t_stage,
    n_stage        = out_n_stage,
    overall_stage  = out_overall_stage,
    is_cancer      = out_is_cancer,
    sample_idx     = out_sample_idx,
)

print(f"\nSaved:")
print(f"  insilio_X.npy      {X_dev.shape} ({X_dev.nbytes / 1e6:.1f} MB)")
print(f"  insilio_labels.npz labels for all {row} rows")
print(f"\nLabel NaN rates:")
for name, arr in [("CPE", out_cpe), ("ABSOLUTE", out_absolute),
                  ("T-stage", out_t_stage), ("N-stage", out_n_stage),
                  ("Overall stage", out_overall_stage)]:
    nan_r = np.isnan(arr).mean()
    print(f"  {name:15s}: {nan_r*100:.1f}% NaN")

print("\nNext: python3 scripts/07_train_multitask.py")
