import os
import glob
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

RAW_DIR   = "data/raw/tcga_brca_methylation"
OUT_HDF5  = "data/processed/methylation_matrix.h5"
TOP_N_CPG = 20000

# ── 1. Collect 450K files only ───────────────────────────────────────────────
files = glob.glob(os.path.join(RAW_DIR, "*", "*.methylation_array.sesame.level3betas.txt"))
files = [f for f in files if os.path.getsize(f) > 1_000_000]
print(f"Found {len(files)} methylation files (450K array only)")

# ── 2. CpG index from first file ─────────────────────────────────────────────
cpg_index = pd.read_csv(files[0], sep="\t", header=None, usecols=[0])[0].values
n_cpgs    = len(cpg_index)
n_samples = len(files)
print(f"CpGs: {n_cpgs:,}   Samples: {n_samples}")

# ── 3. Build full matrix ──────────────────────────────────────────────────────
matrix = np.full((n_samples, n_cpgs), np.nan, dtype=np.float32)

for i, fp in enumerate(tqdm(files, desc="Loading files")):
    df = pd.read_csv(fp, sep="\t", header=None)
    vals = df[1].values.astype(np.float32)
    matrix[i, :] = vals

print(f"Matrix shape: {matrix.shape}")
print(f"NaN rate: {np.isnan(matrix).mean():.3%}")
print(f"Non-NaN count: {(~np.isnan(matrix)).sum():,}")

# ── 4. Variance filter — skip all-NaN columns ────────────────────────────────
print("Computing per-CpG variance...")
col_valid = (~np.isnan(matrix)).sum(axis=0)
col_var   = np.nanvar(matrix, axis=0)
col_var[col_valid == 0] = -1

top_idx     = np.argsort(col_var)[::-1][:TOP_N_CPG]
top_idx     = np.sort(top_idx)
matrix_filt = matrix[:, top_idx]
cpg_filt    = cpg_index[top_idx]

print(f"Filtered matrix shape: {matrix_filt.shape}")
print(f"NaN rate after filter: {np.isnan(matrix_filt).mean():.3%}")
print(f"Min valid samples per CpG: {col_valid[top_idx].min()}")

# ── 5. Sample IDs (inner file UUID from filename) ────────────────────────────
sample_ids = [os.path.basename(f).split(".")[0] for f in files]

# ── 6. Save ───────────────────────────────────────────────────────────────────
print(f"Saving to {OUT_HDF5} ...")
with h5py.File(OUT_HDF5, "w") as f:
    f.create_dataset("beta",       data=matrix_filt, compression="gzip", chunks=True)
    f.create_dataset("sample_ids", data=np.array(sample_ids, dtype="S64"))
    f.create_dataset("cpg_ids",    data=np.array(cpg_filt,   dtype="S16"))

print("Done.")
