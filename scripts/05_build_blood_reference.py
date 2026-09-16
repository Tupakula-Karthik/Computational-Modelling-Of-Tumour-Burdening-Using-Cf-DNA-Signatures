"""
build_blood_reference.py

Parses GSE40279 series matrix to extract blood methylation beta values,
then performs differential methylation analysis vs TCGA tumor samples
to select the targeted 661-CpG panel for the deviation encoder.

Outputs:
  data/processed/blood_reference.npy       — (n_blood, n_cpg) median blood betas
  data/processed/blood_ref_mean.npy        — (n_cpg,) mean blood beta per CpG
  data/processed/blood_ref_std.npy         — (n_cpg,) std blood beta per CpG
  data/processed/targeted_cpg_idx.npy      — indices into original 20K HDF5 CpGs
  data/processed/targeted_cpg_ids.npy      — CpG probe IDs for targeted panel
  data/processed/cpg_delta_stats.tsv       — full differential stats for all CpGs
"""

import gzip
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from tqdm import tqdm

PROCESSED = Path("data/processed")
GEO_CACHE = Path("data/raw/geo_cache")
LABELS_FILE = Path("data/labels/labels_with_purity.tsv")

HDF5_FILE = PROCESSED / "methylation_matrix.h5"
BLOOD_MATRIX_FILE = GEO_CACHE / "GSE40279_series_matrix.txt.gz"

TOP_N_PER_STAGE = 500
MIN_DELTA_BETA = 0.2
MIN_EFFECT_SIZE = 0.3

print("Loading TCGA HDF5 methylation matrix...")
with h5py.File(HDF5_FILE, "r") as f:
    tcga_beta = f["beta"][:]
    cpg_ids = f["cpg_ids"][:].astype(str)
    sample_ids = f["sample_ids"][:].astype(str)

print(f"TCGA shape: {tcga_beta.shape}")
print(f"CpGs: {len(cpg_ids)}")

labels = pd.read_csv(LABELS_FILE, sep="\t")
labels = labels.drop_duplicates(subset="file_uuid", keep="first")
labels = labels.set_index("file_uuid")
labels = labels.reindex(sample_ids)

t_ord = labels["t_stage_ord"].values
t1_mask = t_ord == 0
t2_mask = t_ord == 1
t34_mask = t_ord >= 2

print(f"T1: {t1_mask.sum()}, T2: {t2_mask.sum()}, T3/T4: {t34_mask.sum()}")

print("Imputing TCGA NaNs...")
col_medians = np.nanmedian(tcga_beta, axis=0)
nan_mask = np.isnan(tcga_beta)
for j in range(tcga_beta.shape[1]):
    tcga_beta[nan_mask[:, j], j] = col_medians[j]


def parse_series_matrix_blood(path, target_cpgs):
    target_set = set(target_cpgs)
    cpg_to_idx = {c: i for i, c in enumerate(target_cpgs)}

    n_samples = None
    gsm_ids = []
    rows = {}
    in_table = False

    print(f"Streaming blood matrix from: {path}")
    with gzip.open(path, "rt", errors="replace") as fh:
        for raw_line in tqdm(fh, unit="lines", mininterval=2.0):
            line = raw_line.rstrip("\n")

            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break

            if 'ID_REF' in line and (line.startswith('ID_REF') or line.startswith('"ID_REF"')):
                parts = line.split("\t")
                gsm_ids = [p.strip('"').strip() for p in parts[1:] if p.strip().strip('"')]
                n_samples = len(gsm_ids)
                in_table = True
                continue

            if not in_table:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue
            cpg = parts[0].strip('"')
            if cpg not in target_set:
                continue
            try:
                end = n_samples + 1 if n_samples is not None else len(parts)
                vals = [float(v) if v not in ("", "null", "NA", "NaN") else np.nan
                        for v in parts[1:end]]
                rows[cpg] = vals
            except ValueError:
                continue

    if not gsm_ids:
        raise RuntimeError("No sample IDs found in series matrix header.")

    beta_mat = np.full((len(target_cpgs), len(gsm_ids)), np.nan, dtype=np.float32)
    for cpg, vals in rows.items():
        i = cpg_to_idx[cpg]
        beta_mat[i, :len(vals)] = vals

    print(f"  Matched {len(rows)} / {len(target_cpgs)} target CpGs")
    print(f"  Blood samples: {len(gsm_ids)}")
    return beta_mat.T, gsm_ids


if not BLOOD_MATRIX_FILE.exists() or BLOOD_MATRIX_FILE.stat().st_size == 0:
    raise FileNotFoundError(
        f"Blood matrix not found or empty: {BLOOD_MATRIX_FILE}\n"
        f"Run: bash scripts/download_gse40279.sh"
    )

print("Parsing blood methylomes...")
blood_beta, gsm_ids = parse_series_matrix_blood(BLOOD_MATRIX_FILE, cpg_ids)
print(f"Blood beta shape: {blood_beta.shape}")

print("Imputing blood NaNs...")
blood_col_median = np.nanmedian(blood_beta, axis=0)
blood_nan_mask = np.isnan(blood_beta)
for j in range(blood_beta.shape[1]):
    blood_beta[blood_nan_mask[:, j], j] = blood_col_median[j]

print("\nRunning differential methylation analysis...")

blood_mean = blood_beta.mean(axis=0)
blood_std  = blood_beta.std(axis=0) + 1e-8

results = []
for cpg_idx in tqdm(range(len(cpg_ids)), desc="CpGs"):
    b_vals = blood_beta[:, cpg_idx]

    for stage_label, stage_mask in [("T1", t1_mask), ("T2", t2_mask), ("T3T4", t34_mask)]:
        t_vals = tcga_beta[stage_mask, cpg_idx]

        if stage_mask.sum() < 5:
            continue

        t_mean = t_vals.mean()
        t_std  = t_vals.std()
        b_mn   = b_vals.mean()

        delta = t_mean - b_mn
        pooled_std = np.sqrt((t_std**2 + b_vals.std()**2) / 2) + 1e-8
        cohens_d = abs(delta) / pooled_std

        results.append({
            "cpg_idx": cpg_idx,
            "cpg_id": cpg_ids[cpg_idx],
            "stage": stage_label,
            "tumor_mean": t_mean,
            "blood_mean": b_mn,
            "delta_beta": delta,
            "abs_delta": abs(delta),
            "cohens_d": cohens_d,
        })

df = pd.DataFrame(results)
df.to_csv(PROCESSED / "cpg_delta_stats.tsv", sep="\t", index=False)
print(f"Saved differential stats: {len(df)} rows")

df_filtered = df[
    (df["abs_delta"] >= MIN_DELTA_BETA) &
    (df["cohens_d"] >= MIN_EFFECT_SIZE)
].copy()

print(f"\nCpGs passing filters (delta>={MIN_DELTA_BETA}, d>={MIN_EFFECT_SIZE}): "
      f"{df_filtered['cpg_idx'].nunique()} unique")

selected_idx = set()
for stage in ["T1", "T2", "T3T4"]:
    top = (df_filtered[df_filtered["stage"] == stage]
           .sort_values("cohens_d", ascending=False)
           .head(TOP_N_PER_STAGE)["cpg_idx"]
           .tolist())
    selected_idx.update(top)
    print(f"  {stage}: selected {len(top)} CpGs")

selected_idx = sorted(selected_idx)
print(f"\nFinal targeted panel: {len(selected_idx)} CpGs (union, deduplicated)")

targeted_cpg_ids = cpg_ids[selected_idx]

np.save(PROCESSED / "blood_ref_mean.npy", blood_mean)
np.save(PROCESSED / "blood_ref_std.npy",  blood_std)
np.save(PROCESSED / "targeted_cpg_idx.npy", np.array(selected_idx))
np.save(PROCESSED / "targeted_cpg_ids.npy", targeted_cpg_ids)

blood_mean_targeted = blood_mean[selected_idx]
np.save(PROCESSED / "blood_ref_mean_targeted.npy", blood_mean_targeted)

blood_targeted = blood_beta[:, selected_idx]
np.save(PROCESSED / "blood_reference_targeted.npy", blood_targeted.astype(np.float32))

print(f"\nSaved:")
print(f"  blood_ref_mean.npy           shape={blood_mean.shape}")
print(f"  blood_ref_mean_targeted.npy  shape={blood_mean_targeted.shape}")
print(f"  blood_reference_targeted.npy shape={blood_targeted.shape}")
print(f"  targeted_cpg_idx.npy         {len(selected_idx)} indices")
print(f"  targeted_cpg_ids.npy         {len(targeted_cpg_ids)} probe IDs")
print(f"  cpg_delta_stats.tsv          {len(df)} rows")

print("\nTop 10 CpGs by Cohen's d (T3/T4 vs blood):")
top10 = (df_filtered[df_filtered["stage"] == "T3T4"]
         .sort_values("cohens_d", ascending=False)
         .head(10)[["cpg_id", "delta_beta", "cohens_d", "tumor_mean", "blood_mean"]])
print(top10.to_string(index=False))

print("\nNext: python3 scripts/06_build_insilio_mixtures.py")
