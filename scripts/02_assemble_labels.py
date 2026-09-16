import pandas as pd
import numpy as np
import h5py

# ── 1. Load HDF5 sample IDs ───────────────────────────────────────────────────
with h5py.File("data/processed/methylation_matrix.h5", "r") as f:
    sample_ids = [s.decode() for s in f["sample_ids"][:]]
print(f"HDF5 samples: {len(sample_ids)}")

# ── 2. UUID mapping: file_uuid → folder_uuid ─────────────────────────────────
uuid_map = pd.read_csv("data/raw/uuid_mapping.tsv", sep="\t")

# ── 3. Manifest: folder_uuid → case_id ───────────────────────────────────────
manifest = pd.read_csv("data/raw/files_manifest.tsv", sep="\t",
                       names=["sample_type","case_id","file_id","id"], header=0)
manifest = manifest[manifest["sample_type"] == "Primary Tumor"][["file_id","case_id"]].drop_duplicates()
manifest.columns = ["folder_uuid","case_id"]

# ── 4. Chain UUIDs to case_id ─────────────────────────────────────────────────
df = pd.DataFrame({"file_uuid": sample_ids})
df = df.merge(uuid_map, on="file_uuid", how="left")
df = df.merge(manifest, on="folder_uuid", how="left")
print(f"Matched to case_id: {df['case_id'].notna().sum()} / {len(df)}")

# ── 5. Slide: percent_tumor_cells ─────────────────────────────────────────────
slide = pd.read_csv("data/raw/biospecimen/slide.tsv", sep="\t", low_memory=False)
slide = slide[["cases.submitter_id","slides.percent_tumor_cells"]].copy()
slide.columns = ["case_id","percent_tumor_cells"]
slide = slide[slide["percent_tumor_cells"] != "'--"]
slide["percent_tumor_cells"] = pd.to_numeric(slide["percent_tumor_cells"], errors="coerce")
slide = slide.dropna(subset=["percent_tumor_cells"])
slide = slide.groupby("case_id")["percent_tumor_cells"].mean().reset_index()
print(f"Slide cases with valid purity: {len(slide)}")

# ── 6. Clinical: T and N stage ────────────────────────────────────────────────
clin = pd.read_csv("data/raw/clinical/clinical.tsv", sep="\t", low_memory=False)
clin = clin[["cases.submitter_id","diagnoses.ajcc_pathologic_t","diagnoses.ajcc_pathologic_n"]].copy()
clin.columns = ["case_id","t_stage","n_stage"]
clin = clin.replace("'--", np.nan).drop_duplicates("case_id")
print(f"Clinical T coverage: {clin['t_stage'].notna().sum()} / {len(clin)}")
print(f"Clinical N coverage: {clin['n_stage'].notna().sum()} / {len(clin)}")
print(f"T stage values: {sorted(clin['t_stage'].dropna().unique())}")
print(f"N stage values: {sorted(clin['n_stage'].dropna().unique())}")

# ── 7. Overlap: overall stage ─────────────────────────────────────────────────
overlap = pd.read_csv("data/raw/triple_overlap_cases.tsv", sep="\t")

# ── 8. Merge all ──────────────────────────────────────────────────────────────
df = df.merge(slide,   on="case_id", how="left")
df = df.merge(clin,    on="case_id", how="left")
df = df.merge(overlap[["case_id","stage"]], on="case_id", how="left")

print(f"\nFinal shape: {df.shape}")
print(f"percent_tumor_cells: {df['percent_tumor_cells'].notna().sum()} / {len(df)}")
print(f"t_stage:             {df['t_stage'].notna().sum()} / {len(df)}")
print(f"n_stage:             {df['n_stage'].notna().sum()} / {len(df)}")
print(f"stage:               {df['stage'].notna().sum()} / {len(df)}")
print("\nSample rows:")
print(df.head(5).to_string())

# ── 9. Save ───────────────────────────────────────────────────────────────────
df.to_csv("data/labels/labels.tsv", sep="\t", index=False)
print("\nSaved to data/labels/labels.tsv")
