import pandas as pd
import numpy as np

df = pd.read_csv("data/labels/labels.tsv", sep="\t")

# T stage → ordinal 0-4 (TX/Tis → NaN)
t_map = {
    'T1':0,'T1a':0,'T1b':0,'T1c':0,
    'T2':1,'T2a':1,'T2b':1,
    'T3':2,'T3a':2,
    'T4':3,'T4b':3,'T4d':3,
}
df["t_stage_ord"] = df["t_stage"].map(t_map)

# N stage → ordinal 0-3 (NX → NaN)
n_map = {
    'N0':0,'N0 (i+)':0,'N0 (i-)':0,'N0 (mol+)':0,
    'N1':1,'N1a':1,'N1b':1,'N1c':1,'N1mi':1,
    'N2':2,'N2a':2,
    'N3':3,'N3a':3,'N3b':3,
}
df["n_stage_ord"] = df["n_stage"].map(n_map)

# Normalize percent_tumor_cells to 0-1
df["cellularity"] = df["percent_tumor_cells"] / 100.0

print(f"T stage ordinal coverage: {df['t_stage_ord'].notna().sum()} / {len(df)}")
print(f"N stage ordinal coverage: {df['n_stage_ord'].notna().sum()} / {len(df)}")
print(f"\nT distribution:\n{df['t_stage_ord'].value_counts().sort_index()}")
print(f"\nN distribution:\n{df['n_stage_ord'].value_counts().sort_index()}")
print(f"\nCellularity stats:\n{df['cellularity'].describe()}")

df.to_csv("data/labels/labels.tsv", sep="\t", index=False)
print("\nUpdated labels.tsv saved.")
