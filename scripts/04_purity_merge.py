"""
04_purity_merge.py

Downloads Aran et al. 2015 tumor purity estimates (ABSOLUTE, ESTIMATE,
LUMP, consensus CPE) and merges them onto labels.tsv by TCGA case_id.

Outputs:
  data/labels/labels_with_purity.tsv
"""

import re
import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path

LABELS_IN   = Path("data/labels/labels.tsv")
LABELS_OUT  = Path("data/labels/labels_with_purity.tsv")
RAW_DIR     = Path("data/raw")
PURITY_XLSX = RAW_DIR / "aran2015_purity.xlsx"

PURITY_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fncomms9971/MediaObjects/"
    "41467_2015_BFncomms9971_MOESM1236_ESM.xlsx"
)


def download_purity_table(force: bool = False):
    if PURITY_XLSX.exists() and not force:
        print(f"  Using cached {PURITY_XLSX}")
        return
    print("  Downloading Aran 2015 purity table...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(PURITY_URL, PURITY_XLSX)
    print(f"  Saved → {PURITY_XLSX}")


def barcode_to_case_id(barcode: str) -> str:
    """TCGA-XX-XXXX-01A-... → TCGA-XX-XXXX"""
    s = str(barcode)
    m = re.match(r"(TCGA-\w{2}-\w{4})", s)
    return m.group(1) if m else s


def main():
    print(f"Loading labels from {LABELS_IN}")
    labels = pd.read_csv(LABELS_IN, sep="\t")
    print(f"  {len(labels)} rows")

    download_purity_table()
    print(f"Reading {PURITY_XLSX} ...")
    purity_raw = pd.read_excel(PURITY_XLSX, sheet_name=0, header=3)
    print(f"  Shape: {purity_raw.shape}")
    print(f"  Columns: {list(purity_raw.columns)}")

    # Find the sample ID column
    id_col = None
    for col in purity_raw.columns:
        sample = purity_raw[col].dropna().astype(str).iloc[0]
        if sample.startswith("TCGA"):
            id_col = col
            break
    if id_col is None:
        raise ValueError(f"No TCGA barcode column found. Columns: {list(purity_raw.columns)}")
    print(f"  Using '{id_col}' as sample ID column")

    purity_raw["case_id"] = purity_raw[id_col].apply(barcode_to_case_id)
    purity_raw = purity_raw.drop_duplicates(subset="case_id").set_index("case_id")

    # Match purity columns by flexible name search
    col_map = {}
    for target, candidates in {
        "CPE":      ["CPE", "consensus", "PURITY", "purity"],
        "ABSOLUTE": ["ABSOLUTE", "absolute"],
        "ESTIMATE": ["ESTIMATE", "estimate"],
        "LUMP":     ["LUMP", "lump"],
    }.items():
        for c in candidates:
            if c in purity_raw.columns:
                col_map[target] = c
                break

    print(f"  Matched purity columns: {col_map}")
    if not col_map:
        raise ValueError(
            "No purity columns matched. Inspect purity_raw.columns above "
            "and update the candidate name lists."
        )

    purity_sub = purity_raw[list(col_map.values())].rename(
        columns={v: k for k, v in col_map.items()}
    )

    # Merge
    merged = labels.merge(purity_sub, on="case_id", how="left")

    for col in col_map.keys():
        n_valid = merged[col].notna().sum()
        print(f"  {col}: {n_valid}/{len(merged)} matched")

    merged.to_csv(LABELS_OUT, sep="\t", index=False)
    print(f"\nSaved → {LABELS_OUT}")


if __name__ == "__main__":
    main()
