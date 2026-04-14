import pandas as pd
import os
from glob import glob

DATA_DIR = "web-ids23"
OUTPUT_FILE = "web_ids23_merged_clean.csv"

CSV_FILES = glob(os.path.join(DATA_DIR, "*.csv"))

dfs = []

for file in CSV_FILES:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    if "attack_type" not in df.columns:
        df["attack_type"] = "unknown"

    if "attack" not in df.columns:
        df["attack"] = 0

    df["source_file"] = os.path.basename(file)

    if "ts" in df.columns:
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")

    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)
merged_df = merged_df.drop_duplicates()
merged_df = merged_df.fillna(0)

numeric_cols = merged_df.select_dtypes(include=["int64", "float64"]).columns
merged_df[numeric_cols] = merged_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
merged_df = merged_df.fillna(0)

merged_df.to_csv(OUTPUT_FILE, index=False)
