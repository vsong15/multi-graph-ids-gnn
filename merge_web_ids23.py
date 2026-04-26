import pandas as pd
import os
from glob import glob

DATA_DIR = "web-ids23"
OUTPUT_FILE = "web_ids23_merged_clean.csv"

CSV_FILES = sorted(glob(os.path.join(DATA_DIR, "*.csv")))

dfs = []

for file in CSV_FILES:

    df = pd.read_csv(file, low_memory=False)

    df.columns = [c.strip() for c in df.columns]

    if "attack" not in df.columns:
        raise ValueError(f"Missing attack column in {file}")

    # Fix labels
    raw_attack = (
        df["attack"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["attack"] = raw_attack.map({
        "benign": 0,
        "attack": 1
    })

    if df["attack"].isna().any():
        bad_labels = raw_attack[df["attack"].isna()].unique()
        raise ValueError(f"Unknown attack labels in {file}: {bad_labels}")

    df["attack"] = df["attack"].astype(int)

    df["source_file"] = os.path.basename(file)

    if "ts" in df.columns:
        ts = pd.to_datetime(
            df["ts"],
            errors="coerce"
        )
        df["ts"] = ts.astype("string")
        df["ts_epoch"] = (
            ts.astype("int64") / 1_000_000_000
        )
        df.loc[ts.isna(), "ts_epoch"] = pd.NA

    dfs.append(df)

merged_df = pd.concat(
    dfs,
    ignore_index=True
)

merged_df = merged_df.drop_duplicates()

numeric_cols = merged_df.select_dtypes(
    include=["int64", "float64"]
).columns

merged_df[numeric_cols] = merged_df[
    numeric_cols
].apply(
    pd.to_numeric,
    errors="coerce"
)

merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)

object_cols = merged_df.select_dtypes(
    include=["object", "string"]
).columns

merged_df[object_cols] = merged_df[object_cols].fillna("unknown")

merged_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nMerged dataset saved:")
print(OUTPUT_FILE)

print("\nClass balance:\n")

counts = merged_df["attack"].value_counts()

for cls, count in counts.items():

    pct = 100 * count / len(merged_df)

    label = "Benign" if cls == 0 else "Attack"

    print(
        f"{label}: "
        f"{count:,} "
        f"({pct:.2f}%)"
    )
