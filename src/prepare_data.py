import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV = Path("data/raw/complaints.csv")
PROCESSED = Path("data/processed")
SEED = 42
THRESHOLD = 200  # min class size after sampling; sparse classes → "Other"
SAMPLE_N = 25_000
CR_CAP = 4_000  # max Credit Reporting rows before normalization
VAL_FRAC = 0.30   # first split: 70% train / 30% temp
TEST_FRAC = 0.50  # second split of temp: 50% val / 50% test → 15%/15% overall

COLS = ["Consumer complaint narrative", "Product", "Date received", "Company", "State"]

_REDACTION = re.compile(r"X{2,}")   # CFPB redaction markers are uppercase XXXX
_WHITESPACE = re.compile(r"\s+")


def clean_narrative(text: str) -> str:
    text = _REDACTION.sub("", text)   # remove before lowercasing to preserve lowercase xx
    text = text.lower()
    text = text.strip()
    text = _WHITESPACE.sub(" ", text)
    return text


def print_distribution(series: pd.Series, label: str) -> None:
    counts = series.value_counts()
    print(f"\n{label} ({len(counts)} classes, {len(series):,} rows):")
    for name, n in counts.items():
        print(f"  {name}: {n:,}")


def main(raw_csv: Path | None = None, output_dir: Path | None = None, full: bool = False) -> None:
    raw_csv = raw_csv or RAW_CSV
    processed = output_dir or PROCESSED
    processed.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load CSV in chunks, drop null narratives per-chunk ────────────
    print("Loading CSV …")
    chunks = []
    for chunk in pd.read_csv(raw_csv, usecols=COLS, chunksize=100_000):
        chunks.append(chunk.dropna(subset=["Consumer complaint narrative"]))
    df = pd.concat(chunks, ignore_index=True)
    print(f"After null-narrative filter: {len(df):,} rows x {len(df.columns)} columns")
    print_distribution(df["Product"], "Product distribution (full filtered dataset)")

    # ── Step 2: Subsample to ~25K ─────────────────────────────────────────────
    if not full:
        print(f"\nSampling: Credit Reporting capped at {CR_CAP:,}, others proportional …")

        _CR_VARIANTS = {
            "Credit reporting or other personal consumer reports",
            "Credit reporting, credit repair services, or other personal consumer reports",
            "Credit reporting",
        }
        cr_mask = df["Product"].isin(_CR_VARIANTS)
        df_cr = df[cr_mask].sample(n=CR_CAP, random_state=SEED)
        frac = SAMPLE_N / len(df)
        df_other = (
            df[~cr_mask]
            .groupby("Product", group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=SEED))
        )
        df = pd.concat([df_cr, df_other]).sample(frac=1, random_state=SEED).copy()
        print(f"\nSample size: {len(df):,} rows")
        print_distribution(df["Product"], "Product distribution AFTER sampling")
        df.to_csv(processed / "sample_25k.csv", index=False)
        print(f"Saved sample to {processed / 'sample_25k.csv'}")
    else:
        print("\n[--full] Skipping sampling — processing entire dataset.")

    # ── Step 3: Clean narrative text (subsample only) ─────────────────────────
    print("\nCleaning narrative text …")
    df["text"] = df["Consumer complaint narrative"].apply(clean_narrative)
    empty_after = (df["text"].str.strip() == "").sum()
    print(f"After cleaning: {len(df):,} rows")
    print(f"  Fully-redacted (empty after cleaning): {empty_after:,} — dropping")
    df = df[df["text"].str.strip() != ""].copy()
    print(f"After dropping empty-after-cleaning rows: {len(df):,} rows")

    # ── Step 3b: Deduplicate on text after cleaning ───────────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates(subset="text").copy()
    total_filtered = len(df)
    print(f"After deduplication: {total_filtered:,} rows (dropped {before_dedup - total_filtered:,} duplicates)")

    # ── Step 4: Normalize product label variants to canonical names ───────────
    _LABEL_NORMALIZATION = {
        "Credit reporting or other personal consumer reports": "Credit Reporting",
        "Credit reporting, credit repair services, or other personal consumer reports": "Credit Reporting",
        "Credit reporting": "Credit Reporting",
        "Credit card": "Credit Card",
        "Credit card or prepaid card": "Credit Card",
        "Prepaid card": "Credit Card",
        "Payday loan, title loan, personal loan, or advance loan": "Payday or Personal Loan",
        "Payday loan, title loan, or personal loan": "Payday or Personal Loan",
        "Payday loan": "Payday or Personal Loan",
        "Consumer Loan": "Payday or Personal Loan",
        "Money transfer, virtual currency, or money service": "Money Transfer",
        "Money transfers": "Money Transfer",
        "Virtual currency": "Money Transfer",
        "Checking or savings account": "Bank Account",
        "Bank account or service": "Bank Account",
    }
    df["Product"] = df["Product"].replace(_LABEL_NORMALIZATION)
    print("\nApplied label normalization.")
    print_distribution(df["Product"], "Product distribution AFTER normalization")

    # ── Step 5: Consolidate sparse labels → "Other" ───────────────────────────
    print(f"\nConsolidating labels (threshold = {THRESHOLD:,}) …")
    counts = df["Product"].value_counts()
    small = set(counts[counts < THRESHOLD].index)
    if small:
        print(f"  Merging into 'Other': {sorted(small)}")
    df["label_name"] = df["Product"].apply(lambda x: "Other" if x in small else x)
    print_distribution(df["label_name"], "Product distribution AFTER consolidation")

    # ── Step 6: Label encoder → label_map.json ────────────────────────────────
    sorted_names = sorted(df["label_name"].unique())
    label_map = {name: idx for idx, name in enumerate(sorted_names)}
    map_path = processed / "label_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"\nSaved label_map.json → {len(label_map)} classes: {map_path}")
    df["label"] = df["label_name"].map(label_map).astype(int)

    # ── Step 7: Stratified 70 / 15 / 15 split ────────────────────────────────
    print("\nSplitting data (70 / 15 / 15, stratified) …")
    cols = ["text", "label", "label_name"]
    # Preserve original .index — train_test_split keeps it; needed for overlap check
    train, temp = train_test_split(
        df[cols], test_size=VAL_FRAC, random_state=SEED, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, test_size=TEST_FRAC, random_state=SEED, stratify=temp["label"]
    )
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # ── Step 8: Save splits ───────────────────────────────────────────────────
    print("\nSaving splits …")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        out = processed / f"{name}.csv"
        split.to_csv(out, index=False)
        print(f"  Saved {name}.csv: {len(split):,} rows → {out}")

    # ── Step 9: Final shapes and class distribution ───────────────────────────
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print_distribution(split["label_name"], f"{name.upper()} class distribution")

    # ── Step 10: Validation ───────────────────────────────────────────────────
    print("\n─── Validation ───────────────────────────────────────────────────")

    total = len(train) + len(val) + len(test)
    assert total == total_filtered, f"Row sum mismatch: {total} != {total_filtered}"
    print(f"Train size: {len(train):,} — OK")
    print(f"Val size:   {len(val):,} — OK")
    print(f"Test size:  {len(test):,} — OK")
    print(f"Total row count ({total:,} == {total_filtered:,}) — OK")

    for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        n_dupes = split_df["text"].duplicated().sum()
        assert n_dupes == 0, f"{split_name} has {n_dupes:,} duplicate text rows"
        print(f"No duplicate text in {split_name} — OK")

    assert not (set(train.index) & set(val.index)), "Index overlap: train ∩ val"
    assert not (set(train.index) & set(test.index)), "Index overlap: train ∩ test"
    assert not (set(val.index) & set(test.index)), "Index overlap: val ∩ test"
    print("No index overlap between splits — OK")

    all_classes = set(train["label"].unique())
    for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        missing = all_classes - set(split_df["label"].unique())
        assert not missing, f"{split_name} missing classes: {missing}"
        print(f"All {len(all_classes)} classes present in {split_name} — OK")

    for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        n_null = split_df["text"].isna().sum()
        assert n_null == 0, f"{split_name} has {n_null:,} null text values"
        print(f"Zero null text in {split_name} — OK")

    for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        n_empty = (split_df["text"].str.strip() == "").sum()
        assert n_empty == 0, f"{split_name} has {n_empty:,} empty-after-cleaning text rows"
        print(f"Empty-text rows: {n_empty} — OK")

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", default=False,
                        help="Process the entire dataset without sampling.")
    args = parser.parse_args()
    main(full=args.full)
