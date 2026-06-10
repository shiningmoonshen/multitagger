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


def main() -> None:
    # ── Step 1: Load CSV ──────────────────────────────────────────────────────
    print("Loading CSV …")
    header_df = pd.read_csv(RAW_CSV, nrows=0)
    all_cols = header_df.columns.tolist()
    df = pd.read_csv(
        RAW_CSV,
        low_memory=False,
        usecols=["Product", "Consumer complaint narrative"],
    )
    print(f"Shape: {len(df):,} rows x {len(all_cols)} columns")
    print(f"Columns: {all_cols}")

    # ── Step 2: Filter null / blank narratives ────────────────────────────────
    print("\nFiltering null narratives …")
    mask = df["Consumer complaint narrative"].notna() & \
           df["Consumer complaint narrative"].str.strip().ne("")
    df = df[mask].copy()
    total_filtered = len(df)
    print(f"After narrative filter: {total_filtered:,} rows")

    # ── Step 2b: Sample — cap Credit Reporting at CR_CAP, others proportional ─
    print(f"\nSampling: Credit Reporting capped at {CR_CAP:,}, others proportional …")
    print_distribution(df["Product"], "Product distribution BEFORE sampling")

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
    total_filtered = len(df)
    print(f"\nSample size: {total_filtered:,} rows")
    print_distribution(df["Product"], "Product distribution AFTER sampling")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "sample_25k.csv", index=False)
    print(f"Saved sample to {PROCESSED / 'sample_25k.csv'}")

    # ── Step 3b: Normalize product label variants to canonical names ─────────
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

    # ── Step 4: Consolidate sparse labels → "Other" ───────────────────────────
    print(f"\nConsolidating labels (threshold = {THRESHOLD:,}) …")
    counts = df["Product"].value_counts()
    small = set(counts[counts < THRESHOLD].index)
    if small:
        print(f"  Merging into 'Other': {sorted(small)}")
    df["label_name"] = df["Product"].apply(lambda x: "Other" if x in small else x)
    print_distribution(df["label_name"], "Product distribution AFTER consolidation")

    # ── Step 5: Label encoder → label_map.json ────────────────────────────────
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sorted_names = sorted(df["label_name"].unique())
    label_map = {name: idx for idx, name in enumerate(sorted_names)}
    map_path = PROCESSED / "label_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"\nSaved label_map.json → {len(label_map)} classes: {map_path}")
    df["label"] = df["label_name"].map(label_map).astype(int)

    # ── Step 6: Clean narrative text ──────────────────────────────────────────
    print("\nCleaning narrative text …")
    df["text"] = df["Consumer complaint narrative"].apply(clean_narrative)
    empty_after = (df["text"] == "").sum()
    print(f"After cleaning: {len(df):,} rows")
    print(f"  Fully-redacted (empty after cleaning): {empty_after:,}")

    # ── Step 6b: Deduplicate on text after cleaning ───────────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates(subset="text").copy()
    total_filtered = len(df)  # update total for validation
    print(f"After deduplication: {total_filtered:,} rows (dropped {before_dedup - total_filtered:,} duplicates)")

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
        out = PROCESSED / f"{name}.csv"
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

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
