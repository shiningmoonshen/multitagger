import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))
from predict import predict_batch  # noqa: E402

PROCESSED = Path("data/processed")
LABEL_MAP_PATH = PROCESSED / "label_map.json"
REPORTS = Path("reports")
BATCH_SIZE = 64


def print_distribution(series: pd.Series, label: str) -> None:
    counts = series.value_counts()
    print(f"\n{label} ({len(counts)} classes, {len(series):,} rows):")
    for name, n in counts.items():
        print(f"  {name}: {n:,}")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load test set ─────────────────────────────────────────────────
    print("Loading test set …")
    df = pd.read_csv(PROCESSED / "test.csv")
    print(f"Test shape: {df.shape}")
    print_distribution(df["label_name"], "Test class distribution")

    # ── Step 2: Run inference in batches ──────────────────────────────────────
    print("\nRunning inference …")
    texts = df["text"].tolist()
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        results.extend(predict_batch(texts[i : i + BATCH_SIZE]))
        print(f"  Scored {min(i + BATCH_SIZE, len(texts))}/{len(texts)} …", end="\r")
    print()

    pred_tags = [r["tag"] for r in results]
    low_conf_flags = [r["low_confidence"] for r in results]
    true_tags = df["label_name"].tolist()

    # ── Step 3: Classification report ─────────────────────────────────────────
    print("\n=== Classification Report (Test Set — Final) ===")
    print(classification_report(true_tags, pred_tags, zero_division=0))
    report_dict = classification_report(true_tags, pred_tags, output_dict=True, zero_division=0)

    # ── Step 4: Confusion matrix PNG ──────────────────────────────────────────
    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        label_map: dict[str, int] = json.load(f)
    ordered_labels = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    cm = confusion_matrix(true_tags, pred_tags, labels=ordered_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=ordered_labels,
        yticklabels=ordered_labels,
        ax=ax,
        cmap="Blues",
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — Test Set")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = REPORTS / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved to {cm_path}")

    # ── Step 5: Low-confidence summary ────────────────────────────────────────
    n_low = sum(low_conf_flags)
    pct_low = n_low / len(results) * 100
    print(f"\nLow-confidence predictions: {n_low} ({pct_low:.1f}%)")

    low_conf_per_class: dict[str, int] = {}
    for true_tag, is_low in zip(true_tags, low_conf_flags):
        if is_low:
            low_conf_per_class[true_tag] = low_conf_per_class.get(true_tag, 0) + 1
    if low_conf_per_class:
        print("  Per true class:")
        for cls, cnt in sorted(low_conf_per_class.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {cnt}")

    # ── Step 6: Save eval_results.json ────────────────────────────────────────
    per_class = {
        k: v
        for k, v in report_dict.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }
    eval_results = {
        "test_size": len(df),
        "accuracy": round(report_dict["accuracy"], 4),
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
        "per_class": {
            cls: {k: round(v, 4) for k, v in metrics.items()}
            for cls, metrics in per_class.items()
        },
        "low_confidence": {
            "total": n_low,
            "pct": round(pct_low, 2),
            "per_class": low_conf_per_class,
        },
    }
    results_path = REPORTS / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {results_path}")

    # ── Step 7: Final summary ─────────────────────────────────────────────────
    print(
        f"\n=== Eval Complete ===\n"
        f"Test examples:     {len(df):,}\n"
        f"Macro F1:          {eval_results['macro_f1']:.4f}\n"
        f"Weighted F1:       {eval_results['weighted_f1']:.4f}\n"
        f"Accuracy:          {eval_results['accuracy']:.4f}\n"
        f"Low-confidence:    {n_low} ({pct_low:.1f}%)\n"
        f"Results saved to:  {results_path}\n"
        f"Confusion matrix:  {cm_path}"
    )


if __name__ == "__main__":
    main()
