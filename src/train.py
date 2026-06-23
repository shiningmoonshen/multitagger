import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Issue, Sub-issue, Sub-product are not in the processed CSVs and must never be used for training.

PROCESSED = Path("data/processed")
MODEL_DIR = Path("models/distilbert-v1")
FINAL_DIR = MODEL_DIR / "final"
# All model loads use local_files_only=True — no data or
# requests ever leave this machine. Fails loudly if weights missing.
BASE_MODEL_DIR = "models/base/distilbert-base-uncased"
MAX_LENGTH = 128
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _print_distribution(series: pd.Series, label: str) -> None:
    counts = series.value_counts()
    print(f"\n{label} ({len(counts)} classes, {len(series):,} rows):")
    for name, n in counts.items():
        print(f"  {name}: {n:,}")


def load_data(processed: Path):
    print("=" * 60)
    print("Step 1: Loading processed splits")
    print("=" * 60)
    train_df = pd.read_csv(processed / "train.csv")
    val_df = pd.read_csv(processed / "val.csv")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    _print_distribution(train_df["label_name"], "TRAIN class distribution")
    _print_distribution(val_df["label_name"], "VAL class distribution")

    # Drop empty/null text rows, which cause tokenization errors. Should be very rare after processing, but just in case... log a warning if any are dropped.
    for df, path in [(train_df, "train.csv"), (val_df, "val.csv")]:
        n_before = len(df)
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip() != ""]
        df["text"] = df["text"].astype(str)
        dropped = n_before - len(df)
        if dropped:
            print(f"WARNING: dropped {dropped} empty/null text rows from {path}")

    with open(processed / "label_map.json") as f:
        label_map = json.load(f)
    num_labels = len(label_map)
    print(f"\nnum_labels: {num_labels}")

    return train_df, val_df, label_map, num_labels


def build_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer, max_length: int):
    print("\n" + "=" * 60)
    print("Step 2: Building HuggingFace Datasets and tokenizing")
    print("=" * 60)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)

    train_ds = train_ds.map(tokenize, batched=True, desc="Tokenizing train")
    val_ds = val_ds.map(tokenize, batched=True, desc="Tokenizing val")

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    print(f"Train dataset: {len(train_ds):,} examples")
    print(f"Val dataset:   {len(val_ds):,} examples")
    return train_ds, val_ds


def load_model(num_labels: int, base_model_dir: str):
    print("\n" + "=" * 60)
    print("Step 3: Loading model")
    print("=" * 60)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_dir, num_labels=num_labels, local_files_only=True
        )
    except OSError:
        raise OSError(
            f"Base model not found locally at {base_model_dir}. "
            "Run scripts/download_base_model.py once."
        )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return model


def compute_class_weights(train_df: pd.DataFrame, num_labels: int) -> torch.Tensor:
    print("\n" + "=" * 60)
    print("Step 4: Computing class weights")
    print("=" * 60)
    # Class weights over the FULL label space (0..num_labels-1), not just
    # classes present in the (possibly subsampled) training data.
    counts = np.bincount(train_df["label"].values, minlength=num_labels)
    total = counts.sum()
    # "balanced" formula: total / (num_classes * count); guard against
    # zero-count classes (e.g. in smoke tests) by clamping to 1.
    weights = total / (num_labels * np.clip(counts, 1, None))
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"Class weights: {np.round(weights, 3)}")
    return class_weights


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def make_compute_metrics(id2label: dict, num_labels: int):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        per_class = f1_score(
            labels, preds, average=None,
            labels=list(range(num_labels)), zero_division=0,
        )
        if wandb.run is not None:
            wandb.log({
                "val/f1_per_class": {
                    id2label[i]: float(per_class[i]) for i in range(num_labels)
                }
            })
        return {
            "f1_macro": macro_f1,
            "f1_weighted": weighted_f1,
            "accuracy": accuracy_score(labels, preds),
        }
    return compute_metrics


class WandbMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if wandb.run is None or not logs:
            return
        metrics = {}
        if "loss" in logs:
            metrics["train/loss"] = logs["loss"]
        if "eval_loss" in logs:
            metrics["val/loss"] = logs["eval_loss"]
        if "eval_f1_macro" in logs:
            metrics["val/macro_f1"] = logs["eval_f1_macro"]
        if "eval_f1_weighted" in logs:
            metrics["val/weighted_f1"] = logs["eval_f1_weighted"]
        if "epoch" in logs:
            metrics["epoch"] = logs["epoch"]
        if metrics:
            wandb.log(metrics)


def main(
    processed_dir: Path | None = None,
    model_dir: Path | None = None,
    base_model_dir: str | None = None,
    smoke_test: bool | None = None,
    processed_artifact: str | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
    num_epochs: int | None = None,
):
    if not os.environ.get("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY environment variable not set. Exiting.")
        sys.exit(1)

    if smoke_test is None or batch_size is None or max_length is None or num_epochs is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--smoke-test", action="store_true",
                            help="Quick sanity check: 200/100 rows, 20 steps.")
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--max-length", type=int, default=128)
        parser.add_argument("--num-epochs", type=int, default=3)
        args = parser.parse_args()
        if smoke_test is None:
            smoke_test = args.smoke_test
        if batch_size is None:
            batch_size = args.batch_size
        if max_length is None:
            max_length = args.max_length
        if num_epochs is None:
            num_epochs = args.num_epochs

    processed = processed_dir or PROCESSED
    base_model = base_model_dir or BASE_MODEL_DIR

    train_df, val_df, label_map, num_labels = load_data(processed)
    id2label = {v: k for k, v in label_map.items()}
    target_names = [id2label[i] for i in range(num_labels)]

    if smoke_test:
        print("\n[smoke-test] Subsampling: 200 train / 100 val, max_steps=20")
        train_df = train_df.sample(200, random_state=SEED)
        val_df = val_df.sample(100, random_state=SEED)

    run_name = f"distilbert-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project="multitagger",
        name=run_name,
        tags=["distilbert", "multiclass"],
        config={
            "model_checkpoint": base_model,
            "max_length": max_length,
            "seed": SEED,
            "learning_rate": 2e-5,
            "batch_size_train": batch_size,
            "batch_size_eval": batch_size,
            "num_train_epochs": num_epochs,
            "notes": str(processed / "train.csv"),
        },
    )

    if processed_artifact:
        wandb.run.use_artifact(processed_artifact)

    if model_dir is None:
        model_dir = Path("models") / f"distilbert-{wandb.run.id}"
    final_dir = model_dir / "final"

    class_weights = compute_class_weights(train_df, num_labels)

    print("\n" + "=" * 60)
    print("Loading tokenizer")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)

    train_ds, val_ds = build_datasets(train_df, val_df, tokenizer, max_length)
    data_collator = DataCollatorWithPadding(tokenizer)

    model = load_model(num_labels, base_model)

    print("\n" + "=" * 60)
    print("Step 5-7: Configuring training")
    print("=" * 60)
    model_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=SEED,
        report_to="none",
    )

    if smoke_test:
        training_args.max_steps = 20
        training_args.output_dir = "models/smoke-test"

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id2label, num_labels),
        callbacks=[WandbMetricsCallback()],
    )

    print("\n" + "=" * 60)
    print("Step 8: Training")
    print("=" * 60)
    trainer.train()

    eval_logs = [l for l in trainer.state.log_history if "eval_loss" in l]
    best_log = min(eval_logs, key=lambda x: x["eval_loss"])
    print(
        f"\n=== Training Summary ===\n"
        f"Best epoch:     {best_log.get('epoch')}\n"
        f"Best eval loss: {best_log.get('eval_loss', float('nan')):.4f}\n"
        f"Val macro F1:   {best_log.get('eval_f1_macro', float('nan')):.4f}"
    )

    print("\n" + "=" * 60)
    print("Step 9: Saving model and tokenizer")
    print("=" * 60)
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved to {final_dir}")

    if wandb.run is not None:
        artifact = wandb.Artifact("multitagger-distilbert", type="model")
        artifact.add_dir(str(final_dir))
        wandb.log_artifact(artifact)

    print("\n" + "=" * 60)
    print("Step 10: Standalone evaluation on val set")
    print("=" * 60)
    pred_output = trainer.predict(val_ds)
    preds = np.argmax(pred_output.predictions, axis=-1)
    true_labels = val_df["label"].values

    print("\n=== Classification Report (Val Set — Session 2 Baseline) ===")
    print(classification_report(true_labels, preds, target_names=target_names))

    if wandb.run is not None:
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds=preds.tolist(),
                y_true=true_labels.tolist(),
                class_names=target_names,
            )
        })

    wandb.finish()


if __name__ == "__main__":
    main()
