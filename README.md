# CFPB Complaint Classifier

A fine-tuned DistilBERT model that classifies Consumer Financial Protection Bureau (CFPB) consumer complaint narratives into 10 product categories (Bank Account, Credit Card, Credit Reporting, Debt collection, Money Transfer, Mortgage, Other, Payday or Personal Loan, Student loan, Vehicle loan or lease). Given a free-text complaint narrative, the model outputs a predicted category and a confidence score; predictions below 0.65 confidence are flagged for human review via a Streamlit interface.

## Architecture

| Stage | Scripts | What it does |
|---|---|---|
| Data prep | `prepare_data.py` / `prepare_modal.py` | Cleans narratives, consolidates labels, stratified 70/15/15 split |
| Training | `train.py` / `train_modal.py` | Fine-tunes DistilBERT with class-weighted loss; logs to W&B |
| Inference | `predict.py` | Single and batch predictions with confidence scores |
| Review tool | `src/app.py` | Streamlit UI: predict, review low-confidence queue, submit overrides |

## Setup

```bash
pip install -r requirements.txt
export WANDB_API_KEY=<your key>
```

`requirements.txt` uses a CPU PyTorch wheel for local use. Modal training runs use GPU automatically.

## How to run

### Full pipeline on Modal (GPU)

```bash
# One-time: upload raw data to W&B artifacts
# WARNING: re-running this creates spurious duplicate artifact versions in W&B — only run once
python scripts/upload_raw_data.py

# Data preparation
modal run src/prepare_modal.py           # default: stratified ~25k sample
modal run src/prepare_modal.py --full    # full dataset

# Training
modal run src/train_modal.py
modal run src/train_modal.py --resume-from <artifact>        # resume from W&B checkpoint
modal run src/train_modal.py --batch-size 64 --max-length 256   # override GPU defaults
```

### Local inference

```bash
python src/predict.py
```

Runs 5 example predictions. Each result contains `tag`, `confidence`, `low_confidence`, and `top_2`.

### Streamlit review tool

Run from the project root (the app reads `data/processed/label_map.json` via relative path):

```bash
streamlit run src/app.py
```

### Docker

```bash
docker compose up --build
```

Builds the image and runs `python src/train.py`. `data/` and `models/` are mounted as volumes so outputs persist on the host.

## Eval results

Evaluated on the held-out test set (1,700 examples). Model trained on a stratified sample from the ~47k-complaint CFPB dataset.

| Metric | Score |
|---|---|
| Weighted F1 | 0.769 |
| Macro F1 | 0.646 |
| Accuracy | 0.762 |
| Low-confidence (<0.65) | 23.1% (393 / 1,700) |

Per-class results are in `reports/eval_results.json`. Confusion matrix: `reports/confusion_matrix.png`.

## Known limitations

- **"Payday or Personal Loan"** is the most frequently misclassified category (F1 0.31, precision 0.24). The model confuses these with Credit Card and Debt collection complaints, which use similar financial distress language.
- **"Other"** is a sparse catch-all with F1 0.00 on the test set (5 examples). Complaints predicted as "Other" should be treated as unclassified.
- **23% of predictions fall below the 0.65 confidence threshold.** These are surfaced in the Streamlit low-confidence queue for manual review.
- The model was trained on a stratified subset of the available data. Rare categories (Vehicle loan, Payday loan) have limited training examples and correspondingly weaker performance.
