# CFPB Complaint Classifier — Eval Summary

## What it does
Classifies free-text CFPB consumer complaint narratives into one of 10 consolidated
product categories (e.g. "Credit Reporting", "Mortgage", "Debt collection").

## Training setup
- Model: DistilBERT (`distilbert-base-uncased`), fine-tuned for sequence classification
- Data: 25,000 rows stratified-sampled from the CFPB complaints dataset; Credit
  Reporting capped at 4,000 rows to reduce class dominance
- Preprocessing: lowercase, strip XXXX redaction markers, normalize whitespace
- Max token length: 128
- Loss: cross-entropy with class weights (inverse-frequency) to handle imbalance
- Split: 70% train / 15% val / 15% test, stratified, seed 42
- Training: 3 epochs, batch size 8, best checkpoint by eval loss

## Test set results (held-out, 1,700 examples)

| Metric        | Score |
|---------------|-------|
| Accuracy      | 0.762 |
| Macro F1      | 0.646 |
| Weighted F1   | 0.769 |
| Low-confidence (< 0.65) | 393 / 1700 (23.1%) |

### Per-class F1

| Class                   | Precision | Recall | F1   | Support |
|-------------------------|-----------|--------|------|---------|
| Student loan            | 0.82      | 0.83   | 0.83 | 60      |
| Mortgage                | 0.92      | 0.75   | 0.82 | 142     |
| Credit Reporting        | 0.89      | 0.77   | 0.82 | 497     |
| Bank Account            | 0.83      | 0.72   | 0.77 | 188     |
| Debt collection         | 0.76      | 0.78   | 0.77 | 381     |
| Money Transfer          | 0.65      | 0.85   | 0.74 | 99      |
| Credit Card             | 0.68      | 0.80   | 0.73 | 233     |
| Vehicle loan or lease   | 0.65      | 0.69   | 0.67 | 51      |
| Payday or Personal Loan | 0.24      | 0.43   | 0.31 | 44      |
| Other                   | 0.00      | 0.00   | 0.00 | 5       |

## Known failure modes

**"Other" (F1 = 0.00):** Only 5 test examples. The model never predicts this
class — there were too few training examples for it to learn anything useful.
If "Other" predictions matter, the label consolidation threshold should be
raised or this class dropped entirely.

**"Payday or Personal Loan" (F1 = 0.31):** Weakest substantive class in both
val (F1 = 0.39) and test. Low support (44 test examples) combined with narrative
language that overlaps heavily with "Debt collection" and "Bank Account".

**Credit Reporting / Debt Collection confusion:** These two are the most
frequently confused pair. Narratives about collection accounts on credit reports
straddle both categories linguistically — confirmed in the predict.py smoke test
and visible in the confusion matrix.

**Low-confidence rate (23.1%):** Nearly 1 in 4 predictions falls below the 0.65
threshold, concentrated in Credit Reporting (127) and Debt collection (93). These
are the largest classes and share a lot of vocabulary.

## Artifacts
- `reports/eval_results.json` — machine-readable metrics for downstream use
- `reports/confusion_matrix.png` — full 10×10 confusion matrix heatmap
