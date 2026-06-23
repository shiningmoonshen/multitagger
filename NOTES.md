## Session 6 — Streamlit Wiring, Modal Integration, End-to-End Test — [06/22/2026]

### Done
- Revised Sprint 2 Day 3 brief: introduced `data/predictions.csv` as a
  full inference log to provide denominator for override rate and
  low-confidence count; `data/overrides.csv` confirmed as corrections-only
  (no empty rows)
- Sprint 2 Day 3 implemented: summary stats panel, confidence filter slider,
  low-confidence queue tab; `overridden` column derived by joining predictions
  to overrides on narrative text
- Designed Modal integration as Sprint 2 Day 4: `prepare_modal.py` (CPU,
  16GB RAM) and `train_modal.py` (T4 GPU) with W&B Artifacts as data
  persistence layer for full lineage: `cfpb-complaints-raw` →
  `cfpb-complaints-processed` → model artifact
- Added `--full` flag to `prepare_data.py` and `prepare_modal.py` to toggle
  between ~47k sample and full dataset
- Added `--resume-from` CLI flag to `train_modal.py` to optionally start
  from a prior fine-tuned W&B artifact instead of baked base weights
- Fixed `modal.Mount` AttributeError — replaced with current Modal API for
  local file access
- Fixed classifier head size mismatch — `_download_base_model()` changed to
  use `AutoModel` (not `AutoModelForSequenceClassification`) so baked weights
  are backbone-only; classifier head initialized fresh by `train.py` with
  correct `num_labels` at runtime
- Increased Modal training parameters for GPU: batch size 16 → 64,
  max_length 128 → 256
- Decided against full dataset training run — 47k sample baseline (0.769
  weighted F1) is sufficient for portfolio/interview purposes; all target
  tools and pipeline stages are integrated
- Sprint 2 Day 5 completed: end-to-end test passed, README written
- `scripts/upload_raw_data.py` is a one-time script — re-running creates
  spurious artifact versions in W&B; noted in README

### Eval results
No new eval run. Baseline remains 0.769 weighted F1 from Sprint 1 sample run.
Full dataset training deferred — not required for project goals.

### Next session
- Sprint 3 options: Airflow DAG for scheduled retraining, Evidently drift
  monitoring, closing the feedback loop from overrides back into training data
- Interview prep when ready

## Session 5 — W&B Integration — [06/18/2026]

### Done
- Added wandb dependency to requirements.txt
- Instrumented train.py with W&B — wandb.init(), per-epoch metric logging,
  per-class F1 logging with readable label names, confusion matrix, and
  model artifact; wandb.finish() at close
- WANDB_API_KEY handled via environment variable; script exits with warning
  if key is not set
- Training run completed and verified in W&B dashboard — all metrics
  populated, artifact visible in Artifacts tab
- F1 unchanged from Sprint 1 baseline, confirming instrumentation is
  purely observational

### Eval results
No new eval run — F1 parity with Sprint 1 baseline confirmed via W&B dashboard.
See Session 3 eval results.

### Next session
- Sprint 2, Day 3 — refer to sprint plan (Streamlit review tool)

## Session 4 - Docker Setup - [06/17/2026]

### Done
- Download and installed Docker Desktop
- Setup Docker, initiated build under 'multitagger'

### Docker training run results:
=== Training Summary ===
Best epoch:     1.0
Best eval loss: 0.9656
Val macro F1:   0.6448

=== Classification Report (Val Set — Session 2 Baseline) ===
                         precision    recall  f1-score   support

           Bank Account       0.69      0.76      0.72       189
            Credit Card       0.74      0.77      0.76       233
       Credit Reporting       0.89      0.76      0.82       496
        Debt collection       0.77      0.76      0.77       381
         Money Transfer       0.69      0.76      0.72        99
               Mortgage       0.85      0.83      0.84       142
                  Other       0.00      0.00      0.00         5
Payday or Personal Loan       0.25      0.55      0.34        44
           Student loan       0.85      0.87      0.86        61
  Vehicle loan or lease       0.66      0.58      0.62        50

               accuracy                           0.76      1700
              macro avg       0.64      0.66      0.64      1700
           weighted avg       0.78      0.76      0.77      1700


## Session 3 — Inference Wrapper + Eval Report — [06/16/2026]

### Done
- Skipped improve-and-iterate (Wednesday) — first-pass macro F1 0.65
  already exceeded 0.75 weighted F1 target
- predict.py written and validated — smoke test passed with no errors
- eval.py written and validated — results look promising
- reports/confusion_matrix.png saved
- reports/eval_results.json saved
- reports/eval_summary.md written

### Eval results
- refer to reports/eval_summary.md:
- for summary:
#### Test set results (held-out, 1,700 examples)
| Metric        | Score |
|---------------|-------|
| Accuracy      | 0.762 |
| Macro F1      | 0.646 |
| Weighted F1   | 0.769 |
| Low-confidence (< 0.65) | 393 / 1700 (23.1%) |


### Next session
- Week 2 Sprint — refer to sprint plan
- First task: Dockerize (Monday) — write Dockerfile, verify container
  reproduces same F1 scores as local run

## Session 2 - Training - [06/10/2026]

### Done
- train.py written and validated
- base model downloaded
- base model finetuned with 25k stratified sample, using train and val splits
- baseline saved to models/distilbert-v1

### Eval results
=== Training Summary ===
Best epoch:     1.0
Best eval loss: 0.9522
Val macro F1:   0.6478

Step 10: Standalone evaluation on val set

/opt/homebrew/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
100%|███████████████████████████████████████████████| 107/107 [00:16<00:00,  6.39it/s]

=== Classification Report (Val Set — Session 2 Baseline) ===
                         precision    recall  f1-score   support

           Bank Account       0.75      0.63      0.69       189
            Credit Card       0.70      0.81      0.75       233
       Credit Reporting       0.88      0.79      0.83       496
        Debt collection       0.77      0.78      0.77       381
         Money Transfer       0.63      0.74      0.68        99
               Mortgage       0.90      0.80      0.84       142
                  Other       0.00      0.00      0.00         5
Payday or Personal Loan       0.30      0.55      0.39        44
           Student loan       0.85      0.87      0.86        61
  Vehicle loan or lease       0.62      0.72      0.67        50

               accuracy                           0.76      1700
              macro avg       0.64      0.67      0.65      1700
           weighted avg       0.78      0.76      0.77      1700

### Next session
- refer to sprint plan

## Session 1 — Data Prep — [06/09/2026]

### Done
- prepare_data.py written and validated
- 25k stratified sample saved to data/processed/sample_25k.csv
- train/val/test splits saved to data/processed/ (using sample_25k.csv)
- all 10 assertions passing
- text cleaning working (XXXX stripped, // and {$} are known 
  redaction artifacts — acceptable)

### Outstanding before training
- None, all merges fixed and ran

### Next session
- Write src/train.py (DistilBERT, max_length=128, batch_size=8, 
  class-weighted loss)
- refer to Notion page for start of task brief for train.py