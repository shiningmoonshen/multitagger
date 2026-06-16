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