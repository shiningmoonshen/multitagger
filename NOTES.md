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