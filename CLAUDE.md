# CFPB Complaint Classifier

## Project
Multiclass text classifier on CFPB consumer complaint narratives.
Fine-tuned transformer (DistilBERT or RoBERTa) predicting complaint
product category from free-text narrative.

## Stack
Python 3.11, pandas, scikit-learn, transformers, torch, datasets

## Data
- Raw CSV: data/raw/complaints.csv
- Key columns:
  - `Consumer complaint narrative` — free text input (may be null)
  - `Product` — target label (15+ categories, needs consolidation)
  - `Date received`, `Company`, `State` — metadata, keep but don't train on
  - `Issue`, `Sub-issue`, `Sub-product` - metadata that if used would constitute data-leak, do not use for training under any circumstances
- Processed outputs go in data/processed/

## File structure
- src/prepare_data.py — cleaning, label mapping, train/val/test split
- src/train.py — fine-tuning script
- src/predict.py — inference wrapper
- src/eval.py — per-class F1, confusion matrix

## Rules
- NEVER write to data/raw/ — treat it as read-only
- NEVER touch data/processed/test.csv after it's created
- Always filter out rows where narrative is null before any processing
- Use class-weighted loss (class imbalance is significant)
- Confidence threshold for low-confidence flagging: 0.65 (can consider producing this number through cross-validation in future)
- Random seed: 42 everywhere for reproducibility
- Print shape and class distribution after every major transformation