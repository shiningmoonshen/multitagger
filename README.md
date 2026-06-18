# CFPB Complaint Classifier

Multiclass text classifier on CFPB consumer complaint narratives. Fine-tunes DistilBERT to predict the complaint product category from free-text input.

**Stack:** Python 3.11, HuggingFace Transformers, PyTorch, scikit-learn, pandas

---

## Docker

All commands assume you are in the project root.

### Build the image

```bash
docker build -t multitagger .
```

### Run training

```bash
docker compose run --rm trainer
```

This runs `src/train.py` inside the container. Processed data is read from `./data/processed/` via the volume mount.

> **Note:** `train.py` loads the base model from `models/base/distilbert-base-uncased/` with
> `local_files_only=True`. You must make that directory available inside the container. Either add a
> second volume mount to `docker-compose.yml`:
>
> ```yaml
> - ./models:/app/models
> ```
>
> …or download the weights first by running:
>
> ```bash
> docker compose run --rm trainer python src/download_base_model.py
> ```

### Run a one-off prediction

```bash
docker compose run --rm trainer python src/predict.py
```

### Open a shell for debugging

```bash
docker compose run --rm trainer bash
```
