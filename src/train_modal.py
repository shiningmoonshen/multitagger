import sys
from pathlib import Path

import modal

app = modal.App("cfpb-train")


def _download_base_model():
    import os
    from transformers import AutoModel, AutoTokenizer

    dest = "/models/base/distilbert-base-uncased"
    os.makedirs(dest, exist_ok=True)
    AutoTokenizer.from_pretrained("distilbert-base-uncased").save_pretrained(dest)
    AutoModel.from_pretrained("distilbert-base-uncased").save_pretrained(dest)


train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.48.2",
        "datasets==3.2.0",
        "accelerate==1.2.1",
        "pandas==2.2.3",
        "scikit-learn==1.6.0",
        "numpy==1.26.4",
        "wandb",
    )
    .run_function(_download_base_model)
    .add_local_file(
        Path(__file__).parent / "train.py",
        remote_path="/root/train.py",
    )
)


@app.function(
    image=train_image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    gpu="T4",
    memory=16384,
    timeout=7200,
)
def run_training(resume_from: str | None = None, batch_size: int = 64, max_length: int = 256):
    sys.path.insert(0, "/root")
    import train
    import wandb

    api = wandb.Api()

    if resume_from:
        checkpoint_artifact = api.artifact(resume_from)
        base_model_dir = checkpoint_artifact.download()
    else:
        base_model_dir = "/models/base/distilbert-base-uncased"

    data_artifact = api.artifact("multitagger/cfpb-complaints-processed:latest")
    data_dir = Path(data_artifact.download())

    train.main(
        processed_dir=data_dir,
        model_dir=None,
        base_model_dir=base_model_dir,
        smoke_test=False,
        processed_artifact="multitagger/cfpb-complaints-processed:latest",
        batch_size=batch_size,
        max_length=max_length,
    )


@app.local_entrypoint()
def main(
    resume_from: str = "",
    batch_size: int = 64,
    max_length: int = 256,
):
    run_training.remote(
        resume_from=resume_from or None,
        batch_size=batch_size,
        max_length=max_length,
    )
