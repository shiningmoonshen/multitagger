import sys
import tempfile
from pathlib import Path

import modal

app = modal.App("cfpb-prepare")

prepare_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.3",
        "scikit-learn==1.6.0",
        "numpy==1.26.4",
        "wandb",
    )
    .add_local_file(
        Path(__file__).parent / "prepare_data.py",
        remote_path="/root/prepare_data.py",
    )
)


@app.function(
    image=prepare_image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    cpu=4,
    memory=16384,
    timeout=3600,
)
def run_prepare(full: bool = False):
    sys.path.insert(0, "/root")
    import prepare_data
    import wandb

    run = wandb.init(project="multitagger", job_type="prepare")
    artifact = run.use_artifact("cfpb-complaints-raw:latest")
    data_dir = Path(artifact.download())

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        prepare_data.main(
            raw_csv=data_dir / "complaints.csv",
            output_dir=output_dir,
            full=full,
        )
        out_artifact = wandb.Artifact("cfpb-complaints-processed", type="dataset")
        for fname in ["train.csv", "val.csv", "test.csv", "label_map.json"]:
            out_artifact.add_file(str(output_dir / fname))
        run.log_artifact(out_artifact)

    run.finish()


@app.local_entrypoint()
def main(full: bool = False):
    run_prepare.remote(full=full)
