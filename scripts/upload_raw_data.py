import wandb

run = wandb.init(project="multitagger", job_type="upload-raw")
artifact = wandb.Artifact("cfpb-complaints-raw", type="dataset")
artifact.add_file("data/raw/complaints.csv")
run.log_artifact(artifact)
run.finish()
