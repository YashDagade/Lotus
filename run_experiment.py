import yaml, wandb
import torch
import os
from generator.embed_sequences import embed
from generator.train import train
from generator.downstream import evaluate_downstream
from generator.solver import sample_sequences
from utils.mmseqs_split import split_clusters
from generator.models import FlowMatchingNet

# 1. Load config & W&B
cfg = yaml.safe_load(open("config.yaml"))

# Create output directories
os.makedirs("outs", exist_ok=True)
os.makedirs("outs/mmseqs2", exist_ok=True)  # For MMseqs2 files
os.makedirs("outs/embeddings", exist_ok=True)  # For ESM embeddings
os.makedirs("outs/models", exist_ok=True)  # For saved models
os.makedirs("outs/samples", exist_ok=True)  # For generated sequences
os.makedirs("logs", exist_ok=True)  # For log files

wandb.init(entity=cfg["wandb"]["entity"],
           project=cfg["wandb"]["project"],
           config=cfg,
           name=cfg["wandb"].get("name", "latent_flow_run"))

# 2. (Optional) split into train/val/test
split_clusters(cfg["dataset_path"], "dataset/splits",
               cfg["cluster"]["min_seq_id"],
               cfg["cluster"]["coverage"])

# 3. Embed
embed(cfg["dataset_path"], "outs/embeddings/cas9_embeddings.pt", cfg["esm_model"])

# 4. Train FM
model = train(cfg)

# 5. Downstream eval
evaluate_downstream(cfg)

# 6. Generate sequences after training
print("Generating sequences after training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best model if not returned by train()
if model is None:
    model = FlowMatchingNet(cfg["flow"]["emb_dim"], cfg["flow"]["hidden_dim"]).to(device)
    model.load_state_dict(torch.load("best_flow.pt", map_location=device))

# Create solver
solver = model.get_solver(cfg)

# Sample sequences
sample_sequences(
    model=model,
    solver=solver,
    cfg=cfg,
    num_samples=cfg["sampling"]["num_samples"],
    method=cfg["sampling"]["method"],
    steps=cfg["sampling"]["steps"],
    output_dir=cfg["sampling"]["output_dir"],
    run_name=cfg["wandb"].get("name", "latent_flow_run")
)

# Log samples to W&B
samples_file = os.path.join(cfg["sampling"]["output_dir"], f"{cfg['wandb'].get('name', 'latent_flow_run')}_samples.fasta")
if os.path.exists(samples_file):
    wandb.save(samples_file)
    print(f"Uploaded {samples_file} to W&B")

wandb.finish() 