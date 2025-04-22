import yaml, wandb
import torch
import os
import sys
from pathlib import Path
from utils.mmseqs_split import split_clusters
from evoflow_generator.train import train_evoflow
from evoflow_generator.generate import generate_sequences
from utils.downstream import evaluate_downstream

# 1. Load config & W&B
try:
    cfg = yaml.safe_load(open("evoflow_config.yaml"))
except FileNotFoundError:
    print("Error: evoflow_config.yaml not found in the current directory")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing evoflow_config.yaml: {e}")
    sys.exit(1)

# Initialize wandb
try:
    wandb.init(entity=cfg["wandb"]["entity"],
               project=cfg["wandb"]["project"],
               config=cfg,
               name=cfg["wandb"].get("name", "evo_flow_ft_1"))
except Exception as e:
    print(f"Warning: Failed to initialize wandb: {e}")
    print("Continuing without wandb logging...")
    raise

wandb.log({"step_status": "W&B initialized starting experiment"})

# Step 2 - Create directory structure and split into train/val/test
def create_directories(cfg):
    """Create all necessary directories from config"""
    for dir_path in cfg["dirs"].values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory created: {dir_path}")

create_directories(cfg)

splits_dir = cfg["cluster"].get("splits_dir", "dataset/splits")
os.makedirs(splits_dir, exist_ok=True)
wandb.log({"step_status": f"Splitting sequences into {splits_dir}..."})

# Check if splits already exist, if not create them
if not all(os.path.exists(os.path.join(splits_dir, f"{split}.fasta")) for split in ["train", "val", "test"]):
    split_clusters(
        cfg["dataset_path"],
        splits_dir,
        cfg["cluster"]["min_seq_id"],
        cfg["cluster"]["coverage"],
        frac_train=cfg["cluster"]["frac_train"],
        frac_val=cfg["cluster"]["frac_val"],
        seed=cfg["cluster"].get("seed"),
        cleanup=cfg["cluster"].get("cleanup", False)
    )
else:
    print(f"Using existing splits in {splits_dir}")

wandb.log({"step_status": "Split into train/val/test completed"})

# 3. Train EvoFlow model
wandb.log({"step_status": "Starting EvoFlow fine-tuning..."})
model = train_evoflow(cfg)
wandb.log({"step_status": "EvoFlow fine-tuning completed"})

# 4. Generate sequences
wandb.log({"step_status": "Generating sequences with fine-tuned EvoFlow..."})
samples_dir = cfg["sampling"]["output_dir"]
os.makedirs(samples_dir, exist_ok=True)

sequences = generate_sequences(
    cfg=cfg,
    num_samples=cfg["sampling"]["num_samples"],
    output_dir=samples_dir,
    run_name=cfg["wandb"].get("name", "evo_flow_ft_1")
)
wandb.log({"step_status": "Sequence generation completed"})

# 5. Validate generated sequences
wandb.log({"step_status": "Running downstream evaluation on generated sequences..."})

# We need to adapt the config to make downstream evaluation work
# by pointing it to the generated sequences
cfg["best_model_path"] = os.path.join(cfg["evoflow"]["model_dir"], "evoflow_best.pt")
evaluate_downstream(cfg)
wandb.log({"step_status": "Downstream evaluation completed"})

# Log samples to W&B
samples_file = os.path.join(samples_dir, f"{cfg['wandb'].get('name', 'evo_flow_ft_1')}_samples.fasta")
if os.path.exists(samples_file):
    try:
        wandb.save(samples_file)
        print(f"Uploaded {samples_file} to W&B")
        wandb.log({"step_status": "Samples uploaded to W&B"})
    except Exception as e:
        print(f"Warning: Failed to upload samples file to W&B: {e}")

# Finish wandb run
try:
    wandb.finish()
except:
    pass

print("EvoFlow experiment completed successfully!") 