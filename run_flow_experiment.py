import yaml, wandb
import torch
import os
import sys
from pathlib import Path
from flow_generator.embed_sequences import embed
from flow_generator.train import train
from utils.downstream import evaluate_downstream
from flow_generator.solver import sample_sequences
from utils.mmseqs_split import split_clusters
from flow_generator.models import FlowMatchingNet

# 1. Load config & W&B
try:
    cfg = yaml.safe_load(open("flow_config.yaml")) # hence cfg is a dict with all the config parameters specified in the YAML!!!
except FileNotFoundError:
    print("Error: flow_config.yaml not found in the current directory")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing flow_config.yaml: {e}")
    sys.exit(1)

# Initialize wandb
try:
    wandb.init(entity=cfg["wandb"]["entity"],
               project=cfg["wandb"]["project"],
               config=cfg,
               name=cfg["wandb"].get("name", "latent_flow_run"))
except Exception as e:
    print(f"Warning: Failed to initialize wandb: {e}")
    print("Continuing without wandb logging...")
    raise

wandb.log({"step_status": "W&B initialized starting experiment"})

# Step 2 - Create directory structure and split into train/val/test
def create_directories(cfg):
    """Create all necessary directories from config"""
    for dir_path in cfg["dirs"].values(): # lowkey the keys are kind of useless we just need the values
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory created: {dir_path}")

create_directories(cfg)


splits_dir = cfg["cluster"].get("splits_dir", "dataset/splits")
os.makedirs(splits_dir, exist_ok=True)
wandb.log({"step_status": f"Splitting sequences into {splits_dir}..."})


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

wandb.log({"step_status": "Split into train/val/test completed, starting to embed now"})

# Step 3. Embed sequences
embeddings_path = cfg["train_embeddings_path"] # this is where we save the embeddings after they are trained
# Ensure parent directory exists
os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
wandb.log({"step_status": f"Generating ESM embeddings, saving to: {embeddings_path}"})
# IMPORTNAT: Let's only embed the training set for now
train_fasta = os.path.join(splits_dir, "train.fasta")
embed(train_fasta, embeddings_path, cfg["esm_model"])

wandb.log({"step_status": "Embedding completed, starting to train flow matching model..."}) #AWESOME! So far this should work!



### ABOVE Where all relitrevely trivial steps



# 4. Train flow matching model
model = train(cfg)
wandb.log({"step_status": "Training completed"})

# 5. Downstream evaluation
wandb.log({"step_status": "Running downstream evaluation..."})
evaluate_downstream(cfg)
wandb.log({"step_status": "Downstream evaluation completed"})

# 6. Generate sequences after training
wandb.log({"step_status": "Generating sequences after training..."})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best model if not returned by train()
if model is None:
    model_path = cfg["best_model_path"]
    wandb.log({"step_status": f"Loading best model from: {model_path}"})
    try:
        model = FlowMatchingNet(cfg["flow"]["emb_dim"], cfg["flow"]["hidden_dim"]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Using newly initialized model.")
        model = FlowMatchingNet(cfg["flow"]["emb_dim"], cfg["flow"]["hidden_dim"]).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using newly initialized model.")
        model = FlowMatchingNet(cfg["flow"]["emb_dim"], cfg["flow"]["hidden_dim"]).to(device)

# Create solver
solver = model.get_solver(cfg)

# Ensure output directory exists
samples_dir = cfg["sampling"]["output_dir"]
os.makedirs(samples_dir, exist_ok=True)

# Sample sequences
print(f"Sampling {cfg['sampling']['num_samples']} sequences...")
sample_sequences(
    model=model,
    solver=solver,
    cfg=cfg,
    num_samples=cfg["sampling"]["num_samples"],
    method=cfg["sampling"]["method"],
    steps=cfg["sampling"]["steps"],
    output_dir=samples_dir,
    run_name=cfg["wandb"].get("name", "latent_flow_run")
)
wandb.log({"step_status": "Sampling completed"})

# Log samples to W&B
samples_file = os.path.join(samples_dir, f"{cfg['wandb'].get('name', 'latent_flow_run')}_samples.fasta")
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

print("Experiment completed successfully!") 