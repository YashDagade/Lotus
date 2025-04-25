import yaml
import wandb
import torch
import os
import sys
import time
import argparse
from pathlib import Path
from utils.mmseqs_split import split_clusters
from evoflow_generator.train import train_evoflow
from evoflow_generator.generate import generate_sequences
from utils.downstream import evaluate_downstream
import traceback

def parse_args():
    parser = argparse.ArgumentParser(description="Run EvoFlow experiment")
    parser.add_argument("--config", type=str, default="evoflow_config.yaml",
                        help="Path to the configuration file (default: evoflow_config.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform a dry run with minimal processing")
    return parser.parse_args()

args = parse_args()
config_file = args.config
dry_run = args.dry_run

start_time = time.time()

print("="*80)
print(f"Starting EvoFlow experiment with config: {config_file}")
if dry_run:
    print("DRY RUN MODE: Will perform minimal processing for testing")
print("="*80)

# 1. Load config & W&B
try:
    cfg = yaml.safe_load(open(config_file))
    print(f"Loaded configuration from {config_file}")
    
    # Print key configuration parameters
    print(f"Dataset: {cfg['dataset_path']}")
    print(f"Model checkpoint: {cfg.get('model_checkpoint', 'fredzzp/EvoFlow-650M-context-3070')}")
    print(f"W&B project: {cfg['wandb']['project']}")
    print(f"W&B run name: {cfg['wandb']['name']}")
    print(f"Training epochs: {cfg['evoflow']['epochs']}")
    print(f"Batch size: {cfg['evoflow']['batch_size']}")
    print(f"Learning rate: {cfg['evoflow']['learning_rate']}")
    print(f"Number of samples to generate: {cfg['sampling']['num_samples']}")
except FileNotFoundError:
    print("Error: evoflow_config.yaml not found in the current directory")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing evoflow_config.yaml: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error processing configuration: {e}")
    traceback.print_exc()
    sys.exit(1)

# Initialize wandb
wandb_enabled = True
try:
    wandb.init(entity=cfg["wandb"]["entity"],
               project=cfg["wandb"]["project"],
               config=cfg,
               name=cfg["wandb"].get("name", "evo_flow_ft_4"))
    print(f"Successfully initialized W&B run: {cfg['wandb'].get('name', 'evo_flow_ft_4')}")
except Exception as e:
    print(f"Warning: Failed to initialize wandb: {e}")
    print("Continuing without wandb logging...")
    wandb_enabled = False

# Helper function for safer wandb logging
def log_wandb(metrics):
    if wandb_enabled:
        try:
            wandb.log(metrics)
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")

log_wandb({"step_status": "W&B initialized starting experiment"})

try:
    # Step 2 - Create directory structure and split into train/val/test
    def create_directories(cfg):
        """Create all necessary directories from config"""
        for key, dir_path in cfg["dirs"].items():
            os.makedirs(dir_path, exist_ok=True)
            print(f"Directory created: {dir_path}")

    create_directories(cfg)

    splits_dir = cfg["cluster"].get("splits_dir", "dataset/splits")
    os.makedirs(splits_dir, exist_ok=True)
    log_wandb({"step_status": f"Splitting sequences into {splits_dir}..."})

    # Force recreate the splits for this run
    print(f"Creating train/val/test splits in {splits_dir}...")
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

    log_wandb({"step_status": "Split into train/val/test completed"})

    # Verify splits were created
    for split in ["train", "val", "test"]:
        split_file = os.path.join(splits_dir, f"{split}.fasta")
        if os.path.exists(split_file):
            from Bio import SeqIO
            count = sum(1 for _ in SeqIO.parse(split_file, "fasta"))
            print(f"{split.capitalize()} split: {count} sequences")
            log_wandb({f"data/{split}_count": count})
        else:
            print(f"Warning: {split} split file not found!")

    # 3. Train EvoFlow model
    log_wandb({"step_status": "Starting EvoFlow fine-tuning..."})
    print("\n" + "="*50)
    print("STEP 3: TRAINING EVOFLOW MODEL")
    print("="*50)
    
    train_start_time = time.time()
    model = train_evoflow(cfg)
    train_duration = time.time() - train_start_time
    
    print(f"Training completed in {train_duration:.1f} seconds ({train_duration/60:.1f} minutes)")
    log_wandb({"timing/training_seconds": train_duration})
    log_wandb({"step_status": "EvoFlow fine-tuning completed"})

    # 4. Generate sequences
    log_wandb({"step_status": "Generating sequences with fine-tuned EvoFlow..."})
    print("\n" + "="*50)
    print("STEP 4: GENERATING SEQUENCES")
    print("="*50)
    
    samples_dir = cfg["sampling"]["output_dir"]
    os.makedirs(samples_dir, exist_ok=True)

    generation_start_time = time.time()
    sequences = generate_sequences(
        cfg=cfg,
        num_samples=cfg["sampling"]["num_samples"],
        output_dir=samples_dir,
        run_name=cfg["wandb"].get("name", "evo_flow_ft_4"),
        batch_size=1  # Use batch size of 1 to avoid memory issues
    )
    generation_duration = time.time() - generation_start_time
    
    print(f"Generation completed in {generation_duration:.1f} seconds ({generation_duration/60:.1f} minutes)")
    log_wandb({"timing/generation_seconds": generation_duration})
    log_wandb({"step_status": "Sequence generation completed"})

    # 5. Validate generated sequences
    log_wandb({"step_status": "Running downstream evaluation on generated sequences..."})
    print("\n" + "="*50)
    print("STEP 5: DOWNSTREAM EVALUATION")
    print("="*50)

    # We need to adapt the config to make downstream evaluation work
    # by pointing it to the generated sequences
    cfg["best_model_path"] = os.path.join(cfg["evoflow"]["model_dir"], "evoflow_best.pt")
    
    eval_start_time = time.time()
    eval_metrics = evaluate_downstream(cfg)
    eval_duration = time.time() - eval_start_time
    
    print(f"Evaluation completed in {eval_duration:.1f} seconds ({eval_duration/60:.1f} minutes)")
    log_wandb({"timing/evaluation_seconds": eval_duration})
    log_wandb({"step_status": "Downstream evaluation completed"})

    # Log key metrics
    if eval_metrics:
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)) and not key.endswith("_hist"):
                print(f"{key}: {value}")

    # Log samples to W&B
    samples_file = os.path.join(samples_dir, f"{cfg['wandb'].get('name', 'evo_flow_ft_4')}_samples.fasta")
    if os.path.exists(samples_file):
        try:
            if wandb_enabled:
                wandb.save(samples_file)
                print(f"Uploaded {samples_file} to W&B")
                log_wandb({"step_status": "Samples uploaded to W&B"})
        except Exception as e:
            print(f"Warning: Failed to upload samples file to W&B: {e}")

    # Calculate total runtime
    total_duration = time.time() - start_time
    print(f"\nTotal runtime: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    log_wandb({"timing/total_seconds": total_duration})

except Exception as e:
    print(f"Error during experiment execution: {e}")
    traceback.print_exc()
    log_wandb({"step_status": f"Error: {str(e)}"})
    if wandb_enabled:
        wandb.finish(exit_code=1)
    sys.exit(1)

# Finish wandb run
if wandb_enabled:
    try:
        wandb.finish()
    except:
        pass

print("="*80)
print("EvoFlow experiment completed successfully!")
print("="*80) 