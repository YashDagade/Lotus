import os
import tempfile
import shutil
import torch
import wandb
from flow_generator.decode import decode_latents
from flow_generator.solver import ODESolver
import sys

# Helper function for safer wandb logging
def log_wandb(metrics):
    try:
        wandb.log(metrics)
    except Exception as e:
        print(f"Warning: Failed to log to wandb: {e}")

def evaluate_downstream(cfg):
    """
    Evaluate generated sequences on downstream tasks:
      1. Load best model
      2. Sample latents & decode to sequences
      3. Compute sequence‐level metrics
      4. Optionally run structure prediction & HMM scans
      5. Log everything to W&B and return metrics dict
    """
    print("Running downstream evaluation...")
    
    # Check if W&B is active
    wandb_enabled = wandb.run is not None
    
    try:
        if "flow" in cfg:
            # Flow model path
            from flow_generator.models import FlowMatchingNet
            model_path = cfg["best_model_path"]
            if not os.path.exists(model_path):
                print(f"Warning: Best model not found at {model_path}, skipping model-based tasks")
                model = None
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Loading model from: {model_path} to {device}")
                model = FlowMatchingNet(cfg["flow"]["emb_dim"], cfg["flow"]["hidden_dim"]).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print("Model loaded successfully")
        else:
            # Handling EvoFlow or other models that don't need latent generation
            model = None
            print("No flow model specified in config, skipping model-based tasks")

        # Get samples dir
        samples_dir = cfg["sampling"]["output_dir"]
        os.makedirs(samples_dir, exist_ok=True)
        
        # Find generated sequences
        fasta_path = None
        if model is not None:
            # Sample from model
            solver = ODESolver(model, cfg)
            num_samples = cfg["downstream"].get("num_samples", 20)
            print(f"Sampling {num_samples} sequences...")
            zs = solver.sample_latents(num_samples)
            print(f"Latents shape: {zs.shape}")
            print("Decoding sequences...")
            sequences = decode_latents(zs)
            print(f"Generated {len(sequences)} sequences")
            
            # Write to FASTA
            fasta_path = os.path.join(samples_dir, "downstream_seqs.fasta")
        else:
            # Look for previously generated sequences
            run_name = cfg["wandb"].get("name", "samples")
            potential_fasta = os.path.join(samples_dir, f"{run_name}_samples.fasta")
            if os.path.exists(potential_fasta):
                fasta_path = potential_fasta
                print(f"Using existing generated sequences from {fasta_path}")
            else:
                print(f"No sequences found at {potential_fasta}")
                return {"downstream/error": "No sequences found for evaluation"}
        
        # Check if we have sequences to evaluate
        if fasta_path is None or not os.path.exists(fasta_path):
            print("No sequences available for downstream evaluation.")
            return {"downstream/error": "No sequences available"}
            
        # Get sequences from file
        from Bio import SeqIO
        sequences = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq))
        
        print(f"Loaded {len(sequences)} sequences from {fasta_path}")
        
        # 3) Basic sequence metrics
        valid = [s for s in sequences if len(s) > 10 and not s.startswith("MOCK")]
        seq_quality = len(valid) / max(1, len(sequences))
        aas = set("ACDEFGHIKLMNPQRSTVWY")
        aa_vals, lengths = [], []
        for s in valid:
            aa_vals.append(sum(1 for c in s if c in aas)/len(s))
            lengths.append(len(s))
        avg_validity = float(sum(aa_vals)/len(aa_vals)) if aa_vals else 0.0
        avg_length   = float(sum(lengths)/len(lengths)) if lengths else 0.0

        metrics = {
            "downstream/seq_quality": seq_quality,
            "downstream/aa_validity":  avg_validity,
            "downstream/avg_length":   avg_length,
            "downstream/num_valid":    len(valid),
        }
        
        print(f"Sequence quality: {seq_quality:.4f}")
        print(f"AA validity: {avg_validity:.4f}")
        print(f"Average length: {avg_length:.1f}")
        print(f"Valid sequences: {len(valid)}/{len(sequences)}")

        # length histogram
        if lengths:
            if wandb_enabled:
                metrics["downstream/length_hist"] = wandb.Histogram(lengths)

        # 4) Save FASTA as artifact
        with open(fasta_path, "w") as fw:
            for i, s in enumerate(valid):
                fw.write(f">sample_{i}\n{s}\n")
        print(f"Saved {len(valid)} sequences to {fasta_path}")
        
        if wandb_enabled:
            try:
                wandb.save(fasta_path)
                metrics["downstream/fasta_path"] = fasta_path
            except Exception as e:
                print(f"Warning: Could not save FASTA to wandb: {e}")

        # 5) Structural + HMM evaluation
        # — ColabFold
        try:
            from colabfold.batch import run as colabfold_run
            if cfg["downstream"].get("use_structural_validation", True):
                struct_dir = os.path.join(samples_dir, "structures")
                os.makedirs(struct_dir, exist_ok=True)
                print("Running ColabFold...")
                colabfold_run(
                    fasta_path,
                    result_dir=struct_dir,
                    model_type=cfg["downstream"].get("model_type","AlphaFold2-ptm"),
                    use_templates=False,
                    use_amber=False,
                    num_models=1,
                    num_recycles=cfg["downstream"].get("num_recycles",1)
                )
                if wandb_enabled:
                    try:
                        wandb.save(os.path.join(struct_dir, "*"))
                        metrics["downstream/colabfold_path"] = struct_dir
                    except Exception as e:
                        print(f"Warning: Could not save ColabFold results to wandb: {e}")
        except ImportError:
            print("ColabFold not installed; skipping structure prediction")
        except Exception as e:
            print(f"Error running ColabFold: {e}")

        # — TM‐score against references
        refs = cfg["downstream"].get("ref_pdbs", [])
        if refs and valid:
            try:
                from utils.validate import calculate_tm_score
                tm_scores = []
                for i in range(len(valid)):
                    pdb_f = os.path.join(struct_dir, f"val_{i}_unrelaxed_rank_0.pdb")
                    if os.path.exists(pdb_f):
                        for r in refs:
                            tm_scores.append(calculate_tm_score(pdb_f, r))
                if tm_scores:
                    metrics.update({
                        "downstream/tm_avg": sum(tm_scores)/len(tm_scores),
                        "downstream/tm_min": min(tm_scores),
                        "downstream/tm_max": max(tm_scores),
                    })
                    print(f"TM-scores: avg={metrics['downstream/tm_avg']:.4f}, min={metrics['downstream/tm_min']:.4f}, max={metrics['downstream/tm_max']:.4f}")
            except Exception as e:
                print(f"Error calculating TM-scores: {e}")

        # — HMMER bit‐scores
        hmm_profile = cfg["downstream"].get("hmm_profile","")
        if hmm_profile and os.path.exists(hmm_profile):
            try:
                from utils.validate import hmm_bit_scores
                bits = hmm_bit_scores(fasta_path, hmm_profile)
                if bits:
                    metrics["downstream/hmm_bit_avg"] = sum(bits)/len(bits)
                    if wandb_enabled:
                        metrics["downstream/hmm_bit_hist"] = wandb.Histogram(bits)
                    print(f"HMM bit-score average: {metrics['downstream/hmm_bit_avg']:.4f}")
            except Exception as e:
                print(f"Error calculating HMM bit-scores: {e}")

        # 6) Log & return
        if wandb_enabled:
            log_wandb(metrics)
        return metrics
        
    except Exception as e:
        print(f"Error in downstream evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {"downstream/error": str(e)}
