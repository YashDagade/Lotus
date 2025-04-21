import os
import tempfile
import shutil
import torch
import wandb
from .decode import decode_latents
from .solver import ODESolver

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
    from .models import FlowMatchingNet

    # 1) Load model
    model_path = cfg["best_model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {model_path}")
    model = FlowMatchingNet(cfg["flow"]["emb_dim"], cfg["flow"]["hidden_dim"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 2) Sample & decode
    solver = ODESolver(model, cfg)
    num_samples = cfg["downstream"].get("num_samples", 20)
    print(f"Sampling {num_samples} sequences...")
    zs = solver.sample_latents(num_samples)
    print("Decoding sequences...")
    sequences = decode_latents(zs)

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

    # length histogram
    if lengths:
        metrics["downstream/length_hist"] = wandb.Histogram(lengths)

    # 4) Save FASTA as artifact
    samples_dir = cfg["sampling"]["output_dir"]
    os.makedirs(samples_dir, exist_ok=True)
    fasta_path = os.path.join(samples_dir, "downstream_seqs.fasta")
    with open(fasta_path, "w") as fw:
        for i, s in enumerate(valid):
            fw.write(f">sample_{i}\n{s}\n")
    print(f"Saved {len(valid)} sequences to {fasta_path}")
    wandb.save(fasta_path)
    metrics["downstream/fasta_path"] = fasta_path

    # 5) Structural + HMM evaluation
    # — ColabFold
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
            wandb.save(os.path.join(struct_dir, "*"))
            metrics["downstream/colabfold_path"] = struct_dir
    except ImportError:
        print("ColabFold not installed; skipping structure prediction")

    # — TM‐score against references
    refs = cfg["downstream"].get("ref_pdbs", [])
    if refs and valid:
        from .validate import calculate_tm_score
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

    # — HMMER bit‐scores
    hmm_profile = cfg["downstream"].get("hmm_profile","")
    if hmm_profile and os.path.exists(hmm_profile):
        from .validate import hmm_bit_scores
        bits = hmm_bit_scores(fasta_path, hmm_profile)
        if bits:
            metrics["downstream/hmm_bit_avg"]  = sum(bits)/len(bits)
            metrics["downstream/hmm_bit_hist"] = wandb.Histogram(bits)

    # 6) Log & return
    wandb.log(metrics)
    return metrics
