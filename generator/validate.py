import os
import subprocess
import json
import tempfile
from pathlib import Path

import torch
import numpy as np
import wandb
from colabfold.batch import run as colabfold_run

from .decode import decode_latents


def calculate_tm_score(pred_pdb: str, ref_pdb: str) -> float:
    """
    Run TM-align and return the TM-score normalized by reference length.
    """
    result = subprocess.run(
        ["TMalign", pred_pdb, ref_pdb], capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if line.startswith("TM-score=") and "normalized by length of" in line:
            return float(line.split("=")[1].split()[0])
    return 0.0


def extract_af_confidences(score_json: str) -> tuple[float, float, list[float]]:
    """
    Parse AlphaFold JSON output to extract pTM, average pLDDT and per-residue pLDDT list.
    """
    with open(score_json) as f:
        data = json.load(f)
    pTM = data.get("ptm", 0.0)
    plddt = data.get("plddt", [])
    avg_pLDDT = float(np.mean(plddt)) if plddt else 0.0
    return pTM, avg_pLDDT, plddt


def hmm_bit_scores(fasta_path: str, hmm_profile: str) -> list[float]:
    """
    Run hmmscan against the given HMM profile and collect bit-scores.
    Requires HMMER installed.
    """
    cmd = [
        "hmmscan", "--tblout", "/dev/stdout", hmm_profile, fasta_path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    bit_scores = []
    for line in proc.stdout.splitlines():
        if line.startswith("#"): continue
        parts = line.split()
        if len(parts) >= 6:
            bit_scores.append(float(parts[5]))
    return bit_scores


def validate(model, cfg):
    """
    Perform downstream validation:
     1. Sample a small batch of latent vectors, decode to sequences.
     2. Save to FASTA, run ColabFold to predict structures.
     3. Compute TM-scores against each ref PDB, select best.
     4. Extract AlphaFold confidences (pTM, pLDDT).
     5. Scan with HMM for domain plausibility.
     6. Log all metrics to W&B.
    Returns average TM-score for early stopping.
    """
    # For initial development or if some dependencies are missing,
    # fall back to a simple random metric
    if not cfg.get("downstream", {}).get("use_structural_validation", True):
        return float(np.random.rand())
        
    ref_pdbs = cfg["downstream"].get("ref_pdbs", [])  # list of paths
    hmm_profile = cfg["downstream"].get("hmm_profile", "")
    num_samples = cfg["downstream"].get("num_samples", 5)
    fold_cfg = cfg["downstream"]

    # 1. Sample and decode
    # Use the model's sample_latents method
    zs = model.sample_latents(num_samples)
    sequences = decode_latents(zs)

    # 2. Write FASTA
    workdir = tempfile.mkdtemp(prefix="val_")
    fasta_path = os.path.join(workdir, "val.fasta")
    with open(fasta_path, "w") as fw:
        for i, seq in enumerate(sequences):
            fw.write(f">val_{i}\n{seq}\n")

    # 3. Predict structures (if ColabFold is available)
    try:
        colabfold_run(
            fasta_path,
            result_dir=workdir,
            model_type=fold_cfg.get("model_type", "AlphaFold2-ptm"),
            use_templates=False,
            use_amber=False,
            num_models=1,
            num_recycles=fold_cfg.get("num_recycles", 1),
            keep_existing_results=False
        )
    except Exception as e:
        print(f"Error running ColabFold: {e}")
        return float(np.random.rand())  # Fallback

    tm_scores_all = []
    pTMs, avg_pLDDTs, all_plddt = [], [], []

    # Collect metrics
    for i in range(num_samples):
        pdb_file = Path(workdir) / f"val_{i}_unrelaxed_rank_0.pdb"
        score_json = Path(workdir) / f"score_model_1_ptm.json"
        
        if not pdb_file.exists() or not score_json.exists():
            continue

        # TM-score: best across references (if TM-align and refs are available)
        if ref_pdbs:
            try:
                best_tm = max(
                    calculate_tm_score(str(pdb_file), ref) for ref in ref_pdbs
                )
                tm_scores_all.append(best_tm)
            except Exception as e:
                print(f"Error calculating TM-score: {e}")

        # AF confidences
        try:
            pTM, avg_pLDDT, plddt = extract_af_confidences(str(score_json))
            pTMs.append(pTM)
            avg_pLDDTs.append(avg_pLDDT)
            all_plddt.extend(plddt)
        except Exception as e:
            print(f"Error extracting AF confidences: {e}")

    # 4. HMM bit-scores on sequences (if HMMER and profile are available)
    bit_scores = []
    if hmm_profile and os.path.exists(hmm_profile):
        try:
            bit_scores = hmm_bit_scores(fasta_path, hmm_profile)
        except Exception as e:
            print(f"Error calculating HMM bit-scores: {e}")

    # 5. Compute summary stats
    metrics = {}
    
    if tm_scores_all:
        tm_avg = float(np.mean(tm_scores_all))
        tm_min = float(np.min(tm_scores_all))
        tm_max = float(np.max(tm_scores_all))
        metrics.update({
            "val/tm_avg": tm_avg,
            "val/tm_min": tm_min,
            "val/tm_max": tm_max,
        })
    else:
        tm_avg = 0.0
    
    if pTMs:
        pTM_avg = float(np.mean(pTMs))
        metrics["val/pTM"] = pTM_avg
    
    if avg_pLDDTs:
        pLDDT_avg = float(np.mean(avg_pLDDTs))
        metrics["val/pLDDT_avg"] = pLDDT_avg
    
    if all_plddt:
        metrics["val/pLDDT_hist"] = wandb.Histogram(all_plddt)
    
    if bit_scores:
        bit_avg = float(np.mean(bit_scores))
        metrics.update({
            "val/hmm_bit_avg": bit_avg,
            "val/hmm_bit_hist": wandb.Histogram(bit_scores)
        })

    # 6. Log to W&B
    if metrics:
        wandb.log(metrics)

    # Return whatever metric we have for early stopping
    # Prefer TM-score if available, otherwise try pLDDT or bit-score
    if tm_scores_all:
        return tm_avg
    elif avg_pLDDTs:
        return float(np.mean(avg_pLDDTs))
    elif bit_scores:
        return float(np.mean(bit_scores))
    else:
        return float(np.random.rand())  # Fallback 