import os
import subprocess
import json
import tempfile
from pathlib import Path

import torch
import numpy as np
import wandb

from .decode import decode_latents
from .solver import ODESolver


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
    # Import needed modules
    import os
    import tempfile
    from pathlib import Path
    import numpy as np
    import wandb
    
    # Import local modules
    from .decode import decode_latents
    from .solver import ODESolver
    
    # Check if dependencies are available
    has_colabfold = False
    try:
        from colabfold.batch import run as colabfold_run
        has_colabfold = True
    except ImportError:
        print("ColabFold not available, structural validation metrics will be limited")
    
    # For testing mode, perform a simplified but still meaningful validation
    if not cfg.get("downstream", {}).get("use_structural_validation", True):
        print("Structural validation disabled, performing simplified validation")
        # Create a solver for the model
        solver = ODESolver(model, cfg)
        
        # Sample a small batch of latents
        num_samples = cfg["downstream"].get("num_samples", 5)
        zs = solver.sample_latents(num_samples)
        
        # Attempt to decode sequences (this is a real test)
        try:
            sequences = decode_latents(zs)
            # Check if sequences were generated properly
            valid_sequences = [seq for seq in sequences if len(seq) > 10 and not seq.startswith("MOCK")]
            seq_quality = len(valid_sequences) / max(1, len(sequences))
            print(f"Generated {len(valid_sequences)} valid sequences out of {len(sequences)}")
            
            # Check for sequence validity (simple AA composition check)
            valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
            aa_validity = []
            for seq in valid_sequences:
                valid_letters = sum(1 for aa in seq if aa in valid_aas)
                aa_validity.append(valid_letters / max(1, len(seq)))
            
            if aa_validity:
                avg_validity = sum(aa_validity) / len(aa_validity)
                print(f"Average sequence validity: {avg_validity:.4f}")
                # Log to wandb
                wandb.log({
                    "val/seq_quality": seq_quality,
                    "val/aa_validity": avg_validity
                })
                return avg_validity  # Return a meaningful metric
            else:
                print("No valid sequences generated")
                return 0.0  # Return a poor score to indicate failure
                
        except Exception as e:
            print(f"Error in sequence decoding: {e}")
            return 0.0  # Return a poor score to indicate failure
    
    # Main validation path with structural validation
    ref_pdbs = cfg["downstream"].get("ref_pdbs", [])  # list of paths
    hmm_profile = cfg["downstream"].get("hmm_profile", "")
    num_samples = cfg["downstream"].get("num_samples", 5)
    fold_cfg = cfg["downstream"]

    # 1. Sample and decode
    solver = ODESolver(model, cfg)
    zs = solver.sample_latents(num_samples)
    sequences = decode_latents(zs)

    # Check if we got valid sequences
    valid_sequences = [seq for seq in sequences if len(seq) > 10 and not seq.startswith("MOCK")]
    if not valid_sequences:
        print("No valid sequences generated, returning poor score")
        return 0.0
        
    # 2. Write FASTA
    workdir = tempfile.mkdtemp(prefix="val_")
    fasta_path = os.path.join(workdir, "val.fasta")
    with open(fasta_path, "w") as fw:
        for i, seq in enumerate(sequences):
            fw.write(f">val_{i}\n{seq}\n")

    # 3. Predict structures (if ColabFold is available)
    tm_scores_all = []
    pTMs, avg_pLDDTs, all_plddt = [], [], []
    
    if has_colabfold:
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
    
    # Always include sequence metrics
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    aa_validity = []
    for seq in valid_sequences:
        valid_letters = sum(1 for aa in seq if aa in valid_aas)
        aa_validity.append(valid_letters / max(1, len(seq)))
    
    if aa_validity:
        avg_validity = sum(aa_validity) / len(aa_validity)
        metrics["val/aa_validity"] = avg_validity
    
    if tm_scores_all:
        tm_avg = float(np.mean(tm_scores_all))
        tm_min = float(np.min(tm_scores_all))
        tm_max = float(np.max(tm_scores_all))
        metrics.update({
            "val/tm_avg": tm_avg,
            "val/tm_min": tm_min,
            "val/tm_max": tm_max,
        })
    
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

    # Return the best metric we have for early stopping
    if tm_scores_all:
        return float(np.mean(tm_scores_all))
    elif avg_pLDDTs:
        return float(np.mean(avg_pLDDTs))
    elif bit_scores:
        return float(np.mean(bit_scores))
    elif aa_validity:
        return float(np.mean(aa_validity))
    else:
        # If we have no metrics, return a poor score to indicate failure
        return 0.0

if __name__ == "__main__":
    print("Testing ODE Solvers...")
    
    # Import locally to avoid circular imports
    from flow_generator.models import FlowMatchingNet
    
    # Create a mock model
    mock_cfg = {
        "flow": {
            "emb_dim": 32,
            "hidden_dim": 64
        },
        "downstream": {
            "num_samples": 2,
            "use_structural_validation": False,  # Disable for testing
            "ref_pdbs": [],
            "hmm_profile": ""
        },
        "wandb": {
            "project": "test"
        }
    }
    
    # Initialize model 
    emb_dim = mock_cfg["flow"]["emb_dim"]
    hidden_dim = mock_cfg["flow"]["hidden_dim"]
    model = FlowMatchingNet(emb_dim, hidden_dim)
    
    try:
        # Test validation with structural validation disabled
        print("Testing validation with structural validation disabled...")
        mock_validation_score = validate(model, mock_cfg)
        print(f"Validation returned score: {mock_validation_score}")
        
        # Test individual validation functions
        print("\nTesting TM-score calculation (mock)...")
        
        # Create temporary files for testing
        with tempfile.NamedTemporaryFile(suffix=".pdb") as f1, \
             tempfile.NamedTemporaryFile(suffix=".pdb") as f2:
            
            # Write mock PDB content
            f1.write(b"MOCK PDB 1")
            f2.write(b"MOCK PDB 2")
            f1.flush()
            f2.flush()
            
            # Test function
            try:
                score = calculate_tm_score(f1.name, f2.name)
                print(f"TM-score calculation attempted: {score}")
            except Exception as e:
                print(f"TM-score calculation failed (expected if TMalign not installed): {e}")
        
        # Test extract_af_confidences (mock)
        print("\nTesting extract_af_confidences (mock)...")
        
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            # Write mock JSON content
            mock_json = {
                "ptm": 0.75,
                "plddt": [80.5, 85.2, 90.1, 87.3]
            }
            f.write(json.dumps(mock_json).encode())
            f.flush()
            
            try:
                ptm, avg_plddt, plddt_list = extract_af_confidences(f.name)
                print(f"Extracted pTM: {ptm}")
                print(f"Extracted avg pLDDT: {avg_plddt}")
                print(f"Extracted pLDDT values: {plddt_list}")
            except Exception as e:
                print(f"extract_af_confidences failed: {e}")
        
        print("\nAll validation tests completed.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    print("Test finished.") 