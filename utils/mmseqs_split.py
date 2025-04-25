import subprocess, pandas as pd, random, os, shutil
from Bio import SeqIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_clusters(
    fasta, out_dir,
    id_min=0.3, cov=0.8,
    frac_train=0.8, frac_val=0.1,
    seed=None, cleanup=False
):
    if seed is not None:
        random.seed(seed)
    
    if not os.path.exists(fasta):
        raise FileNotFoundError(f"Input FASTA file not found: {fasta}")

    # Count sequences in input file
    n_seqs = sum(1 for _ in SeqIO.parse(fasta, "fasta"))
    logger.info(f"Found {n_seqs} sequences in input file")
    
    if n_seqs == 0:
        raise ValueError("No sequences found in input file")

    # ensure output dirs
    os.makedirs(out_dir, exist_ok=True)
    mmseqs_dir = os.path.join(out_dir, "mmseqs2")
    
    # Clean up existing files if they exist
    if os.path.exists(mmseqs_dir):
        shutil.rmtree(mmseqs_dir)
    os.makedirs(mmseqs_dir)

    # Create absolute paths
    seqdb = os.path.abspath(os.path.join(mmseqs_dir, "seqDB"))
    clustdb = os.path.abspath(os.path.join(mmseqs_dir, "clusterDB"))
    tmp = os.path.abspath(os.path.join(mmseqs_dir, "tmp"))
    tsv = os.path.abspath(os.path.join(mmseqs_dir, "clusters.tsv"))

    # Ensure tmp directory exists
    os.makedirs(tmp, exist_ok=True)

    # run MMseqs2 with error handling
    try:
        # Create sequence database
        logger.info("Creating sequence database...")
        subprocess.run(["mmseqs", "createdb", os.path.abspath(fasta), seqdb], 
                      check=True, capture_output=True)

        # Run clustering
        logger.info(f"Clustering sequences (min_seq_id={id_min}, coverage={cov})...")
        subprocess.run(["mmseqs", "cluster", seqdb, clustdb, tmp,
                       "--min-seq-id", str(id_min), 
                       "-c", str(cov),
                       "--threads", "1"],  # Reduce threads to avoid resource issues
                      check=True, capture_output=True)

        # Create TSV output
        logger.info("Creating cluster TSV file...")
        subprocess.run(["mmseqs", "createtsv", seqdb, seqdb, clustdb, tsv],
                      check=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running MMseqs2 command: {e.cmd}")
        logger.error(f"Error output: {e.stderr.decode()}")
        raise

    # Read and process clusters
    df = pd.read_csv(tsv, sep="\t", names=["seq","seq2","cluster"])
    clusters = df.cluster.unique().tolist()
    
    if len(clusters) == 0:
        raise ValueError("No clusters found after MMseqs2 clustering")
        
    logger.info(f"Found {len(clusters)} clusters")
    random.shuffle(clusters)

    n = len(clusters)
    i1 = int(frac_train*n)
    i2 = int((frac_train+frac_val)*n)
    splits = {
        "train": set(clusters[:i1]),
        "val":   set(clusters[i1:i2]),
        "test":  set(clusters[i2:])
    }

    # write FASTAs
    records = SeqIO.index(fasta, "fasta")
    mapping = df.set_index("seq")["cluster"].to_dict()
    
    split_counts = {}
    for split, clset in splits.items():
        out_fa = os.path.join(out_dir, f"{split}.fasta")
        to_write = [records[r] for r, c in mapping.items() if c in clset]
        split_counts[split] = len(to_write)
        SeqIO.write(to_write, out_fa, "fasta")
        logger.info(f"Wrote {len(to_write)} sequences to {split} split")

    if cleanup and os.path.exists(mmseqs_dir):
        shutil.rmtree(mmseqs_dir)
        
    return split_counts
