#!/usr/bin/env python
# Split by MMseqs2 clusters into train/val/test

import subprocess, pandas as pd
import random, os, sys

def split_clusters(fasta, out_dir, id_min=0.3, cov=0.8):
    os.makedirs(out_dir, exist_ok=True)
    # Create mmseqs2 output directory
    mmseqs_dir = os.path.join("outs", "mmseqs2")
    os.makedirs(mmseqs_dir, exist_ok=True)
    
    # Use mmseqs_dir for all temporary and cluster files
    seqdb_path = os.path.join(mmseqs_dir, "seqDB")
    clusterdb_path = os.path.join(mmseqs_dir, "clusterDB")
    tmp_path = os.path.join(mmseqs_dir, "tmp")
    clusters_tsv = os.path.join(mmseqs_dir, "clusters.tsv")
    
    # create DB & cluster
    subprocess.run(f"mmseqs createdb {fasta} {seqdb_path}", shell=True, check=True)
    subprocess.run(f"mmseqs cluster {seqdb_path} {clusterdb_path} {tmp_path} --min-seq-id {id_min} -c {cov}", shell=True, check=True)
    subprocess.run(f"mmseqs createtsv {seqdb_path} {seqdb_path} {clusterdb_path} {clusters_tsv}", shell=True, check=True)

    df = pd.read_csv(clusters_tsv, sep="\t", names=["seq","seq2","cluster"])
    clusters = df.cluster.unique().tolist()
    random.shuffle(clusters)
    n = len(clusters)
    train_c = set(clusters[:int(0.8*n)])
    val_c   = set(clusters[int(0.8*n):int(0.9*n)])
    test_c  = set(clusters[int(0.9*n):])
    splits = {"train": train_c, "val": val_c, "test": test_c}

    # write FASTA splits
    from Bio import SeqIO
    recs = SeqIO.index(fasta, "fasta")
    # map seq->cluster
    mapping = df.set_index("seq")["cluster"].to_dict()
    for split, clset in splits.items():
        with open(f"{out_dir}/{split}.fasta","w") as fw:
            for seqid, clu in mapping.items():
                if clu in clset:
                    SeqIO.write(recs[seqid], fw, "fasta")
                    
    # Clean up temporary files if needed
    if os.path.exists(tmp_path):
        subprocess.run(f"rm -rf {tmp_path}", shell=True)

if __name__=="__main__":
    # Test the split_clusters function
    print("Testing mmseqs_split.py...")
    
    # Check if test_sequence.fasta exists
    test_fasta = "test_sequence.fasta"
    if not os.path.exists(test_fasta):
        print(f"Error: {test_fasta} not found. Please create it first.")
        sys.exit(1)
    
    # Create a test output directory
    test_out_dir = "test_splits"
    
    try:
        # For testing, use higher identity threshold since we only have one sequence
        print(f"Splitting {test_fasta} into {test_out_dir}...")
        split_clusters(test_fasta, test_out_dir, id_min=0.9, cov=0.9)
        
        # Check if splits were created
        split_files = [f"{test_out_dir}/{split}.fasta" for split in ["train", "val", "test"]]
        for file in split_files:
            if os.path.exists(file):
                print(f"Successfully created {file}")
            else:
                print(f"Failed to create {file}")
                
        print("Test completed.")
    except Exception as e:
        print(f"Error during testing: {e}")
        
    # Clean up test files
    mmseqs_dir = os.path.join("outs", "mmseqs2")
    if os.path.exists(mmseqs_dir):
        subprocess.run(f"rm -rf {mmseqs_dir}", shell=True)
        print("Cleaned up temporary files.")
    
    print("Test finished.") 