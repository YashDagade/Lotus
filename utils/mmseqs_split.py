#!/usr/bin/env python
# Split by MMseqs2 clusters into train/val/test

import subprocess, pandas as pd
import random, os, sys

def split_clusters(fasta, out_dir, id_min=0.3, cov=0.8):
    os.makedirs(out_dir, exist_ok=True)
    # create DB & cluster
    subprocess.run(f"mmseqs createdb {fasta} seqDB", shell=True, check=True)
    subprocess.run(f"mmseqs cluster seqDB clusterDB tmp --min-seq-id {id_min} -c {cov}", shell=True, check=True)
    subprocess.run("mmseqs createtsv seqDB seqDB clusterDB clusters.tsv", shell=True, check=True)

    df = pd.read_csv("clusters.tsv", sep="\t", names=["seq","seq2","cluster"])
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

if __name__=="__main__":
    split_clusters(sys.argv[1], sys.argv[2]) 