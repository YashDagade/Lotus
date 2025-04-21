import subprocess, pandas as pd, random, os, shutil
from Bio import SeqIO

def split_clusters(
    fasta, out_dir,
    id_min=0.3, cov=0.8,
    frac_train=0.8, frac_val=0.1,
    seed=None, cleanup=False
):
    if seed is not None:
        random.seed(seed)

    # ensure output dirs
    os.makedirs(out_dir, exist_ok=True)
    mmseqs_dir = os.path.join(out_dir, "mmseqs2")
    os.makedirs(mmseqs_dir, exist_ok=True)

    seqdb = os.path.join(mmseqs_dir, "seqDB")
    clustdb = os.path.join(mmseqs_dir, "clusterDB")
    tmp = os.path.join(mmseqs_dir, "tmp")
    tsv = os.path.join(mmseqs_dir, "clusters.tsv")

    # run MMseqs2
    for cmd in [
      ["mmseqs", "createdb", fasta, seqdb],
      ["mmseqs", "cluster", seqdb, clustdb, tmp,
       "--min-seq-id", str(id_min), "-c", str(cov)],
      ["mmseqs", "createtsv", seqdb, seqdb, clustdb, tsv],
    ]:
        subprocess.run(cmd, check=True)

    df = pd.read_csv(tsv, sep="\t", names=["seq","seq2","cluster"])
    clusters = df.cluster.unique().tolist()
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

    for split, clset in splits.items():
        out_fa = os.path.join(out_dir, f"{split}.fasta")
        to_write = (records[r] for r, c in mapping.items() if c in clset)
        SeqIO.write(to_write, out_fa, "fasta")

    if cleanup and os.path.exists(mmseqs_dir):
        shutil.rmtree(mmseqs_dir)
