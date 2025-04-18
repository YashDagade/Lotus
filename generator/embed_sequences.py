import torch, esm
from Bio import SeqIO
from tqdm import tqdm
import argparse

def embed(fasta, out_pt, model_name):
    model, alphabet = esm.pretrained.__dict__[model_name]()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    recs = [(r.id, str(r.seq)) for r in SeqIO.parse(fasta,"fasta")]
    embs, ids = [], []
    for i in tqdm(range(0, len(recs), 8)):
        batch = recs[i:i+8]
        labels, seqs, toks = batch_converter(batch)
        with torch.no_grad():
            out = model(toks, repr_layers=[model.num_layers])
        rep = out["representations"][model.num_layers].mean(1)
        embs.append(rep.cpu()); ids += labels
    embs = torch.cat(embs,0)
    torch.save({"ids":ids,"embeddings":embs}, out_pt)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fasta"); p.add_argument("--out"); p.add_argument("--model")
    args=p.parse_args()
    embed(args.fasta, args.out, args.model) 