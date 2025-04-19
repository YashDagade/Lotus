import torch, esm
from Bio import SeqIO
from tqdm import tqdm
import argparse
import os
import sys

def embed(fasta, out_pt, model_name):
    # Updated model loading to handle attribute error
    try:
        # Use getattr to access the model from esm.pretrained if it exists
        model_fn = getattr(esm.pretrained, model_name, None)
        if model_fn is not None:
            model, alphabet = model_fn()
        elif model_name == "esm2_t33_650M_UR50D":
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif model_name == "esm2_t6_8M_UR50D":
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        else:
            raise ValueError(f"Model {model_name} not supported or not found")
    except Exception as e:
        raise RuntimeError(f"Failed to load ESM model {model_name}: {e}")
    
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
    # Use argparse if provided with args
    if len(sys.argv) > 1:
        p=argparse.ArgumentParser()
        p.add_argument("--fasta"); p.add_argument("--out"); p.add_argument("--model")
        args=p.parse_args()
        embed(args.fasta, args.out, args.model)
    else:
        # Test embedding functionality
        print("Testing embed_sequences.py...")
        
        # Check if test_sequence.fasta exists
        test_fasta = "test_sequence.fasta"
        if not os.path.exists(test_fasta):
            print(f"Error: {test_fasta} not found. Please create it first.")
            sys.exit(1)
        
        test_output = "test_embeddings.pt"
        test_model = "esm2_t6_8M_UR50D"  # Use smaller model for testing
        
        print(f"Embedding {test_fasta} with {test_model}...")
        try:
            embed(test_fasta, test_output, test_model)
            
            # Verify the output file was created and contains expected data
            if os.path.exists(test_output):
                data = torch.load(test_output)
                if "ids" in data and "embeddings" in data:
                    print(f"Successfully created embeddings of shape: {data['embeddings'].shape}")
                    print(f"IDs: {data['ids']}")
                else:
                    print("Output file is missing expected keys.")
            else:
                print(f"Failed to create {test_output}")
        except Exception as e:
            print(f"Error during testing: {e}")
        
        print("Test finished.") 