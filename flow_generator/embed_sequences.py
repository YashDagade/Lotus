import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from Bio import SeqIO
from tqdm import tqdm
import argparse
import os
import sys

def embed(fasta, out_pt, model_name="facebook/esm2_t33_650M_UR50D"):
    """
    Generate embeddings for sequences in a FASTA file using the ESM-2 model
    from HuggingFace transformers.
    
    Args:
        fasta: Path to the FASTA file
        out_pt: Path to save the embeddings
        model_name: Name of the ESM-2 model to use
    """
    print(f"Loading model {model_name}...")
    # Load ESM model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)
    
    # Load sequences from FASTA
    records = list(SeqIO.parse(fasta, "fasta"))
    print(f"Processing {len(records)} sequences...")
    
    # Process sequences in batches
    all_embeddings = []
    all_ids = []
    batch_size = 1  # Process one sequence at a time due to potentially long sequences
    
    for i in tqdm(range(0, len(records), batch_size)):
        batch_records = records[i:i+batch_size]
        
        for record in batch_records:
            seq_id = record.id
            sequence = str(record.seq)
            
            # Skip sequences that are too long for the model
            if len(sequence) > model.config.max_position_embeddings - 2:  # Account for special tokens
                print(f"Skipping {seq_id} - too long ({len(sequence)} > {model.config.max_position_embeddings-2})")
                continue
                
            # Tokenize and move to device
            inputs = tokenizer(sequence, return_tensors="pt")
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Get last layer embeddings, remove special tokens (first and last)
                esm_embeddings = outputs.hidden_states[-1]
                esm_embeddings = esm_embeddings[:, 1:-1, :]  # [batch, seq_len, emb_dim]
                
                # Save embeddings and ID
                all_embeddings.append(esm_embeddings.cpu())
                all_ids.append(seq_id)
    
    # Concatenate all embeddings
    if all_embeddings:
        embeddings = torch.cat(all_embeddings, dim=0)
        # Save embeddings and IDs
        print(f"Saving embeddings with shape {embeddings.shape} to {out_pt}")
        torch.save({"ids": all_ids, "embeddings": embeddings}, out_pt)
        print(f"Successfully saved embeddings for {len(all_ids)} sequences")
    else:
        print("No embeddings were generated. Check if all sequences were skipped.")

if __name__ == "__main__":
    # Use argparse if provided with args
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", help="Path to FASTA file")
    p.add_argument("--out", help="Path to output file")
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D", help="ESM model name")
    args = p.parse_args()
    embed(args.fasta, args.out, args.model)
    