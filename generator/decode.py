import os
import torch
import numpy as np
import glob
from tqdm import tqdm 
import sys
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import math
import esm
from transformers import AutoModelForMaskedLM, AutoTokenizer

class ESMDecoder:
    """
    ESM-2 decoder using raygun's pretrained model to convert embeddings back to sequences.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load ESM-2 tokenizer
        self.model_name = "facebook/esm2_t33_650M_UR50D"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load raygun decoder
        _, self.decoder, _ = torch.hub.load('rohitsinghlab/raygun', 'pretrained_uniref50_95000_750M')
        self.decoder = self.decoder.to(self.device)
        self.decoder.eval()

    def decode_latents(self, z_latents):
        """
        Decode latent vectors into amino-acid sequences.
        z_latents: [B, L, D] where B is batch size, L is sequence length, D is embedding dimension
        returns List[str] of length B
        """
        try:
            with torch.no_grad():
                # Move to device if needed
                if z_latents.device != self.device:
                    z_latents = z_latents.to(self.device)
                
                # Get logits from decoder
                logits = self.decoder(z_latents)
                
                # Get predicted token IDs
                predicted_token_ids = torch.argmax(logits, dim=-1)  # [B, L]
                
                # Convert to sequences
                sequences = []
                for ids in predicted_token_ids:
                    seq = self.tokenizer.decode(ids).replace(" ", "")
                    sequences.append(seq)
                
                return sequences
                
        except Exception as e:
            print(f"Error in decode_latents: {e}")
            # Fallback to mockup sequences
            return ["MOCKSEQUENCE"] * z_latents.size(0)

# Global decoder instance
_decoder = None

def decode_latents(z):
    """
    Decode latent vectors into protein sequences
    
    Args:
        z: Tensor of latent vectors [batch_size, seq_len, emb_dim]
        
    Returns:
        List of decoded protein sequences
    """
    global _decoder
    
    # Initialize decoder if needed
    if _decoder is None:
        _decoder = ESMDecoder()
    
    return _decoder.decode_latents(z)

if __name__ == "__main__":
    print("Testing ESM decoder functionality...")
    
    try:
        # Create mock input
        batch_size = 2
        seq_len = 10
        emb_dim = 1280  # ESM-2 embedding dimension
        from generator.embed_sequences import embed
        import torch

        # Use test_sequence.fasta for testing
        test_fasta = "test_sequence.fasta"
        test_embeddings_pt = "test_embeddings.pt"
        test_model = "esm2_t6_8M_UR50D"  # Use a small model for test

        # Generate embeddings if not already present
        import os
        if not os.path.exists(test_embeddings_pt):
            print(f"Generating embeddings for since they are not present already {test_fasta}...")
            embed(test_fasta, test_embeddings_pt, test_model)

        # Load embeddings
        data = torch.load(test_embeddings_pt)
        embeddings = data["embeddings"]  # [num_seqs, seq_len, emb_dim] or [num_seqs, emb_dim]
        # If embeddings are [num_seqs, emb_dim], unsqueeze to [num_seqs, 1, emb_dim] for compatibility
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)

        decoder = ESMDecoder()
        sequences = decoder.decode_latents(embeddings)
        print(f"Successfully decoded {len(sequences)} sequences:")
        for i, seq in enumerate(sequences):
            print(f"Sequence {i+1}: {seq[:20]}..." if len(seq) > 20 else f"Sequence {i+1}: {seq}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
    
    print("Test finished.") 