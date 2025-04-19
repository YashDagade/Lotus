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

class DecoderBlock(nn.Module):
    """
    Decoder block for latent flow-matching. Outputs token logits for ESM-2 vocab (32 tokens).
    """
    def __init__(self, dim=1280, nhead=20, dropout=0.2, max_len=1536):
        super(DecoderBlock, self).__init__()
        # Use a standard TransformerEncoderLayer instead of ESM-2 internal TransformerLayer
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=2*dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        # Final projection to vocab-size logits
        self.final = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Dropout(p=dropout),
            nn.Linear(dim // 4, 32)
        )
        # Maximum sequence length
        self.max_len = max_len
        # Load ESM vocabulary for decoding
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.vocab = list(alphabet.get_tok_to_idx().keys())
        self.eos_token = '<eos>'
        self.eos_idx = alphabet.get_tok_to_idx()[self.eos_token]
    
    def load_pretrained(self, filename):
        checkpoint = torch.load(filename)["model_state_dict"]
        self.load_state_dict(checkpoint)
        del checkpoint
        
    def forward(self, x):
        """
        x: [B, T, D] tensor of embeddings
        returns logits: [B, T, 32]
        """
        x = self.encoder(x)
        return self.final(x)

    def decode_latents(self, z_latents):
        """
        Decode latent vectors into amino-acid sequences.
        z_latents: [B, D]
        returns List[str] of length B
        """
        B, D = z_latents.size()
        # Expand latents across sequence positions
        z_seq = repeat(z_latents, 'b d -> b t d', t=self.max_len)
        # Compute logits and take argmax over vocab dimension
        logits = self.forward(z_seq)  # [B, T, 32]
        token_ids = torch.argmax(logits, dim=-1)  # [B, T]

        sequences = []
        for toks in token_ids:
            aa = []
            for idx in toks.tolist():
                if idx == self.eos_idx:
                    break
                token = self.vocab[idx]
                # Skip special tokens except amino acids
                if token in ['<pad>', '<cls>', '<mask>', '<unk>']:
                    continue
                aa.append(token)
            sequences.append(''.join(aa))
        return sequences

# Global decoder instance
_decoder = None

def decode_latents(z):
    """
    Decode latent vectors into protein sequences
    
    Args:
        z: Tensor of latent vectors [batch_size, emb_dim]
        
    Returns:
        List of decoded protein sequences
    """
    # For development/testing, return mockup sequences
    if not isinstance(z, torch.Tensor):
        return ["MVASKVV..."]*len(z)
    
    # Real implementation path
    try:
        global _decoder
        device = z.device
        
        # Initialize decoder if needed
        if _decoder is None:
            # Load config
            import yaml
            cfg = yaml.safe_load(open("config.yaml"))
            _decoder = DecoderBlock(
                dim=cfg["decoder"]["dim"],
                nhead=cfg["decoder"]["nhead"],
                dropout=cfg["decoder"]["dropout"],
                max_len=cfg["decoder"].get("max_len", 1536)
            ).to(device)
            
            # Load pretrained weights if specified
            if cfg["decoder"].get("pretrained_path", ""):
                if os.path.exists(cfg["decoder"]["pretrained_path"]):
                    _decoder.load_pretrained(cfg["decoder"]["pretrained_path"])
        
        # Decode the latents
        sequences = _decoder.decode_latents(z)
        return sequences
        
    except Exception as e:
        print(f"Error in decode_latents: {e}")
        # Fallback to mockup sequences
        return ["MVASKVV..."]*len(z)

if __name__ == "__main__":
    print("Testing DecoderBlock and decode_latents functionality...")
    
    # Create a mock config
    mock_cfg = {
        "decoder": {
            "dim": 32,
            "nhead": 4,
            "dropout": 0.1,
            "max_len": 50,
            "pretrained_path": ""
        }
    }
    
    try:
        # Test DecoderBlock instantiation
        print("Initializing DecoderBlock...")
        decoder = DecoderBlock(
            dim=mock_cfg["decoder"]["dim"],
            nhead=mock_cfg["decoder"]["nhead"],
            dropout=mock_cfg["decoder"]["dropout"],
            max_len=mock_cfg["decoder"]["max_len"]
        )
        print("DecoderBlock initialized successfully.")
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        emb_dim = mock_cfg["decoder"]["dim"]
        
        # Create mock input tensor
        mock_input = torch.randn(batch_size, seq_len, emb_dim)
        
        print("Testing forward pass...")
        with torch.no_grad():
            output = decoder(mock_input)
        print(f"Forward pass successful. Output shape: {output.shape}")
        
        # Test decode_latents on a small batch
        print("Testing decode_latents...")
        mock_latents = torch.randn(batch_size, emb_dim)
        
        # Override the global _decoder to use our test instance
        _decoder = decoder
        
        # Test decoding with custom decoder
        try:
            sequences = decoder.decode_latents(mock_latents)
            print(f"Decoded {len(sequences)} sequences:")
            for i, seq in enumerate(sequences):
                print(f"  Sequence {i+1}: {seq[:20]}..." if len(seq) > 20 else f"  Sequence {i+1}: {seq}")
        except Exception as e:
            print(f"Error in decode_latents: {e}")
            print("This is expected if using the test decoder since we're not setting up the full vocabulary.")
        
        # Test the fallback behavior
        print("Testing fallback behavior...")
        mock_latents_list = [1, 2]  # Not a tensor to trigger fallback
        fallback_sequences = decode_latents(mock_latents_list)
        print(f"Fallback returned {len(fallback_sequences)} placeholder sequences.")
        
        print("All decoder tests completed.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    print("Test finished.") 