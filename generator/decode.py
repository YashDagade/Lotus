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
from esm.model.esm2 import TransformerLayer
import esm

# ESM2 alphabet for decoding
from esm.data import Alphabet

class DecoderBlock(nn.Module):
    """
    Decoder block for latent flow-matching. Outputs token logits for ESM-2 vocab (32 tokens).
    """
    def __init__(self, dim=1280, nhead=20, dropout=0.2, max_len=1536):
        super(DecoderBlock, self).__init__()
        # Transformer layer from ESM-2
        self.encoder = TransformerLayer(
            embed_dim=dim,
            ffn_embed_dim=2*dim,
            attention_heads=nhead,
            use_rotary_embeddings=True
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
        x, _ = self.encoder(x)
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
    global _decoder
    
    # For development/testing, return mockup sequences
    if not isinstance(z, torch.Tensor):
        return ["MVASKVV..."]*len(z)
    
    # Real implementation path
    try:
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