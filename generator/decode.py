import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys

# Global variables for model instances
_tokenizer = None
_decoder = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_models(model_name="facebook/esm2_t33_650M_UR50D"):
    """Initialize the tokenizer and raygun decoder models"""
    global _tokenizer, _decoder, _device
    
    if _tokenizer is None or _decoder is None:
        print(f"Initializing models on {_device}...")
        # Load ESM-2 tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load raygun decoder
        try:
            _, _decoder, _ = torch.hub.load('rohitsinghlab/raygun', 'pretrained_uniref50_95000_750M')
            _decoder = _decoder.to(_device)
            _decoder.eval()
            print("Successfully loaded raygun decoder")
        except Exception as e:
            print(f"Error loading raygun decoder: {e}")
            raise

def decode_latents(esm_embeddings):
    """
    Decode ESM-2 embeddings back to amino acid sequences using raygun decoder.
    
    Args:
        esm_embeddings: Tensor of shape [batch_size, seq_len, emb_dim]
        
    Returns:
        List of decoded amino acid sequences
    """
    global _tokenizer, _decoder, _device
    
    # Initialize models if needed
    initialize_models()
    
    # Ensure embeddings are on the correct device
    if not isinstance(esm_embeddings, torch.Tensor):
        raise TypeError("Expected esm_embeddings to be a torch.Tensor")
    
    if esm_embeddings.device != _device:
        esm_embeddings = esm_embeddings.to(_device)
    
    try:
        with torch.no_grad():
            # Get token logits from decoder
            logits = _decoder(esm_embeddings)
            
            # Get predicted token IDs (argmax)
            predicted_token_ids = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
            
            # Convert to amino acid sequences
            sequences = []
            for ids in predicted_token_ids:
                # Convert IDs to amino acid sequence
                seq = _tokenizer.decode(ids).replace(" ", "")
                sequences.append(seq)
            
            return sequences
    except Exception as e:
        print(f"Error in decode_latents: {e}")
        raise