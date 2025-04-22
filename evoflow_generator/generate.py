import torch
import os
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def generate_sequences(cfg, num_samples=100, output_dir=None, run_name=None):
    """
    Generate protein sequences using the fine-tuned EvoFlow model.
    
    Args:
        cfg: Configuration dictionary
        num_samples: Number of sequences to generate
        output_dir: Directory to save generated sequences
        run_name: Name for the generated files
        
    Returns:
        List of generated sequences
    """
    print(f"Generating {num_samples} sequences...")
    
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_checkpoint = cfg.get("model_checkpoint", "fredzzp/EvoFlow-650M-context-3070")
    print(f"Loading base model: {model_checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    
    # Load fine-tuned weights if available
    model_dir = cfg["evoflow"].get("model_dir", "models/evoflow")
    best_model_path = os.path.join(model_dir, "evoflow_best.pt")
    
    if os.path.exists(best_model_path):
        print(f"Loading fine-tuned weights from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Warning: Fine-tuned model not found at {best_model_path}. Using base model.")
    
    model.to(device)
    model.eval()
    
    # Get generation parameters
    max_length = cfg["evoflow"].get("max_length", 1024)
    temperature = cfg["evoflow"].get("temperature", 1.0)
    top_k = cfg["evoflow"].get("top_k", 50)
    top_p = cfg["evoflow"].get("top_p", 0.95)
    
    # Generate sequences
    generated_sequences = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Generating sequences"):
            # Start with the start token (CLS)
            input_ids = torch.tensor([[tokenizer.cls_token_id]]).to(device)
            attention_mask = torch.ones_like(input_ids)
            
            # Generate until we hit the end token or max length
            for _ in range(max_length - 2):  # Leave room for SEP token
                # Get model predictions
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Filter out special tokens (except SEP)
                special_tokens_mask = torch.ones_like(next_token_logits)
                special_tokens_mask[:, tokenizer.pad_token_id] = 0
                special_tokens_mask[:, tokenizer.cls_token_id] = 0
                special_tokens_mask[:, tokenizer.mask_token_id] = 0
                next_token_logits = next_token_logits * special_tokens_mask
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if we predict the SEP token
                if next_token.item() == tokenizer.sep_token_id:
                    break
                    
                # Add the predicted token to our sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            
            # Add the SEP token to end the sequence
            input_ids = torch.cat([input_ids, torch.tensor([[tokenizer.sep_token_id]]).to(device)], dim=1)
            
            # Decode the sequence
            sequence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            generated_sequences.append(sequence)
    
    # Save sequences if output_dir is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine file name
        if run_name is None:
            run_name = cfg["wandb"].get("name", "evoflow_samples")
        
        # Save as FASTA
        fasta_path = os.path.join(output_dir, f"{run_name}_samples.fasta")
        records = []
        
        for i, seq in enumerate(generated_sequences):
            record = SeqRecord(
                Seq(seq),
                id=f"{run_name}_sample_{i}",
                description=""
            )
            records.append(record)
        
        SeqIO.write(records, fasta_path, "fasta")
        print(f"Saved {len(generated_sequences)} sequences to {fasta_path}")
    
    return generated_sequences

if __name__ == "__main__":
    # Test generation
    import yaml
    
    try:
        cfg = yaml.safe_load(open("evoflow_config.yaml"))
        print("Loaded config for testing")
        
        # Generate a small number of sequences for testing
        sequences = generate_sequences(cfg, num_samples=2, output_dir="test_output")
        print(f"Generated {len(sequences)} test sequences")
        for i, seq in enumerate(sequences):
            print(f"Sequence {i+1}: {seq[:50]}...")
            
    except Exception as e:
        print(f"Test error: {e}") 