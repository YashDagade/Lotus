import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup
import wandb
from pathlib import Path
from Bio import SeqIO
import random

# Helper function for safer wandb logging
def log_wandb(metrics):
    try:
        wandb.log(metrics)
    except Exception as e:
        print(f"Warning: Failed to log to wandb: {e}")

class ProteinDataset(Dataset):
    def __init__(self, fasta_file, tokenizer, max_length=1024, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.sequences = []
        
        # Check if file exists
        if not os.path.exists(fasta_file):
            print(f"Warning: FASTA file not found: {fasta_file}")
            return
            
        # Load sequences from FASTA file
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                seq = str(record.seq)
                # Filter out invalid sequences (too short or non-standard amino acids)
                if len(seq) <= self.max_length - 2 and len(seq) >= 10:  # Account for special tokens
                    # Check for standard amino acids (optional)
                    if all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq):
                        self.sequences.append(seq)
            
            print(f"Loaded {len(self.sequences)} valid sequences from {fasta_file}")
            
            # Print sequence length statistics if we have sequences
            if self.sequences:
                lengths = [len(seq) for seq in self.sequences]
                print(f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
                
        except Exception as e:
            print(f"Error loading sequences from {fasta_file}: {e}")
            import traceback
            traceback.print_exc()
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Tokenize the sequence
        # Make sure special tokens are added
        encoding = self.tokenizer(
            seq, 
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Remove batch dimension that the tokenizer adds
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # For masked language modeling
        input_ids = item["input_ids"].clone()
        labels = input_ids.clone()
        
        # Create random array of floats with equal dimensions to input_ids
        rand = torch.rand(input_ids.shape)
        
        # Create mask array - avoid masking special tokens
        cls_token_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else 0
        sep_token_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else 2
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1
        
        mask_arr = (rand < self.mask_prob) & (input_ids != cls_token_id) & (input_ids != sep_token_id) & (input_ids != pad_token_id)
        
        # Get indices of masked tokens
        selection = torch.flatten(mask_arr.nonzero()).tolist()
        
        # Mask input_ids with the mask token
        mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else 32
        input_ids[selection] = mask_token_id
        
        item["input_ids"] = input_ids
        item["labels"] = labels
        
        return item

def train_evoflow(cfg):
    print("Starting EvoFlow fine-tuning...")
    
    # Check if W&B is active
    wandb_enabled = wandb.run is not None
    
    # Set up random seeds for reproducibility
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
    print(f"Loading model: {model_checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    
    # Print model details
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check if tokenizer special tokens are properly set
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.mask_token_id is None:
        print("Warning: Some special tokens are not properly set in the tokenizer")
        print(f"CLS token: {tokenizer.cls_token_id}")
        print(f"SEP token: {tokenizer.sep_token_id}")
        print(f"MASK token: {tokenizer.mask_token_id}")
        print(f"PAD token: {tokenizer.pad_token_id}")
        
        # Use defaults from ESM model if needed
        if tokenizer.cls_token_id is None and hasattr(tokenizer, "cls_token"):
            tokenizer.cls_token_id = 0  # <cls>
        if tokenizer.sep_token_id is None and hasattr(tokenizer, "sep_token"):
            tokenizer.sep_token_id = 2  # <eos>
        if tokenizer.mask_token_id is None and hasattr(tokenizer, "mask_token"):
            tokenizer.mask_token_id = 32  # <mask>
        if tokenizer.pad_token_id is None and hasattr(tokenizer, "pad_token"):
            tokenizer.pad_token_id = 1  # <pad>
            
        print("Updated tokenizer special tokens:")
        print(f"CLS token: {tokenizer.cls_token_id}")
        print(f"SEP token: {tokenizer.sep_token_id}")
        print(f"MASK token: {tokenizer.mask_token_id}")
        print(f"PAD token: {tokenizer.pad_token_id}")
    
    model.to(device)
    
    # Set up datasets
    train_file = os.path.join(cfg["cluster"]["splits_dir"], "train.fasta")
    val_file = os.path.join(cfg["cluster"]["splits_dir"], "val.fasta")
    
    print(f"Loading training data from {train_file}")
    train_dataset = ProteinDataset(train_file, tokenizer, max_length=cfg["evoflow"].get("max_length", 1024))
    print(f"Loading validation data from {val_file}")
    val_dataset = ProteinDataset(val_file, tokenizer, max_length=cfg["evoflow"].get("max_length", 1024))
    
    # Log dataset info
    print(f"Training dataset: {len(train_dataset)} sequences")
    print(f"Validation dataset: {len(val_dataset)} sequences")
    
    if len(train_dataset) == 0:
        print("Error: No training sequences found!")
        return None
    
    if len(val_dataset) == 0:
        print("Error: No validation sequences found!")
        return None
    
    # Get a sample to inspect dimensions
    sample = train_dataset[0]
    input_shape = sample["input_ids"].shape
    print(f"Input shape: {input_shape} (sequence length)")
    
    # Set up data loaders
    batch_size = cfg["evoflow"].get("batch_size", 8)
    print(f"Using batch size: {batch_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["evoflow"].get("num_workers", 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["evoflow"].get("num_workers", 4)
    )
    
    # Log dataloader info
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Set up optimizer and scheduler
    learning_rate = cfg["evoflow"].get("learning_rate", 5e-5)
    print(f"Learning rate: {learning_rate}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=cfg["evoflow"].get("weight_decay", 0.01)
    )
    
    total_steps = len(train_loader) * cfg["evoflow"].get("epochs", 30)
    warmup_steps = int(total_steps * cfg["evoflow"].get("warmup_ratio", 0.1))
    print(f"Training steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Set up training variables
    best_val_loss = float('inf')
    epochs = cfg["evoflow"].get("epochs", 30)
    patience = cfg["evoflow"].get("patience", 5)
    print(f"Training for {epochs} epochs with patience {patience}")
    patience_counter = 0
    
    # Create directory for saving models
    model_dir = cfg["evoflow"].get("model_dir", "models/evoflow")
    os.makedirs(model_dir, exist_ok=True)
    print(f"Models will be saved to: {model_dir}")
    
    # Training loop
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["evoflow"].get("max_grad_norm", 1.0))
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            train_steps += 1
            
            # Log batch metrics
            if train_steps % cfg["evoflow"].get("log_interval", 10) == 0:
                if wandb_enabled:
                    log_wandb({
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (train_steps / len(train_loader)),
                    })
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Update metrics
                val_loss += loss.item()
                val_steps += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_steps
        
        # Log epoch metrics
        print(f"Epoch {epoch+1} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")
        
        if wandb_enabled:
            log_wandb({
                "train/epoch_loss": avg_train_loss,
                "val/epoch_loss": avg_val_loss,
                "epoch": epoch + 1,
            })
        
        # Save checkpoint
        checkpoint_path = os.path.join(model_dir, f"evoflow_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(model_dir, "evoflow_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_model_path)
            
            print(f"New best model saved at {best_model_path} with val_loss: {best_val_loss:.4f}")
            if wandb_enabled:
                log_wandb({"val/best_loss": best_val_loss})
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save final model
    final_model_path = os.path.join(model_dir, "evoflow_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    
    print(f"Training completed. Final model saved at {final_model_path}")
    
    # Return the model
    return model

if __name__ == "__main__":
    # Simple test
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    model_checkpoint = "fredzzp/EvoFlow-650M-context-3070"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Create a mock config
    mock_cfg = {
        "evoflow": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 5e-5
        },
        "cluster": {
            "splits_dir": "dataset/splits"
        }
    }
    
    try:
        # Test with a small dataset if available
        if os.path.exists("dataset/splits/train.fasta"):
            dataset = ProteinDataset("dataset/splits/train.fasta", tokenizer, max_length=512)
            print(f"Test dataset created with {len(dataset)} items")
            if len(dataset) > 0:
                item = dataset[0]
                print(f"Sample input shape: {item['input_ids'].shape}")
    except Exception as e:
        print(f"Test error: {e}") 