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

class ProteinDataset(Dataset):
    def __init__(self, fasta_file, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []
        
        # Load sequences from FASTA file
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq)
            if len(seq) <= self.max_length - 2:  # Account for special tokens
                self.sequences.append(seq)
        
        print(f"Loaded {len(self.sequences)} sequences from {fasta_file}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Tokenize the sequence
        encoding = self.tokenizer(seq, 
                                 truncation=True,
                                 max_length=self.max_length,
                                 padding="max_length",
                                 return_tensors="pt")
        
        # Remove batch dimension that the tokenizer adds
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # For masked language modeling
        input_ids = item["input_ids"].clone()
        labels = input_ids.clone()
        
        # Create random array of floats with equal dimensions to input_ids
        rand = torch.rand(input_ids.shape)
        
        # Create mask array
        mask_arr = (rand < 0.15) * (input_ids != self.tokenizer.cls_token_id) * (input_ids != self.tokenizer.sep_token_id) * (input_ids != self.tokenizer.pad_token_id)
        
        # Get indices of masked tokens
        selection = torch.flatten(mask_arr.nonzero()).tolist()
        
        # Mask input_ids
        input_ids[selection] = self.tokenizer.mask_token_id
        
        item["input_ids"] = input_ids
        item["labels"] = labels
        
        return item

def train_evoflow(cfg):
    print("Starting EvoFlow fine-tuning...")
    
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
    model.to(device)
    
    # Set up datasets
    train_file = os.path.join(cfg["cluster"]["splits_dir"], "train.fasta")
    val_file = os.path.join(cfg["cluster"]["splits_dir"], "val.fasta")
    
    train_dataset = ProteinDataset(train_file, tokenizer, max_length=cfg["evoflow"].get("max_length", 1024))
    val_dataset = ProteinDataset(val_file, tokenizer, max_length=cfg["evoflow"].get("max_length", 1024))
    
    # Set up data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["evoflow"].get("batch_size", 8),
        shuffle=True,
        num_workers=cfg["evoflow"].get("num_workers", 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["evoflow"].get("batch_size", 8),
        shuffle=False,
        num_workers=cfg["evoflow"].get("num_workers", 4)
    )
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["evoflow"].get("learning_rate", 5e-5),
        weight_decay=cfg["evoflow"].get("weight_decay", 0.01)
    )
    
    total_steps = len(train_loader) * cfg["evoflow"].get("epochs", 30)
    warmup_steps = int(total_steps * cfg["evoflow"].get("warmup_ratio", 0.1))
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Set up training variables
    best_val_loss = float('inf')
    patience = cfg["evoflow"].get("patience", 3)
    patience_counter = 0
    
    # Create directory for saving models
    model_dir = cfg["evoflow"].get("model_dir", "models/evoflow")
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(cfg["evoflow"].get("epochs", 30)):
        print(f"Starting epoch {epoch+1}/{cfg['evoflow'].get('epochs', 30)}")
        
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
                wandb.log({
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
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "epoch": epoch + 1,
        })
        
        print(f"Epoch {epoch+1} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(model_dir, f"evoflow_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        
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
            
            print(f"New best model saved at {best_model_path}")
            wandb.log({"val/best_loss": best_val_loss})
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