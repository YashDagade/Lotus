import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb
import yaml
import argparse
from tqdm import tqdm
import os
import numpy as np

from .decode import DecoderBlock

def train_decoder(cfg):
    """
    Train the decoder to convert latent embeddings to sequences.
    This is a simplified training script - in a real scenario, you'd need
    paired data of embeddings and their corresponding amino acid sequences.
    """
    # Load embeddings
    data = torch.load("dataset/cas9_embeddings.pt")
    embs = data["embeddings"]
    
    # For a proper implementation, you would need the corresponding sequences
    # and tokenize them using the ESM-2 vocabulary
    
    # This is a mock implementation that demonstrates the flow
    # In practice, you'd need:
    # 1. ESM-2 tokenized sequences aligned with embeddings
    # 2. A proper loss function for sequence prediction
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = DecoderBlock(
        dim=cfg["decoder"]["dim"],
        nhead=cfg["decoder"]["nhead"],
        dropout=cfg["decoder"]["dropout"],
        max_len=cfg["decoder"].get("max_len", 1536)
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        decoder.parameters(), 
        lr=cfg["decoder"].get("learning_rate", 1e-4)
    )
    
    # Create a dummy dataset with random targets
    # In a real application, you would use actual tokenized sequences
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, embeddings, max_len=100):
            self.embeddings = embeddings
            self.max_len = max_len
            
            # For demonstration - random target sequences
            # In a real implementation, these would be actual tokenized sequences
            self.targets = torch.randint(0, 32, (len(embeddings), max_len))
            
        def __len__(self):
            return len(self.embeddings)
            
        def __getitem__(self, idx):
            return self.embeddings[idx], self.targets[idx]
    
    # Create datasets and dataloaders
    # In a real scenario, you'd split by sequence clusters
    train_size = int(0.8 * len(embs))
    train_embs = embs[:train_size]
    val_embs = embs[train_size:]
    
    train_dataset = DummyDataset(train_embs)
    val_dataset = DummyDataset(val_embs)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg["decoder"].get("batch_size", 32),
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg["decoder"].get("batch_size", 32),
        shuffle=False
    )
    
    # Training loop
    num_epochs = cfg["decoder"].get("max_epochs", 100)
    best_val_loss = float('inf')
    
    print(f"Training decoder for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training
        decoder.train()
        train_loss = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for emb, target in train_loader:
                emb, target = emb.to(device), target.to(device)
                
                # Expand embeddings across sequence positions
                # In a real implementation, you might use a more sophisticated approach
                emb_expanded = emb.unsqueeze(1).expand(-1, train_dataset.max_len, -1)
                
                # Forward pass
                logits = decoder(emb_expanded)
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    target.reshape(-1)
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader)
        
        # Validation
        decoder.eval()
        val_loss = 0
        
        with torch.no_grad():
            for emb, target in val_loader:
                emb, target = emb.to(device), target.to(device)
                
                # Expand embeddings
                emb_expanded = emb.unsqueeze(1).expand(-1, train_dataset.max_len, -1)
                
                # Forward pass
                logits = decoder(emb_expanded)
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    target.reshape(-1)
                )
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Log to wandb
        wandb.log({
            "decoder/train_loss": train_loss,
            "decoder/val_loss": val_loss,
            "decoder/epoch": epoch
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state_dict": decoder.state_dict()}, 
                os.path.join("models", "best_decoder.pt")
            )
            print(f"Saved best model with validation loss {val_loss:.4f}")
    
    print("Decoder training complete!")
    return decoder

if __name__ == "__main__":
    # Check if running as main script or testing
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--test":
        print("Running test mode for decoder training...")
        
        # Create mock config
        mock_cfg = {
            "decoder": {
                "dim": 32,
                "nhead": 4,
                "dropout": 0.1,
                "max_len": 50,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "max_epochs": 2  # Very short for testing
            },
            "wandb": {
                "project": "test"
            }
        }
        
        try:
            # Create mock embeddings
            mock_embs = torch.randn(10, mock_cfg["decoder"]["dim"])
            mock_data = {"embeddings": mock_embs, "ids": ["seq1", "seq2", "seq3", "seq4", "seq5", "seq6", "seq7", "seq8", "seq9", "seq10"]}
            os.makedirs("dataset", exist_ok=True)
            torch.save(mock_data, "dataset/mock_embeddings.pt")
            
            # Mock wandb
            class MockWandb:
                def log(self, data):
                    print(f"W&B log: {data}")
                
                def init(self, **kwargs):
                    print(f"W&B init: {kwargs}")
                    return self
                
                def finish(self):
                    print("W&B finish")
            
            # Store original wandb and replace with mock
            original_wandb = wandb
            wandb = MockWandb()
            
            # Mock data loading
            original_torch_load = torch.load
            
            def mock_torch_load(path):
                if "mock_embeddings.pt" in path or "cas9_embeddings.pt" in path:
                    return mock_data
                else:
                    return original_torch_load(path)
            
            torch.load = mock_torch_load
            
            # Create models directory for testing
            os.makedirs("models", exist_ok=True)
            
            # Run test training for 1-2 epochs
            print("Starting mock decoder training...")
            
            # Monkey patch tqdm to move faster
            original_tqdm = tqdm
            tqdm = lambda x, **kwargs: x
            
            # Run training
            decoder = train_decoder(mock_cfg)
            print("Decoder training completed.")
            
            # Test the trained model
            print("\nTesting trained decoder...")
            test_input = torch.randn(2, 10, mock_cfg["decoder"]["dim"])
            with torch.no_grad():
                output = decoder(test_input)
            print(f"Decoder output shape: {output.shape}")
            
            # Restore original functions
            wandb = original_wandb
            torch.load = original_torch_load
            tqdm = original_tqdm
            
            print("\nAll decoder training tests completed.")
            
        except Exception as e:
            print(f"Error during testing: {e}")
            # Restore original functions
            if 'original_wandb' in locals():
                wandb = original_wandb
            if 'original_torch_load' in locals():
                torch.load = original_torch_load
            if 'original_tqdm' in locals():
                tqdm = original_tqdm
        
        print("Test finished.")
        
    else:
        # Normal execution
        parser = argparse.ArgumentParser(description="Train a decoder for latent embeddings")
        parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
        args = parser.parse_args()
        
        # Load config
        cfg = yaml.safe_load(open(args.config))
        
        # Initialize wandb
        wandb.init(
            entity=cfg["wandb"]["entity"],
            project=cfg["wandb"]["project"],
            name=f"{cfg['wandb'].get('name', 'decoder_training')}",
            config=cfg
        )
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Train decoder
        decoder = train_decoder(cfg)
        
        # Finish wandb
        wandb.finish() 