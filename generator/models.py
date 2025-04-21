import torch, torch.nn as nn
import numpy as np
from generator.solver import ODESolver

class FlowMatchingNet(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(emb_dim+hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.emb_dim = emb_dim
        self.solver = None  # Will be initialized in get_solver
        
    def forward(self, z, t):
        """
        z: [B, L, D] where B is batch size, L is sequence length, D is embedding dim
        t: [B] time values
        Returns: [B, L, D] vector field
        """
        B, L, D = z.shape
        # Expand time embeddings to match sequence length
        te = self.time_mlp(t.unsqueeze(1))  # [B, H]
        te = te.unsqueeze(1).expand(-1, L, -1)  # [B, L, H]
        
        # Process each position independently
        z_flat = z.view(B*L, D)  # [B*L, D]
        te_flat = te.view(B*L, -1)  # [B*L, H]
        v_flat = self.net(torch.cat([z_flat, te_flat], dim=1))  # [B*L, D]
        
        # Reshape back to sequence form
        v = v_flat.view(B, L, D)  # [B, L, D]
        return v
    
    def get_solver(self, cfg):
        """Get the ODE solver for this model"""
        if self.solver is None:
            self.solver = ODESolver(self, cfg)
        return self.solver
    
    def sample_latents(self, num_samples, steps=100, sigma=1.0, device=None, cfg=None):
        """
        Sample latent vectors using the ODE solver.
        This method is kept for backward compatibility.
        
        Args:
            num_samples: Number of samples to generate
            steps: Number of integration steps
            sigma: Standard deviation of the initial noise
            device: Device to use (if None, use model device)
            cfg: Configuration dictionary (required if solver not initialized)
            
        Returns:
            Tensor of sampled latent vectors [num_samples, emb_dim]
        """
        if self.solver is None:
            if cfg is None:
                raise ValueError("cfg must be provided if solver is not initialized")
            self.solver = ODESolver(self, cfg)
            
        return self.solver.sample_latents(
            num_samples=num_samples, 
            method='rk4',  # Use RK4 for better accuracy
            steps=steps,
            sigma=sigma
        )

if __name__ == "__main__":
    # Test the FlowMatchingNet
    print("Testing FlowMatchingNet...")
    
    # Create mock config
    mock_cfg = {
        "flow": {
            "emb_dim": 32,
            "hidden_dim": 64,
        }
    }
    
    # Initialize model
    emb_dim = mock_cfg["flow"]["emb_dim"]
    hidden_dim = mock_cfg["flow"]["hidden_dim"]
    model = FlowMatchingNet(emb_dim, hidden_dim)
    
    # Test forward pass
    batch_size = 5
    z = torch.randn(batch_size, emb_dim)
    t = torch.rand(batch_size)
    
    try:
        # Test forward pass
        v = model(z, t)
        print(f"Forward pass successful. Output shape: {v.shape}")
        
        # Test backward pass
        loss = v.sum()
        loss.backward()
        print("Backward pass successful.")
        
        # Test flow-matching loss calculation
        z0 = torch.randn(batch_size, emb_dim)
        z1 = torch.randn(batch_size, emb_dim)
        t = torch.rand(batch_size)
        
        # Compute straight-line vector 
        u = z1 - z0
        
        # Compute point along interpolation path
        zt = t.unsqueeze(1)*z1 + (1-t).unsqueeze(1)*z0
        
        # Predict vector field
        v = model(zt, t)
        
        # Compute loss
        loss = (v-u).pow(2).sum(1).mean()
        print(f"Flow matching loss: {loss.item()}")
        
        # Test sampling (without ODE solver to avoid dependencies)
        print("Model architecture test passed.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    print("Test finished.") 