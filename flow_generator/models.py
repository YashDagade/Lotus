import torch, torch.nn as nn
import numpy as np
from flow_generator.solver import ODESolver

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
