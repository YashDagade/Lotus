import torch
import numpy as np
from tqdm import tqdm

class ODESolver:
    """
    ODE Solver for Flow Matching models that implements various numerical integration methods.
    """
    
    def __init__(self, model, cfg):
        """
        Initialize the ODE solver with a flow model.
        
        Args:
            model: The flow matching model with a forward(z, t) method
            cfg: Configuration dictionary
        """
        self.model = model
        self.cfg = cfg
        
    def euler_integrate(self, z0, steps=100, t_span=(1.0, 0.0), verbose=False):
        """
        Basic Euler integration scheme.
        
        Args:
            z0: Initial point (noise) [B, D]
            steps: Number of integration steps
            t_span: Time range to integrate over (start, end)
            verbose: Whether to show progress bar
            
        Returns:
            Tensor of shape [B, D] with generated samples
        """
        t_start, t_end = t_span
        dt = (t_start - t_end) / steps
        z = z0.clone()
        
        iterator = range(steps)
        if verbose:
            iterator = tqdm(iterator, desc="Sampling")
            
        with torch.no_grad():
            self.model.eval()
            for i in iterator:
                t = torch.ones(z.shape[0], device=z.device) * (t_start - i * dt)
                v = self.model(z, t)
                z = z - v * dt
                
        return z
    
    def heun_integrate(self, z0, steps=100, t_span=(1.0, 0.0), verbose=False):
        """
        Heun's method (improved Euler) for more accurate integration.
        
        Args:
            z0: Initial point (noise) [B, D]
            steps: Number of integration steps
            t_span: Time range to integrate over (start, end)
            verbose: Whether to show progress bar
            
        Returns:
            Tensor of shape [B, D] with generated samples
        """
        t_start, t_end = t_span
        dt = (t_start - t_end) / steps
        z = z0.clone()
        
        iterator = range(steps)
        if verbose:
            iterator = tqdm(iterator, desc="Sampling")
            
        with torch.no_grad():
            self.model.eval()
            for i in iterator:
                t = torch.ones(z.shape[0], device=z.device) * (t_start - i * dt)
                v1 = self.model(z, t)
                
                # Heun's method: predict
                z_tilde = z - v1 * dt
                
                # Heun's method: correct
                t_next = torch.ones(z.shape[0], device=z.device) * (t_start - (i + 1) * dt)
                v2 = self.model(z_tilde, t_next)
                
                # Average the vector fields
                v_avg = (v1 + v2) / 2
                z = z - v_avg * dt
                
        return z
    
    def rk4_integrate(self, z0, steps=100, t_span=(1.0, 0.0), verbose=False):
        """
        4th-order Runge-Kutta integration for high accuracy.
        
        Args:
            z0: Initial point (noise) [B, D]
            steps: Number of integration steps
            t_span: Time range to integrate over (start, end)
            verbose: Whether to show progress bar
            
        Returns:
            Tensor of shape [B, D] with generated samples
        """
        t_start, t_end = t_span
        dt = (t_start - t_end) / steps
        z = z0.clone()
        
        iterator = range(steps)
        if verbose:
            iterator = tqdm(iterator, desc="Sampling")
            
        with torch.no_grad():
            self.model.eval()
            for i in iterator:
                t = torch.ones(z.shape[0], device=z.device) * (t_start - i * dt)
                
                # k1 = f(z, t)
                k1 = self.model(z, t)
                
                # k2 = f(z + k1*dt/2, t + dt/2)
                t_mid = t - dt/2
                k2 = self.model(z - k1 * dt/2, t_mid)
                
                # k3 = f(z + k2*dt/2, t + dt/2)
                k3 = self.model(z - k2 * dt/2, t_mid)
                
                # k4 = f(z + k3*dt, t + dt)
                t_next = t - dt
                k4 = self.model(z - k3 * dt, t_next)
                
                # z = z + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                z = z - dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                
        return z
    
    def sample_latents(self, num_samples, method='rk4', steps=100, sigma=1.0, return_trajectory=False, verbose=False):
        """
        Sample latent vectors by solving the ODE.
        
        Args:
            num_samples: Number of samples to generate
            method: Integration method ('euler', 'heun', or 'rk4')
            steps: Number of integration steps
            sigma: Standard deviation of initial noise
            return_trajectory: Whether to return intermediate points
            verbose: Whether to show progress bar
            
        Returns:
            Tensor of shape [num_samples, D] with sampled latents
            If return_trajectory=True, returns [steps+1, num_samples, D]
        """
        device = next(self.model.parameters()).device
        emb_dim = self.cfg["flow"]["emb_dim"]
        
        # Start from random Gaussian noise
        z0 = torch.randn(num_samples, emb_dim, device=device) * sigma
        
        # Choose integration method
        if method == 'euler':
            integrate_fn = self.euler_integrate
        elif method == 'heun':
            integrate_fn = self.heun_integrate
        elif method == 'rk4':
            integrate_fn = self.rk4_integrate
        else:
            raise ValueError(f"Unknown integration method: {method}")
        
        # Integrate
        if return_trajectory:
            z_traj = [z0.clone()]
            
            # Manually integrate and save points
            t_start, t_end = 1.0, 0.0
            dt = (t_start - t_end) / steps
            z = z0.clone()
            
            iterator = range(steps)
            if verbose:
                iterator = tqdm(iterator, desc="Sampling")
                
            with torch.no_grad():
                self.model.eval()
                for i in iterator:
                    # Use the selected integration method for one step
                    temp_z = integrate_fn(z, steps=1, verbose=False)
                    z = temp_z
                    z_traj.append(z.clone())
                
            return torch.stack(z_traj, dim=0)
        else:
            # Just return final points
            return integrate_fn(z0, steps=steps, verbose=verbose)

def sample_sequences(model, solver, cfg, num_samples=100, method='rk4', steps=100, output_dir=None, run_name=None):
    """
    Sample sequences from the model and save them to a file.
    
    Args:
        model: The flow matching model
        solver: The ODE solver instance
        cfg: Configuration dictionary
        num_samples: Number of sequences to sample
        method: Integration method
        steps: Number of integration steps
        output_dir: Directory to save samples
        run_name: Name of the run for file naming
        
    Returns:
        List of sampled sequences
    """
    import os
    from generator.decode import decode_latents
    
    # Sample latent vectors
    latents = solver.sample_latents(num_samples, method=method, steps=steps, verbose=True)
    
    # Decode to sequences
    sequences = decode_latents(latents)
    
    # Save to file if output_dir is specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine file name
        if run_name is None:
            run_name = cfg["wandb"].get("name", "samples")
        
        file_path = os.path.join(output_dir, f"{run_name}_samples.fasta")
        
        # Write sequences to FASTA file
        with open(file_path, "w") as f:
            for i, seq in enumerate(sequences):
                f.write(f">{run_name}_sample_{i}\n{seq}\n")
                
        print(f"Saved {len(sequences)} sequences to {file_path}")
    
    return sequences 