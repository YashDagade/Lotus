# 🧬 Cas9-FlowGen: Generative Modeling of Cas9 Proteins via Latent Flow Matching

## 🧠 Overview
This project aims to develop a generative model of Cas9 proteins by leveraging pretrained protein language models and flow matching in the latent space. Our goal is to build a controllable and efficient framework for designing new Cas9 variants, optionally conditioned on functional motifs such as PAM preferences. Inspired by the ProtFlow architecture, we aim to enable single-step generation of high-quality, biologically meaningful Cas9 sequences for use in genome editing applications.

## 🧪 Motivation
Cas9 enzymes are the cornerstone of CRISPR-based gene editing systems. However, most current applications rely on a few well-characterized variants like SpCas9 or SaCas9. Designing new Cas9 orthologs with diverse PAM compatibility, improved efficiency, and low off-target activity remains a major bottleneck in expanding the CRISPR toolkit.

Existing methods like autoregressive generation and diffusion-based models are limited by:

- Large sequence modeling space
- Slow inference and sampling
- Poor global semantic coherence
- Need for massive compute

We propose a flow-based generative model that operates on a latent representation space extracted from pretrained models (like ESM-2), enabling:

- Fast, single-step sampling
- Semantic preservation
- Conditional generation

Will be updated soon with more details :D

# LOTUS: Latent Flow-Matching for Protein Sequence Generation

LOTUS (Latent flOw-matching for proTein seqUence generationS) is a computational framework for generating novel protein sequences by learning flow patterns in the latent embedding space of proteins. This approach combines the power of protein language models (ESM-2) with flow-matching generative models to enable continuous exploration of protein sequence space.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Workflow Explanation](#workflow-explanation)
  - [Data Preparation](#data-preparation)
  - [Embedding Generation](#embedding-generation)
  - [Flow Matching Training](#flow-matching-training)
  - [Sequence Generation](#sequence-generation)
  - [Validation and Evaluation](#validation-and-evaluation)
- [Mathematical Details](#mathematical-details)
  - [Flow Matching Theory](#flow-matching-theory)
  - [ODE Solvers](#ode-solvers)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Customization](#customization)

## 🌟 Overview

LOTUS tackles the protein design problem by:

1. Using ESM-2 to project protein sequences into a continuous latent space
2. Learning flow patterns between pairs of sequences in this latent space using flow matching
3. Sampling novel points from the latent space by integrating along learned vector fields 
4. Decoding the sampled latent points back to protein sequences

Unlike traditional methods that directly generate sequences, our approach operates in a continuous embedding space, allowing for more effective exploration of the protein fitness landscape.

## 📁 Project Structure

```
LOTUS/
├── config.yaml               # Configuration file for all parameters
├── requirements.txt          # Dependencies
├── run_experiment.py         # Main script for running the full pipeline
├── slurm_submit.sh           # Script for submitting to Slurm cluster
├── dataset/                  # Data directory
│   ├── cas9_uniprot.fasta    # Example dataset
│   └── splits/               # Train/val/test splits (created during run)
├── generator/                # Core model components
│   ├── embed_sequences.py    # ESM-2 embedding generation
│   ├── models.py             # Flow-matching network architecture
│   ├── train.py              # Training loop for flow-matching
│   ├── solver.py             # ODE solvers for sampling from flow model
│   ├── validate.py           # Validation functions with structural metrics
│   ├── downstream.py         # Downstream evaluation with AlphaFold
│   ├── decode.py             # Decoder for latent→sequence conversion
│   └── train_decoder.py      # Decoder training script
├── utils/                    # Utility functions
│   └── mmseqs_split.py       # Sequence clustering and splitting
├── samples/                  # Directory for generated sequences
└── models/                   # Directory for saved models
```

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yashdagade/lotus.git
cd lotus

# Create a conda environment
conda create -n lotus_env python=3.8
conda activate lotus_env

# Install dependencies
pip install -r requirements.txt

# Optional: Install MMseqs2 for sequence clustering
conda install -c bioconda mmseqs2
```

## 🔄 Workflow Explanation

### Data Preparation

The pipeline starts with a FASTA file containing protein sequences (e.g., Cas9 variants). To ensure unbiased evaluation, we perform sequence clustering based on sequence identity:

```python
# utils/mmseqs_split.py
def split_clusters(fasta, out_dir, id_min=0.3, cov=0.8):
    # Create sequence database
    subprocess.run(f"mmseqs createdb {fasta} seqDB", shell=True, check=True)
    
    # Cluster sequences at specified identity threshold
    subprocess.run(f"mmseqs cluster seqDB clusterDB tmp --min-seq-id {id_min} -c {cov}", 
                  shell=True, check=True)
    
    # Extract cluster information
    subprocess.run("mmseqs createtsv seqDB seqDB clusterDB clusters.tsv", 
                  shell=True, check=True)
    
    # Assign sequences to train/val/test based on cluster membership
    df = pd.read_csv("clusters.tsv", sep="\t", names=["seq","seq2","cluster"])
    clusters = df.cluster.unique().tolist()
    random.shuffle(clusters)
    n = len(clusters)
    
    # Split clusters: 80% train, 10% validation, 10% test
    train_c = set(clusters[:int(0.8*n)])
    val_c = set(clusters[int(0.8*n):int(0.9*n)])
    test_c = set(clusters[int(0.9*n):])
    
    # Write sequences to corresponding split files
    # ...
```

By clustering at 30% sequence identity, we ensure that sequence homology doesn't bias our model evaluation. Splitting by clusters rather than individual sequences helps assess generalization to divergent sequences.

### Embedding Generation

We use the ESM-2 protein language model to convert amino acid sequences into continuous embeddings:

```python
# generator/embed_sequences.py
def embed(fasta, out_pt, model_name):
    # Load ESM-2 model
    model, alphabet = esm.pretrained.__dict__[model_name]()
    model.eval()
    
    # Convert sequences to embeddings
    for batch in batches:
        with torch.no_grad():
            out = model(toks, repr_layers=[model.num_layers])
        rep = out["representations"][model.num_layers].mean(1)
        embs.append(rep.cpu())
    
    # Save embeddings
    torch.save({"ids":ids,"embeddings":embs}, out_pt)
```

We use the final layer representations from ESM-2 and average across sequence positions to obtain a fixed-dimensional embedding (1280D for ESM-2 t33_650M) for each protein. These embeddings capture rich evolutionary and structural information from the protein language model's pretraining.

### Flow Matching Training

The core of LOTUS is training a neural network to model vector fields between pairs of protein embeddings:

```python
# generator/models.py
class FlowMatchingNet(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        # Time embedding network
        self.time_mlp = nn.Sequential(...)
        
        # Main network for predicting vector field
        self.net = nn.Sequential(...)
    
    def forward(self, z, t):
        # z: batch of points in latent space
        # t: interpolation timestep (0→1)
        te = self.time_mlp(t.unsqueeze(1))
        return self.net(torch.cat([z, te], dim=1))
```

Training uses the flow matching objective, which teaches the network to predict the vector field that transforms one embedding into another:

```python
# generator/train.py
def flow_loss(model, z0, z1, t):
    # Compute straight-line vector between z0 and z1
    u = z1 - z0
    
    # Compute point along interpolation path
    zt = t.unsqueeze(1)*z1 + (1-t).unsqueeze(1)*z0
    
    # Predict vector field at this point
    v = model(zt, t)
    
    # Loss is MSE between predicted and actual vector
    return (v-u).pow(2).sum(1).mean()
```

During training, we:
1. Sample random pairs of protein embeddings (z0, z1)
2. Sample random interpolation times t ∈ [0,1]
3. Compute the straight-line vector field u = z1 - z0
4. Compute the interpolated point zt = t*z1 + (1-t)*z0
5. Predict the vector field v at zt using our model
6. Minimize the squared error between v and u

This approach teaches the model to learn a continuous vector field that can transform any protein embedding into another, enabling us to explore the manifold of protein embeddings.

### Sequence Generation

After training, we generate novel protein sequences through a two-step process:

1. **Sample from the latent space** using numerical integration of the learned vector field:

```python
# generator/solver.py
def rk4_integrate(self, z0, steps=100, t_span=(1.0, 0.0), verbose=False):
    """4th-order Runge-Kutta integration for high accuracy."""
    t_start, t_end = t_span
    dt = (t_start - t_end) / steps
    z = z0.clone()
    
    with torch.no_grad():
        for i in range(steps):
            t = torch.ones(z.shape[0], device=z.device) * (t_start - i * dt)
            
            # RK4 integration steps
            k1 = self.model(z, t)
            t_mid = t - dt/2
            k2 = self.model(z - k1 * dt/2, t_mid)
            k3 = self.model(z - k2 * dt/2, t_mid)
            t_next = t - dt
            k4 = self.model(z - k3 * dt, t_next)
            
            # Update z
            z = z - dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return z
```

2. **Decode latent vectors** back to amino acid sequences:

```python
# generator/decode.py
class DecoderBlock(nn.Module):
    """Decoder for latent→sequence conversion"""
    
    def decode_latents(self, z_latents):
        """Convert latent vectors to amino acid sequences"""
        # Expand latents to sequence length
        z_seq = repeat(z_latents, 'b d -> b t d', t=self.max_len)
        
        # Predict amino acid logits at each position
        logits = self.forward(z_seq)
        
        # Convert to token indices 
        token_ids = torch.argmax(logits, dim=-1)
        
        # Convert tokens to amino acid sequences
        sequences = []
        for toks in token_ids:
            # Process tokens to amino acid sequence
            # ...
        
        return sequences
```

The generation process leverages several numerical integration methods:

- **Euler integration**: Simplest method, less accurate but faster
- **Heun's method**: Improved accuracy over Euler with predictor-corrector approach
- **4th-order Runge-Kutta (RK4)**: High accuracy integration with four evaluation points

### Validation and Evaluation

LOTUS uses several validation and evaluation strategies:

1. **Early stopping** based on validation loss:

```python
# generator/train.py
if val_loss < best_val:
    best_val = val_loss
    patience = 0
    torch.save(model.state_dict(), "best_flow.pt")
    best_model = model.state_dict().copy()
else:
    patience += 1
    if patience >= cfg["flow"]["early_stop_patience"]:
        break
```

2. **Structural validation** of generated sequences:

```python
# generator/validate.py
def validate(model, cfg):
    # Sample latent vectors
    zs = model.sample_latents(num_samples)
    
    # Decode to sequences
    sequences = decode_latents(zs)
    
    # Predict structure with AlphaFold
    colabfold_run(...)
    
    # Calculate TM-scores against reference structures
    tm_scores = [calculate_tm_score(pred, ref) for pred, ref in ...]
    
    # Extract AlphaFold confidence metrics (pLDDT, pTM)
    metrics = extract_af_confidences(...)
    
    # Log metrics to W&B
    wandb.log({...})
    
    return tm_avg  # Return average TM-score for early stopping
```

3. **Downstream evaluation** of generated sequences:

```python
# generator/downstream.py
def evaluate_downstream(cfg):
    # Run AlphaFold, TM‑score, etc.
    tm_avg = ...
    pLDDT = ...
    
    # Log metrics to W&B
    wandb.log({...})
    
    return metrics
```

## 🧮 Mathematical Details

### Flow Matching Theory

Flow matching works by learning a time-dependent vector field that transforms between data points. Given two protein embeddings z₀ and z₁, we define a straight-line path:

zt = t·z₁ + (1-t)·z₀ for t ∈ [0,1]

The vector field at any point along this path is simply u = z₁ - z₀.

We train a neural network vθ(z,t) to approximate this vector field at any point in the latent space. The loss function is:

L(θ) = E[‖vθ(zt,t) - u‖²]

After training, we can sample new points by:
1. Starting with random noise z₀ ~ N(0,I)
2. Solving the ODE: dz/dt = -vθ(z,t) from t=1 to t=0

The solution at t=0 gives us a sample from the learned distribution.

### ODE Solvers

We implement several ODE solvers for numerical integration:

1. **Euler Method**:
   z_{n+1} = z_n - vθ(z_n,t_n)·Δt

2. **Heun's Method** (Improved Euler):
   z̃_{n+1} = z_n - vθ(z_n,t_n)·Δt
   z_{n+1} = z_n - 0.5·[vθ(z_n,t_n) + vθ(z̃_{n+1},t_{n+1})]·Δt

3. **4th-order Runge-Kutta**:
   k₁ = vθ(z_n, t_n)
   k₂ = vθ(z_n - k₁·Δt/2, t_n-Δt/2)
   k₃ = vθ(z_n - k₂·Δt/2, t_n-Δt/2)
   k₄ = vθ(z_n - k₃·Δt, t_n-Δt)
   z_{n+1} = z_n - (k₁ + 2k₂ + 2k₃ + k₄)·Δt/6

Higher-order methods give more accurate results but require more computation. We use RK4 by default for its excellent balance of accuracy and efficiency.

## 🚀 How to Run

### Full Pipeline

To run the complete pipeline (data splitting, embedding, training, evaluation, sampling):

```bash
# Using Slurm
sbatch slurm_submit.sh

# Without Slurm
python run_experiment.py
```

### Individual Components

You can also run individual components:

```bash
# Just embedding
python -m generator.embed_sequences --fasta dataset/cas9_uniprot.fasta --out dataset/cas9_embeddings.pt --model esm2_t33_650M_UR50D

# Just training
python -m generator.train

# Just decoder training (if you have labeled data)
python -m generator.train_decoder --config config.yaml
```

## ⚙️ Configuration

All parameters are centralized in `config.yaml`:

```yaml
# Global config
dataset_path: dataset/cas9_uniprot.fasta
esm_model: "esm2_t33_650M_UR50D"

# MMseqs2 clustering
cluster:
  min_seq_id: 0.3
  coverage: 0.8

# Flow-matching hyperparameters
flow:
  emb_dim: 1280
  hidden_dim: 1024
  batch_size: 64
  learning_rate: 1e-4
  max_epochs: 300
  val_freq: 5
  early_stop_patience: 3

# Sampling configuration
sampling:
  num_samples: 100
  method: "rk4"
  steps: 100
  output_dir: "samples"

# Downstream evaluation
downstream:
  num_samples: 10
  num_recycles: 1
  model_type: "AlphaFold2-ptm"
  use_structural_validation: true
  ref_pdbs: []
  hmm_profile: ""

# Decoder settings
decoder:
  dim: 1280
  nhead: 20
  dropout: 0.2
  max_len: 1536
  pretrained_path: ""

# W&B logging
wandb:
  entity: "programmablebio"
  project: "cas9_flow"
  name: "latent_flow_smoke_run"
```

## 🛠️ Customization

### Using Your Own Dataset

To use your own protein dataset:

1. Replace the FASTA file in `dataset/`
2. Update `dataset_path` in `config.yaml`
3. Run the pipeline

### Training Your Own Decoder

For optimal results, train a dedicated decoder on your protein family:

1. Uncomment and set the decoder training parameters in `config.yaml`:
   ```yaml
   decoder:
     train: true
     learning_rate: 1e-4
     batch_size: 32
     max_epochs: 100
   ```

2. Prepare paired data of embeddings and sequences
3. Run the decoder training:
   ```bash
   python -m generator.train_decoder
   ```

### Structural Validation

To enable structural validation:

1. Ensure you have ColabFold installed
2. Provide paths to reference structures in `config.yaml`:
   ```yaml
   downstream:
     ref_pdbs: ["path/to/reference1.pdb", "path/to/reference2.pdb"]
   ```

3. Run the pipeline with structural validation enabled:
   ```yaml
   downstream:
     use_structural_validation: true
   ```

### Advanced Sampling

Fine-tune the sampling process:

1. Adjust sampling parameters in `config.yaml`:
   ```yaml
   sampling:
     num_samples: 1000  # More samples
     method: "heun"     # Alternative integration method
     steps: 500         # More accurate integration
   ```

2. Run generation:
   ```bash
   python run_experiment.py
   ```

## 📈 Monitoring with Weights & Biases

LOTUS integrates with W&B for comprehensive experiment tracking:
- Training metrics (loss curves)
- Validation metrics (TM-scores, pLDDT)
- Generated sequences and their properties
- Hyperparameter tracking

To view results:
1. Create an account at [wandb.ai](https://wandb.ai)
2. Update the W&B configuration in `config.yaml`
3. Access your project dashboard at `https://wandb.ai/[entity]/[project]`