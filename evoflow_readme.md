# EvoFlow for CAS9 Generation

This project fine-tunes the EvoFlow protein language model (PLM) to generate novel CAS9 protein sequences. EvoFlow is a bidirectional non-coding RNA language model built on a masked discrete diffusion model (MDM) formulation.

## Project Structure

```
.
├── evoflow_generator/         # EvoFlow generator code
│   ├── __init__.py            # Package initialization
│   ├── train.py               # Fine-tuning implementation
│   └── generate.py            # Sequence generation using fine-tuned model
├── evoflow_config.yaml        # Configuration parameters
├── evoflow_submit.sh          # SLURM submission script
├── run_evoflow_experiment.py  # Main experiment script
└── evoflow_readme.md          # This documentation file
```

## Overview

The pipeline consists of the following steps:

1. **Data Preparation**: Split protein sequences into train/val/test sets based on sequence identity
2. **Fine-tuning**: Train EvoFlow on CAS9 sequences using masked language modeling (MLM)
3. **Generation**: Sample novel sequences from the fine-tuned model
4. **Evaluation**: Run downstream validation on the generated sequences

## Components

### Data Processing

- Sequences are clustered by similarity using MMseqs2 (via `utils.mmseqs_split`)
- Clusters are split into train/validation/test sets to prevent data leakage
- Sequences are tokenized for input to the language model

### Fine-tuning (evoflow_generator/train.py)

- Uses the pretrained EvoFlow model: `fredzzp/EvoFlow-650M-context-3070`
- Employs masked language modeling (MLM) where 15% of tokens are randomly masked
- Implements early stopping based on validation loss
- Logs metrics to Weights & Biases for monitoring
- Hyperparameters are configurable via the config file

### Generation (evoflow_generator/generate.py)

- Generates sequences using the fine-tuned model
- Implements autoregressive sampling with temperature control
- Uses top-k and top-p (nucleus) filtering for higher quality outputs
- Saves sequences in FASTA format for downstream analysis

### Experiment Execution (run_evoflow_experiment.py)

- Orchestrates the entire pipeline
- Handles dataset preparation, training, generation, and evaluation
- Integrates with Weights & Biases for experiment tracking

## Running the Pipeline

To run the complete pipeline:

```bash
sbatch evoflow_submit.sh
```

This will:
1. Activate the appropriate conda environment
2. Run the entire pipeline from data preparation to evaluation
3. Log results to Weights & Biases

## Configuration

The `evoflow_config.yaml` file contains all configurable parameters:

- **Weights & Biases settings**: project name, run name, etc.
- **Dataset path**: location of the CAS9 sequences
- **Clustering parameters**: sequence identity threshold, coverage
- **Training hyperparameters**: batch size, learning rate, early stopping patience
- **Generation settings**: number of samples, temperature, top-k, top-p
- **Model paths**: where to save and load models

## Key Features

- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Efficient Fine-tuning**: Builds on a powerful pretrained model
- **Quality Control**: Uses sampling strategies for high-quality generation
- **Experiment Tracking**: Comprehensive logging with Weights & Biases
- **Downstream Evaluation**: Assesses the quality of generated sequences

## Requirements

The code requires:
- PyTorch
- Transformers (Hugging Face)
- Weights & Biases
- BioPython
- MMseqs2 (for clustering)

All dependencies are available in the provided conda environment:
```
/home/hice1/pponnusamy7/scratch/Lotus/lotus_env
``` 