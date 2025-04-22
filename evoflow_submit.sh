#!/bin/bash
#SBATCH --job-name=cas9_evoflow
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=16:00:00

# Get current date and time for log file naming
LOG_TIME=$(date +"%Y%m%d_%H%M%S")

#SBATCH --output=logs/cas9_evoflow_${LOG_TIME}_%j.out
#SBATCH --error=logs/cas9_evoflow_${LOG_TIME}_%j.err

# Load conda into shell
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the scratch-based environment
conda activate /home/hice1/pponnusamy7/scratch/Lotus/lotus_env

# Go to the directory from which this script was submitted
cd $SLURM_SUBMIT_DIR

# Make sure logs directory exists
mkdir -p logs

# Start the experiment
python run_evoflow_experiment.py 