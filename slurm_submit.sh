#!/bin/bash
#SBATCH --job-name=cas9_flow
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=48:00:00

# Get current date and time for log file naming
LOG_TIME=$(date +"%Y%m%d_%H%M%S")

#SBATCH --output=logs/cas9_flow_${LOG_TIME}_%j.out
#SBATCH --error=logs/cas9_flow_${LOG_TIME}_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lotus_env

cd $SLURM_SUBMIT_DIR

python run_experiment.py 