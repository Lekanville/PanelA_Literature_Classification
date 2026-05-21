#!/bin/bash
#SBATCH --job-name=gpu_check          # Name of the job
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=1             # Request 1 CPU core
#SBATCH --mem=4G                      # Request 4GB of RAM
#SBATCH --time=00:05:00               # Max runtime (5 minutes)
#SBATCH --output=gpu_check_%j.log     # Standard output and error log

# --- 1. Load Environment (Uncomment the one you use) ---
module load cuda/12.6

source ~/miniforge3/bin/activate ref_project

# --- 2. Run the Diagnostic ---
echo "Running GPU diagnostic for job $SLURM_JOB_ID..."

python -c "import torch; \
print(f'PyTorch Version: {torch.__version__}'); \
print(f'CUDA Available: {torch.cuda.is_available()}'); \
print(f'CUDA Version: {torch.version.cuda}'); \
print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"