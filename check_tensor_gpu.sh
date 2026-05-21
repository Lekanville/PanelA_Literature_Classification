#!/bin/bash
#SBATCH --job-name=tensor_gpu_check         # Name of the job
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --cpus-per-task=1                   # Request 1 CPU core
#SBATCH --mem=4G                            # Request 4GB of RAM
#SBATCH --time=00:05:00                      # Max runtime (5 minutes)
#SBATCH --output=tensor_gpu_check_%j.log     # Standard output and error log

# --- 1. Load Environment (Uncomment the one you use) ---
module load cuda/12.6

export CUDA_HOME=$CUDA_DIR
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source ~/miniforge3/bin/activate ref_project

srun python code/check_tensor_gpu.py