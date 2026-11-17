#!/bin/bash

#SBATCH --job-name=knn_results_table
#SBATCH --output=knn_results_table_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --export=ALL

# CPU-only report generation; no GPUs are required.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
source ~/miniforge3/bin/activate ref_project

python code/create_knn_table.py
