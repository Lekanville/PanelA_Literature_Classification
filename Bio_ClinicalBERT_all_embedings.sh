#!/bin/bash

#SBATCH --job-name=Bio_ClinicalBERT_all_embeddings
#SBATCH --output=Bio_ClinicalBERT_all_embeddings.log
#SBATCH --gpus=4
#SBATCH --ntasks=1

# Request 16 CPU cores for the single task. 
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# --- CRITICAL CLUSTER DRIVERS ---
module load cuda/12.6

# --- ACTIVATE ENVIRONMENT ---
source ~/miniforge3/bin/activate ref_project

# # --- CRITICAL HPC STABILITY FIXES ---
# # FIX 1 & 2: Direct joblib/loky temporary files to local storage ($TMPDIR)
export JOBLIB_TEMP_FOLDER=$TMPDIR
export LOKY_TEMP_FOLDER=$TMPDIR

# # FIX 3: Force Python multiprocessing start method
export PYTHON_START_METHOD='forkserver'

# # FIX 4 (AGGRESSIVE): Explicitly limit all parallel operations to a single thread/core.
# # This is the final step to bypass the hard system limit on IPC resources (semaphores/locks).
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1

# # CRITICAL LOKY BYPASS: Explicitly tell the joblib/loky backend to use only 1 CPU.
# export LOKY_MAX_CPU_COUNT=1

# # GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TF_CPP_MIN_LOG_LEVEL=1

srun python code/modelling.py --input_file data/actual_Bio_ClinicalBERT_all_embeddings.csv  \
                                --sbert_model Bio_clinical_BERT \
                                --output_directory results/all_embeddings_modelling/Bio_ClinicalBERT_all_embeddings