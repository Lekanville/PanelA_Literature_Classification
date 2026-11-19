#!/bin/bash

#SBATCH --job-name=evaluate_multilabel
#SBATCH --output=evaluate_multilabel_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
source ~/miniforge3/bin/activate ref_project

MODEL_OUTPUT_DIR="results/all_embeddings_modelling/BioMedBert_all_embeddings"

python code/evaluate_multilabel.py \
    --model-output-dir "$MODEL_OUTPUT_DIR" \
    --historical-file data/REF_all_submissions.xlsx