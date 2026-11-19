#!/bin/bash

#SBATCH --job-name=preprocess_ref_data
#SBATCH --output=preprocess_ref_data_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
source ~/miniforge3/bin/activate ref_project

python code/preprocess_ref_data.py \
    --input-file data/REF_all_submissions.xlsx \
    --output-dir data/ref_preprocessing \
    --final-output data/REF_Titles_and_Abstracts.csv \
    --sample-per-uoa 1000 \
    --seed 101 \
    --request-delay 1.0 \
    --max-attempts 5