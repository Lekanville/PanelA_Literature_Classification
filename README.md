# PanelA_Literature_Classification

Classifies REF 2021 Main Panel A journal articles (UoAs 1-6) into their Unit of Assessment from the title and abstract, using embeddings from four pretrained language models (PLMs) and three classifiers.

## Pipeline

| Phase | Script | Slurm job | Resources |
|---|---|---|---|
| 1. Preprocessing | `code/preprocess_ref_data.py` | `preprocess_ref_data.sh` | CPU |
| 2. Modelling (4 PLMs) | `code/modelling.py` | one `*_all_embedings.sh` per PLM | 4 GPUs |
| 3. CV and k-NN tables | `code/create_cv_table.py`, `code/create_knn_table.py` | `create_cv_table.sh`, `create_knn_table.sh` | CPU |
| 4. Multi-label evaluation | `code/evaluate_multilabel.py` | `evaluate_multilabel.sh` | CPU |

Submit every job from the project root, for example `sbatch preprocess_ref_data.sh`.

## Setup

```bash
conda env create -f environment.yml   # creates the ref_project environment
```

Create a `.env` file in the project root:

```
HF_TOKEN=<Hugging Face token>
SCOPUS_API_KEY=<Elsevier API key>
```

`data/`, `results/`, logs and all `.csv`/`.xlsx`/`.png` files are git-ignored.

## 1. Preprocessing

```bash
sbatch preprocess_ref_data.sh
```

Input: `data/REF_all_submissions.xlsx` (all REF submissions).

Steps:
1. Keep Main Panel A (UoAs 1-6) outputs of type `D` (journal articles).
2. Normalise DOIs and record every DOI-to-UoA association in a registry (kept for multi-label validation).
3. Deduplicate by DOI, keeping the first UoA in workbook order as the primary single label.
4. Randomly sample 1,000 articles per UoA (seed 101).
5. Retrieve abstracts from the Elsevier Abstract Retrieval API and drop articles without one.

Outputs:
- `data/REF_Titles_and_Abstracts.csv`: the dataset used for modelling.
- `data/ref_preprocessing/`: intermediate files, `doi_uoa_registry.csv`, `abstract_failures.csv` and `preprocessing_report.json`.

API responses are cached in `data/ref_preprocessing/abstract_api_cache.jsonl`, so an interrupted job can be resubmitted without repeating requests. Use `--sample-only` to build the sample without calling the API and `--retry-failed` to retry abstracts that previously failed.

## 2. Modelling

One job per PLM; each runs `code/modelling.py` on the preprocessed CSV and writes to its own folder under `results/all_embeddings_modelling/`.

| Job script | `--sbert_model` | Output folder |
|---|---|---|
| `all-mpnet_all_embedings.sh` | `all-mpnet` | `all-mpnet_all_embeddings` |
| `Bio_ClinicalBERT_all_embedings.sh` | `Bio_clinical_BERT` | `Bio_ClinicalBERT_all_embeddings` |
| `BioMedBert_all_embedings.sh` | `Biomed_BERT` | `BioMedBert_all_embeddings` |
| `pubmedbert-base-embeddings_all_embedings.sh` | `PubMed_BERT` | `pubmedbert-base-embeddings_all_embeddings` |

```bash
sbatch all-mpnet_all_embedings.sh
sbatch Bio_ClinicalBERT_all_embedings.sh
sbatch BioMedBert_all_embedings.sh
sbatch pubmedbert-base-embeddings_all_embedings.sh
```

Each job passes `--skip-topic-modeling true` or `false` to control topic extraction and the UoA network plot. It is currently `false` for all-mpnet and `true` for the other three jobs, so the shared analysis runs once. Set it to `false` in another job and `true` in the current one to move the analysis.

Each run:
1. Computes embeddings from the cleaned title and abstract.
2. Makes a stratified 80/20 train/test split and saves the test set as `test_dataset.csv`.
3. Tunes and evaluates a weighted k-NN baseline.
4. For each UoA, trains one-vs-rest `LinearSVC`, `LightGBM` and `KerasMLP` classifiers on a class-balanced training set, with 5-fold nested cross-validation and evaluation on the held-out test set.

Outputs per run: `modelling_results.xlsx`, `knn_tuning_results.xlsx`, `knn_test_results.csv` and cross-validation ROC plots. The run with `--skip-topic-modeling false` also saves `uoa_network_analysis.png`. The BioMedBERT run also saves its LightGBM models to `model/champion_ovr_model_uoa_<n>.joblib`, which phase 4 uses.

## 3. Result tables

```bash
sbatch create_cv_table.sh    # cross-validation AUC table
sbatch create_knn_table.sh   # k-NN held-out test table
```

- `results/all_embeddings_modelling/cv_auc_table.xlsx`: mean (SD) cross-validation AUC-ROC per UoA, PLM and classifier, with the best score per classifier in each UoA in bold. `KerasMLP` is labelled `MLP`.
- `results/all_embeddings_modelling/knn_results_table.xlsx`: champion k, weighting and multi-class test metrics per PLM. It reads `knn_test_results.csv` and falls back to the run's `.log` file for runs made before that file was saved.

## 4. Multi-label evaluation

Training uses one primary UoA per article, but many articles were submitted to several UoAs. This step compares strict precision (matches the primary UoA) with adjusted precision (matches any UoA the article was historically submitted to).

Set `MODEL_OUTPUT_DIR` in `evaluate_multilabel.sh` to a run folder containing `test_dataset.csv` and `model/` (currently the BioMedBERT run), then submit it:

```bash
sbatch evaluate_multilabel.sh
```

The summary is saved as `multilabel_precision.csv` in that folder.