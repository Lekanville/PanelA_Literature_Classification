import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=no info, 2=no warnings, 3=no errors

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, accuracy_score)

from tools.tools import dep_variable, clean_out_dups, select_random, weighted_knn_uoa_scores, similarities_to_weights
from tools.topics_modelling import extract_uoa_research_areas, plot_uoa_network
from tools.hyperparameter_tuning import hyperparam_search_main
from tools.cross_val import cross_val_main
from tools.final_eval import final_evaluation, knn_instance_based_final_eval
from tools.embeddings import compute_embeddings

import warnings
from sklearn.exceptions import ConvergenceWarning
# Filter out the specific UserWarning regarding feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils._plotting")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import random


def set_reproducibility(seed=42):
    # 1. Basic Python and NumPy seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # 2. TensorFlow global seed
    tf.random.set_seed(seed)
    
    # 3. Force TensorFlow to be as deterministic as possible
    # This helps with GPU operations in MirroredStrategy
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # 4. Optional: Enable op determinism (TF 2.9+)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass

set_reproducibility(42)

# This script performs the modelling for each UoA, including:
# 1. Creating the dependent variable (target vs. others)
# 2. Stratified Train/Test Split (CRITICAL STEP)
# 3. Downsampling the Training Data ONLY
# 4. Running Nested Cross-Validation (Tuning + Evaluation) on the RESAMPLED Training Data
# 5. Final Model Training (Hyperparameter Selection)
# 6. Final Evaluation on the Held-Out Test Set



parser = argparse.ArgumentParser(description= "A script to filter data")
parser.add_argument('-i', '--input_file', type=str, required=True, help= 'The input dataset')
parser.add_argument('-s', '--sbert_model', type=str, required=True, help= 'The SBERT model to use for embeddings')
parser.add_argument('-o', '--output_directory', type=str, required=True, help= 'The output directory to save results')


def uoa_modelling(INPUT, SBERT_MODEL, OUTPUT):

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)
    clean_df = clean_out_dups(df)
    embedding_cols = [f'corpus_embeddings_{i}' for i in range(768)]
    combined_text = ["Titl_and_Abs"]

    # ensure that columns are cleaned
    cols_to_clean = embedding_cols + combined_text
    clean_df = clean_df.drop(columns = cols_to_clean, errors='ignore')

    # embeddings
    embedded_df = compute_embeddings(clean_df, SBERT_MODEL)

    # Scan your processed column for the exact literal phrase
    # real_matches = embedded_df [embedded_df['Titl_and_Abs_Clean'].str.contains(r'\bland plants\b', case=False, na=False)]

    # print(f"Total rows with the literal phrase 'land plants': {len(real_matches)}")

    # If the count is 0 or incredibly low, print out where they are collapsing
    # if len(real_matches) == 0:
    #     ghost_matches = embedded_df[embedded_df['Titl_and_Abs_Clean'].str.contains(r'\bland\b.*\bplants\b', case=False, na=False)]
    #     for text in ghost_matches['Titl_and_Abs_Clean'].head(3):
    #         print("--- GHOST SOURCE DETECTED ---")
    #         print(text)

    
    # Data validation checks
    logger.info(f"Original data shape: {df.shape}")
    logger.info(f"Cleaned data shape: {embedded_df.shape}")
    logger.info(f"Unique UoAs: {embedded_df['Unit_of_assessment_number'].nunique()}")
    
    # Check for missing values in embeddings
    missing_embeddings = embedded_df[embedding_cols].isnull().sum().sum()
    logger.info(f"Missing values in embeddings: {missing_embeddings}")
    
    # Validate embeddings (check norms)
    X_all = embedded_df[embedding_cols]
    embedding_norms = np.linalg.norm(X_all.values, axis=1)
    logger.info(f"Embedding norms - Mean: {embedding_norms.mean():.3f}, Std: {embedding_norms.std():.3f}")
    y_all = embedded_df["Unit_of_assessment_number"]
    
    # Check class distribution
    uoa_counts = embedded_df['Unit_of_assessment_number'].value_counts().sort_index()
    logger.info(f"UoA distribution:\n{uoa_counts}")
    
    uoas = embedded_df["Unit_of_assessment_number"].unique().tolist()
    
    
    # 1. THE MASTER SPLIT (Keep X_test_all in a vault) - X_train_all and y_train_all will be used for both tuning and 
    # final training, but only the training portion is used (and also resampled at OvR) tuning. The test portion 
    # remains untouched until final evaluation.
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # 1b. Capture the original dataframe indices assigned to the test set
    test_indices = X_test_all.index
    test_tracking_df = embedded_df.loc[test_indices]
    test_dataset_path = Path(OUTPUT) / "test_dataset.csv"
    test_tracking_df.to_csv(test_dataset_path, index=True)
    logger.info(f"Held-out test set saved to {test_dataset_path} with {len(test_tracking_df)} rows")



    # 2. Extract UoA research areas (themes) using c-TF-IDF for interpretability and insights
    # a. We use the TRAINING portion of the data to ensure that our thematic analysis is based on the same data that the models
    # will learn from. This avoids any "data leakage" of themes from the test set.
    train_indices = X_train_all.index
    train_df = embedded_df.loc[train_indices].copy()
    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Thematic analysis based on {train_df['Unit_of_assessment_number'].nunique()} UoAs in training set.")

    # b. Extract themes and print them out for each UoA. This is purely for interpretability and to understand the research 
    # focus of each UoA based on the training data.
    themes = extract_uoa_research_areas(train_df)
    for uoa, areas in themes.items():
        print(f"UoA: {uoa} \nAreas: {', '.join(areas)}\n")

    # c. We can also visualize the themes using a network graph (similar to image_85d235.jpg). 
    # This is not critical for modelling but can provide valuable insights.
    plot_uoa_network(themes, OUTPUT)


    # 3. TUNING SPLIT (Use only training data to find best k and w)
    X_train_IBA, X_tune_val_IBA, y_tune_train_IBA, y_tune_val_IBA = train_test_split(
        X_train_all, y_train_all, test_size=0.15, random_state=42, stratify=y_train_all
    )

    # --- 1. Global Instance-Based Approach (IBA) ---
    logger.info("Starting Instance-based Baseline...")
    k_values = [10, 25, 50]
    weightings = ["uniform", "inverse_distance", "inverse_distance_squared", "exponential_similarity", "rank"]
    
    best_accuracy = -1
    champion_k = 25 # Defaults
    champion_w = "inverse_distance"
    epsilon = 1e-6  # To prevent division by zero
    tuning_results = []

    for k in k_values:
        for w in weightings:
            # We predict on "tune_val_IBA" using Tune_Train as the neighbors
            # This ensures the 'Best K' is chosen based on unseen data
            scores, names, _ = weighted_knn_uoa_scores(
                X_train_IBA, y_tune_train_IBA.tolist(),
                X_new=X_tune_val_IBA,                 
                k=k, weighting=w, epsilon=epsilon
            )
            
            preds = [names[i] for i in np.argmax(scores, axis=1)]
            accuracy = np.mean([1 if p == a else 0 for p, a in zip(preds, y_tune_val_IBA)])
            
            tuning_results.append({"k": k, "weighting": w, "accuracy": accuracy})
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                champion_k = k
                champion_w = w

    logger.info(f"🏆 Champion Config Found: k={champion_k}, weighting={champion_w} (Acc: {best_accuracy:.4f})")
    
    # Save tuning results
    pd.DataFrame(tuning_results).to_excel(f"{OUTPUT}/knn_tuning_results.xlsx", index=False)

    logger.info("Testing the KNN prediction of the overall best configuration on the held-out test set...")
    knn_instance_based_final_eval(X_train_all, y_train_all.tolist(), X_test_all, y_test_all.tolist(), 
                                 champion_k, champion_w, epsilon)



    # --- 2. OvR Approach ---
    logger.info("Starting OvR Approach...")
    REPEATS = 1
    # algorithms = ["LinearSVC", "LightGBM", "MLP", "KerasMLP"]
    algorithms = ["LinearSVC", "LightGBM", "KerasMLP"]
    results = []

    full_training_data = pd.concat([X_train_all, y_train_all], axis=1)
    full_test_data = pd.concat([X_test_all, y_test_all], axis=1)

    for uoa in uoas:
        logger.info(f"-------------Modelling for UoA {uoa} started--------------")

        df_dep_uoa_training = dep_variable(full_training_data, uoa)
        df_dep_uoa_test = dep_variable(full_test_data, uoa)

        X_train_outer = df_dep_uoa_training[embedding_cols]
        y_train_outer = df_dep_uoa_training["dependant_var"]
        X_test = df_dep_uoa_test[embedding_cols]
        y_test = df_dep_uoa_test["dependant_var"]

        #  X_train_initial, X_test, y_train_initial, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
        # logger.info(f"Training set ratio (imbalanced): {y_train_initial.sum()}/{len(y_train_initial)} ({y_train_initial.mean()*100:.2f}%)")

        X_train_resampled, y_train_resampled = select_random(X_train_outer, y_train_outer)
        logger.info(f"Training set ratio (resampled): {y_train_resampled.sum()}/{len(y_train_resampled)} ({y_train_resampled.mean()*100:.2f}%)")
        logger.info(f"Held-out Test set size: {len(X_test)}")

        for alg in algorithms:
            logger.info(f"\n--- Running {alg} for UoA {uoa} ---")

            grid_search_cv = hyperparam_search_main(ALG=alg)
            # X_train_resampled =  normalize(np.asarray(X_train_resampled, dtype=np.float32), norm="l2")
            # X_test = normalize(np.asarray(X_test, dtype=np.float32), norm="l2")

            mean_auc, std_auc = cross_val_main(grid_search_cv, X_train_resampled, y_train_resampled, REPEATS, uoa, OUTPUT, alg_name=alg)
            logger.info(f"Cross-validation AUC: {mean_auc:.2f} ± {std_auc:.2f}")
            logger.info(f"Completed cross-validation for UoA {uoa} with {alg}")

            logger.info(f"Testing for UoA {uoa} with {alg} started")
            final_gs = hyperparam_search_main(ALG=alg)
            logger.info("Training final model (GridSearch) on ALL resampled training data...")
            final_gs.fit(X_train_resampled, y_train_resampled)
            best_params = final_gs.best_params_
            best_estimator = final_gs.best_estimator_
            logger.info(f"Best Hyperparameters selected for UoA {uoa} ({alg}):")
            formatted_params = {k.replace('classifier__', ''): v for k, v in best_params.items()}
            logger.info(formatted_params)
            final_roc_auc, final_f1, final_precision, final_recall, final_accuracy = final_evaluation(best_estimator, X_test, y_test, uoa)
            logger.info(f"Testing for UoA {uoa} with {alg} completed")
            logger.info(f"Completed analysis and final evaluation for UoA {uoa} with {alg}")
            logger.info("---------------------------------------------------------")
            results.append({
                "uoa": uoa,
                "algorithm": alg,
                "cv_mean_auc": mean_auc,
                "cv_std_auc": std_auc,
                "test_roc_auc": final_roc_auc,
                "test_f1": final_f1,
                "test_precision": final_precision,
                "test_recall": final_recall,
                "test_accuracy": final_accuracy
            })

            if (SBERT_MODEL == "Biomed_BERT") and (alg == "LightGBM"):  
                # Create the target model folder path securely
                model_dir = Path(OUTPUT) / "model"
                model_dir.mkdir(parents=True, exist_ok=True) 
                model_filename = model_dir / f"champion_ovr_model_uoa_{uoa}.joblib"
                
                # Save the final optimized Scikit-learn Pipeline estimator
                joblib.dump(best_estimator, model_filename) 
                logger.info(f"Successfully archived champion model for UoA {uoa} at: {model_filename}")
        # --- The "Fair Fight" KNN Baseline Logic ---
        # logger.info(f"\n--- Running KNN_Baseline for UoA {uoa} ---")
        # # 1. Prepare data (SBERT embeddings are already L2-normalized)
        # X_train_norm = normalize(np.asarray(X_train_outer, dtype=np.float32), norm="l2")  # Normalize training data
        # X_test_norm = normalize(np.asarray(X_test, dtype=np.float32), norm="l2")  # Normalize test data (just in case)

        # # 2. Calculate Cosine Similarity
        # # (Since they are normalized, dot product = cosine similarity)
        # sim_matrix = np.dot(X_test_norm, X_train_norm.T)

        # # 3. Apply Weighting (Inverse Distance)
        # k = champion_k
        # w = champion_w
        # # episom defined above

        # # Get top k indices and their similarity values
        # top_k_indices = np.argpartition(-sim_matrix, kth=k-1, axis=1)[:, :k]

        # # Get the actual similarity values for those top k
        # rows = np.arange(sim_matrix.shape[0])[:, None]
        # neighbour_sims = sim_matrix[rows, top_k_indices]

        # weights = similarities_to_weights(neighbour_sims, w, k, epsilon)

        # # 4. Calculate Weighted Score
        # # Get labels of the neighbours (0s and 1s)
        # neighbour_labels = y_train_outer.values[top_k_indices] 

        # # Apply weights: (sum of weighted labels) / (sum of weights)
        # weighted_scores = np.sum(neighbour_labels * weights, axis=1) / np.sum(weights, axis=1)
        # y_pred = (weighted_scores >= 0.5).astype(int)

        # # 5. Evaluate
        # cm = confusion_matrix(y_test, y_pred)
        # logger.info(f"Confusion Matrix:\n{cm}")

        # knn_auc = roc_auc_score(y_test, weighted_scores)
        # knn_f1 = f1_score(y_test, y_pred)
        # knn_precision = precision_score(y_test, y_pred)
        # knn_recall = recall_score(y_test, y_pred)
        # knn_accuracy = accuracy_score(y_test, y_pred)
        # results.append({
        #     "uoa": uoa,
        #     "algorithm": "KNN_Baseline",
        #     "cv_mean_auc": 0.0, # No CV for KNN
        #     "cv_std_auc": 0.0,  # No CV for KNN
        #     "test_roc_auc": knn_auc,
        #     "test_f1": knn_f1,
        #     "test_precision": knn_precision,
        #     "test_recall": knn_recall,
        #     "test_accuracy": knn_accuracy
        # })
        # logger.info(f"Completed KNN Baseline for UoA {uoa}")


    # Save results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(f"{OUTPUT}/modelling_results.xlsx", index=False)
    logger.info(f"Final testing results saved to {OUTPUT}/modelling_results.xlsx")




if __name__ == "__main__":
    args = parser.parse_args()
    uoa_modelling(args.input_file, args.sbert_model, args.output_directory)