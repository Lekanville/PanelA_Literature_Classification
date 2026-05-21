import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tools.tools import weighted_knn_uoa_scores

def final_evaluation(model, X_test, y_test, uoa_dat):
    """
    Evaluates the final best model on the original, imbalanced, held-out test set.
    """
    # Convert to NumPy arrays to avoid feature name warnings
    X_test = X_test.values if hasattr(X_test, 'values') else X_test
    y_test = y_test.values if hasattr(y_test, 'values') else y_test
    # 1. Predict probabilities for ROC AUC
    # Use decision_function for models without predict_proba (e.g., LinearSVC)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For LinearSVC and similar models, use decision function
        y_proba = model.decision_function(X_test)
    
    # 2. Predict class labels (using default threshold of 0.5 for F1)
    y_pred = model.predict(X_test)
    
    final_roc_auc = roc_auc_score(y_test, y_proba)
    final_f1 = f1_score(y_test, y_pred)
    final_precision = precision_score(y_test, y_pred)
    final_recall = recall_score(y_test, y_pred)
    final_accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    logger.info(f"--- Final Model Evaluation on Held-Out Test Set (UoA {uoa_dat}) ---")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"  Final ROC AUC: {final_roc_auc:.2f}")
    logger.info(f"  Final F1 Score: {final_f1:.2f}")
    logger.info(f"  Final Precision: {final_precision:.2f}")
    logger.info(f"  Final Recall: {final_recall:.2f}")
    logger.info(f"  Final Accuracy: {final_accuracy:.2f}")
    
    return final_roc_auc, final_f1, final_precision, final_recall, final_accuracy


def knn_instance_based_fnal_eval(X_train_all, y_tune_train_all, X_test_all, y_test_all, champion_k, champion_w, epsilon):

    # 1. Run inference using your found champion configuration parameters
    final_scores, names, _ = weighted_knn_uoa_scores(
        X_train_all, y_tune_train_all, X_new=X_test_all,                                 
        k=champion_k, weighting=champion_w, epsilon=epsilon
    )

    # Convert raw scores to absolute predictions
    y_pred = [names[i] for i in np.argmax(final_scores, axis=1)]

    # 2. Compute Multi-class Macro Metrics for direct OvR Comparison
    # This provides global Precision, Recall, and F1 across all UoAs combined
    iba_report = classification_report(y_test_all, y_pred, output_dict=True)
    logger.info(f"IBA Overall Macro F1-Score: {iba_report['macro avg']['f1-score']:.4f}")
    logger.info(f"IBA Overall Macro Precision: {iba_report['macro avg']['precision']:.4f}")
    logger.info(f"IBA Overall Macro Recall: {iba_report['macro avg']['recall']:.4f}")

    # 3. YES, you can calculate Multi-class ROC-AUC!
    # Using the "one-vs-one" or "one-vs-rest" probability handling method:
    # We pass the raw normalized probabilities (scores) to evaluate the ranking capability.
    try:
        # Normalize final scores to sum to 1 row-wise to represent probabilities
        proba_matrix = final_scores / np.sum(final_scores, axis=1, keepdims=True)
        iba_roc_auc = roc_auc_score(y_test_all, proba_matrix, multi_class='ovr', average='macro')
        logger.info(f"IBA Multi-class Macro ROC-AUC: {iba_roc_auc:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate multi-class AUC: {e}")

    # 4. Multi-class Confusion Matrix
    string_axes = [str(x) for x in names]
    iba_cm = confusion_matrix(y_test_all, y_pred, labels=names)
    logger.info(f"IBA Matrix Axis Order: {string_axes}")
    logger.info(f"IBA Final Multi-class Confusion Matrix:\n{iba_cm}")