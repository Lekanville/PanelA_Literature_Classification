import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
# import pandas as pd
# import tensorflow as tf

def custom_f1_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return f1_score(y, y_pred)

def custom_auc_scorer(estimator, X, y):
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X)
        if y_score.ndim > 1:
            y_score = y_score[:, 1]
    else:
        y_score = estimator.predict(X)
    return roc_auc_score(y, y_score)

def cross_val_main(gs_cv, X_dat, y_dat, repeats, uoa_dat, output_dir, alg_name):

    # Convert to NumPy arrays to avoid feature name warnings with tree-based models
    X_dat = X_dat.values if hasattr(X_dat, 'values') else X_dat
    y_dat = y_dat.values if hasattr(y_dat, 'values') else y_dat

    custom_cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=repeats,
        random_state=42 # Using a different random state
    )

    if alg_name == "KerasMLP":
        cv_n_jobs = 1
        scoring_logic = {
            'f1': custom_f1_scorer,
            'roc_auc': custom_auc_scorer 
        }
    else:
        cv_n_jobs = -1
        scoring_logic = ('f1', 'roc_auc')

    # The cross-validation object
    cv_results = cross_validate(gs_cv, X_dat, y_dat, cv=custom_cv, scoring=scoring_logic, return_estimator=True, return_indices=True,
                               n_jobs=cv_n_jobs
                               )
    
    # The number of splits
    n_splits = len(cv_results['indices']["test"])
    
    # For colours
    # prop_cycle = plt.rcParams["axes.prop_cycle"]
    # colors = prop_cycle.by_key()["color"]
    # curve_kwargs_list = [dict(alpha=0.3, lw=1, color=colors[fold % len(colors)]) for fold in range(n_splits)]
    # curve_kwargs_list = [dict(color=colors[fold % len(colors)]) for fold in range(n_splits)]
    
    # names = [f"ROC fold {idx}" for idx in range(n_splits)]
    
    mean_fpr = np.linspace(0, 1, 100)   
    interp_tprs = []


    fig, ax = plt.subplots(figsize=(6, 6))
    
    mean_fpr = np.linspace(0, 1, 100)   
    interp_tprs = []
    roc_aucs = []

    # Manually iterate through the outer CV folds
    for i, (est, test_idx) in enumerate(zip(cv_results['estimator'], cv_results['indices']['test'])):
        # 1.  Extract the best model found during the inner tuning loop. This works for any algorithm wrapped in GridSearchCV
        best_model = est.best_estimator_
        inner_model = best_model.named_steps['classifier']
        # X_test_transformed = best_model.named_steps['scaler'].transform(X_dat[test_idx])

        X_test_transformed = X_dat[test_idx]
        for name, step in best_model.steps[:-1]:
            X_test_transformed = step.transform(X_test_transformed)

        # 2. Get the probabilities manually using your 'method' logic
        if hasattr(inner_model, "predict_proba"):
            # For Keras, this will return probabilities
            y_score = inner_model.predict_proba(X_test_transformed)
            # Take the positive class column (usually index 1)
            if y_score.ndim > 1:
                y_score = y_score[:, 1]
        elif hasattr(inner_model, "decision_function"):
            y_score = inner_model.decision_function(X_test_transformed)
        else:
            y_score = inner_model.predict(X_test_transformed)

        # 3. Use the MANUAL plotting method to bypass Scikit-Learn's strict checks   
        fpr, tpr, _ = roc_curve(y_dat[test_idx], y_score)
        roc_auc = auc(fpr, tpr)

        viz = RocCurveDisplay(
            fpr=fpr, 
            tpr=tpr, 
            roc_auc=roc_auc, 
            estimator_name=f"Fold {i}"
        )
        
        # Now plot using the object we built manually
        viz.plot(ax=ax, alpha=0.3, lw=1)
        
        # Add to your tracking lists for the mean curve later
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
        roc_aucs.append(viz.roc_auc)

    # Calculate Mean and Std Dev
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(roc_aucs)
    
    # Plot Mean ROC
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    # Plot Chance Line
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", label="Chance level (AUC = 0.5)", alpha=0.8)
   
    std_tpr = np.std(interp_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability - {alg_name} - UoA {uoa_dat}",
    )
    ax.legend(loc="lower right")
    
    # Filename includes algorithm name
    filename = f"{alg_name}_cross_val_roc_uoa_{uoa_dat}.png"
    # plt.savefig(os.path.join(output_dir, filename))
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    if alg_name == "KerasMLP":
        import tensorflow as tf
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

    return mean_auc, std_auc

    # tprs.append(interp_tpr)
    # aucs.append(viz.roc_auc)

    # # Plot individual fold
    # viz = RocCurveDisplay.from_estimator(
    #     inner_model,
    #     X_test_transformed,
    #     y_dat[test_idx],
    #     name=f"ROC fold {i}",
    #     ax=ax,
    #     alpha=0.3,
    #     linewidth=1,
    #     response_method=method,
    #     **curve_kwargs_list[i]
    # )
    
    # Capture metrics for mean calculation
    # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    # interp_tpr[0] = 0.0
    
    # fig, ax = plt.subplots(figsize=(6, 6))
    # viz = RocCurveDisplay.from_cv_results(
    #     cv_results,
    #     X_dat,
    #     y_dat,
    #     ax=ax,
    #     name=names,
    #     curve_kwargs=curve_kwargs_list,
    #     plot_chance_level=True,
    # )
    
    # for idx in range(n_splits):
    #     interp_tpr = np.interp(mean_fpr, viz.fpr[idx], viz.tpr[idx])
    #     interp_tpr[0] = 0.0
    #     interp_tprs.append(interp_tpr)
    
    # mean_tpr = np.mean(interp_tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(viz.roc_auc)
    
    # ax.plot(
    #     mean_fpr,
    #     mean_tpr,
    #     color="b",
    #     label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    #     lw=2,
    #     alpha=0.8,
    # )
    
    # std_tpr = np.std(interp_tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # ax.fill_between(
    #     mean_fpr,
    #     tprs_lower,
    #     tprs_upper,
    #     color="grey",
    #     alpha=0.2,
    #     label=r"$\pm$ 1 std. dev.",
    # )
    
    # ax.set(
    #     xlabel="False Positive Rate",
    #     ylabel="True Positive Rate",
    #     title=f"Mean ROC curve with variability - {alg_name} - UoA {uoa_dat}",
    # )
    # ax.legend(loc="lower right")
    
    # # Filename includes algorithm name
    # filename = f"{alg_name}_cross_val_roc_uoa_{uoa_dat}.png"
    # plt.savefig(os.path.join(output_dir, filename))
    # plt.close(fig)
    
    # Return metrics for results table
    # return mean_auc, std_auc