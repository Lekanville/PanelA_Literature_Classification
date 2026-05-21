from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer
from tools.get_algorithm import get_algorithm



def custom_f1_scorer(estimator, X, y):
    from sklearn.metrics import f1_score
    y_pred = estimator.predict(X)
    return f1_score(y, y_pred)

def hyperparam_search_main(ALG): 
    classifier, param_grid = get_algorithm(ALG)

    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('classifier', classifier)
    ])
    
    # Determine parallel jobs: 
    # Use all cores (-1) for CPU models, but only 1 for GPU-heavy Keras models
    n_jobs_to_use = 1 if ALG == "KerasMLP" else -1
    # f1_binary = make_scorer(f1_score, response_method='predict')
    if ALG == "KerasMLP":
        # We wrap it manually to ensure it only ever calls .predict()
        f1_scorer = custom_f1_scorer
    else:
        f1_scorer = 'f1'
    
    cv_inner = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid, 
        n_jobs=n_jobs_to_use,  # Dynamically assigned
        cv=cv_inner, 
        scoring=f1_scorer,
        error_score='raise'
    )

    return grid_search