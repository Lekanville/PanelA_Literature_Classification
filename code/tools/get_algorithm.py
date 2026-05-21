import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier



def create_keras_model(hidden_layers=(256, 128), dropout=0.3, learning_rate=0.001, input_dim=768, **kwargs):
    """Create a Keras MLP model."""
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Move strategy inside to avoid pickling/serialization errors during GridSearchCV
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    # Add hidden layers
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    
    return model

def get_algorithm(alg_name):

    # Define hyperparameter grids for Logistic Regression
    if alg_name == "LogReg":
        classifier = LogisticRegression(random_state=42) # random state for reproducibility

        # This is the full max_iter grid:
        # classifier__max_iter': [10000, 50000, 100000, 200000, 300000]
        # Using smaller grid for faster testing

        param_grid = [
            # Grid for None penalty (No Regularization)
            {'classifier__solver': ['newton-cg', 'lbfgs', 'sag'], 'classifier__penalty': [None], 'classifier__max_iter': [10000]},

            # Grid for 'l1' penalty
            {'classifier__solver': ['liblinear', 'saga'], 'classifier__penalty': ['l1'], 'classifier__C': [100, 10, 1.0, 0.1, 0.01], 'classifier__max_iter': [10000]},

            # Grid for 'l2' penalty
            {'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'classifier__penalty': ['l2'], 'classifier__C': [100, 10, 1.0, 0.1, 0.01], 'classifier__max_iter': [10000]},

            # Grid for 'elasticnet' penalty
            {'classifier__solver': ['saga'], 'classifier__penalty': ['elasticnet'], 'classifier__C': [100, 10, 1.0, 0.1, 0.01], 'classifier__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0], 'classifier__max_iter': [10000]}
        ]

        return (classifier, param_grid)   

    # LinearSVC - much faster than LogReg for high-dimensional data (like SBERT embeddings)
    elif alg_name == "LinearSVC":
        classifier = LinearSVC(random_state=42, max_iter=10000)

        param_grid = [
            # Hinge MUST use dual=True
            {'classifier__C': [0.01, 0.1, 1, 10], 'classifier__loss': ['hinge'], 'classifier__dual': [True]},
            # Squared Hinge is flexible (False is usually faster for n_samples > n_features)
            {'classifier__C': [0.01, 0.1, 1, 10], 'classifier__loss': ['squared_hinge'], 'classifier__dual': [False]},
        ]

        return (classifier, param_grid)
    
    # LightGBM - fast gradient boosting, good for high-dimensional data
    elif alg_name == "LightGBM":
        classifier = lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=1, force_col_wise=True)

        param_grid = [
            {'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.05, 0.1], 'classifier__num_leaves': [31, 50]},
            {'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.05, 0.1], 'classifier__max_depth': [5, 7]},
        ]

        return (classifier, param_grid)
    
    # Random Forest - robust ensemble method
    elif alg_name == "RandomForest":
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1)

        param_grid = [
            {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [10, 20], 'classifier__min_samples_split': [2, 5]},
            {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None], 'classifier__min_samples_leaf': [1, 2]},
        ]

        return (classifier, param_grid)
    
    # MLP - Multi-Layer Perceptron (Neural Network)
    elif alg_name == "MLP":
        classifier = MLPClassifier(random_state=42, max_iter=500, early_stopping=True)

        param_grid = [
            {'classifier__hidden_layer_sizes': [(128,), (256,)], 'classifier__activation': ['relu'], 'classifier__alpha': [0.0001, 0.001]},
            {'classifier__hidden_layer_sizes': [(128, 64), (256, 128)], 'classifier__activation': ['relu'], 'classifier__alpha': [0.0001, 0.001]},
        ]

        return (classifier, param_grid)
    
    # Keras MLP - GPU-accelerated Neural Network using TensorFlow/Keras
    elif alg_name == "KerasMLP":
        from scikeras.wrappers import KerasClassifier
        from sklearn.base import ClassifierMixin
        import tensorflow as tf

        tf.keras.backend.clear_session()
        class FinalKerasClassifier(KerasClassifier, ClassifierMixin):
            @property
            def _estimator_type(self):
                return "classifier"

            # We provide both a getter and a setter for classes_
            @property
            def classes_(self):
                import numpy as np
                return np.array([0, 1])

            @classes_.setter
            def classes_(self, value):
                # We let SciKeras think it's setting it, but we stay as [0, 1]
                pass
        
        # FIX: Pass the custom parameters into the constructor here.
        # SciKeras will now recognize 'hidden_layers', 'dropout', and 'learning_rate' as valid params.
        classifier = FinalKerasClassifier(
            model=create_keras_model, 
            epochs=30, 
            batch_size=64, 
            random_state=42,
            verbose=0, # Set to 0 to avoid log flooding during GridSearch
            hidden_layers=(256, 128),
            dropout=0.3,
            learning_rate=0.001,
            loss="binary_crossentropy",
            metrics=["accuracy"],
            input_dim=768,
            classes=[0, 1],
        )

        # The param_grid remains as you have it, because 'classifier__' targets the 
        # pipeline step, and 'dropout' is now a valid attribute of that step.
        param_grid = [
            {
                'classifier__hidden_layers': [(256,), (128, 64)], 
                'classifier__dropout': [0.3], 
                'classifier__learning_rate': [0.001]
            },
            {
                'classifier__hidden_layers': [(256, 128)], 
                'classifier__dropout': [0.3, 0.5], 
                'classifier__learning_rate': [0.001]
            },
        ]

        return (classifier, param_grid)


