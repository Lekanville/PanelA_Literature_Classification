import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def dep_variable(dataframe, uoa):
    # Dependent Variable
    dataframe = dataframe.copy()
    dataframe["dependant_var"] = dataframe['Unit_of_assessment_number'].apply(lambda x: 1 if x == uoa else 0)
    return (dataframe)

# Took out articles that might have been submitted into multiple UoAs
def clean_out_dups(dataframe):
    dataframe = dataframe.copy()

    #Take out duplicates (non-NaN duplicates)
    doi_duplicates = dataframe[(dataframe["DOI"].notnull()) & (dataframe["DOI"].duplicated())]["DOI"].tolist()
    dataframe = dataframe[~dataframe["DOI"].isin(doi_duplicates)]

    if "_id" in dataframe.columns:
        dataframe.drop("_id", axis = 1, inplace = True)

    dataframe.reset_index(inplace = True, drop = True)

    return dataframe

# This is done to downsample the negative class
def select_random_initial(train_initial, uoa):
    n_to_select = len(train_initial[train_initial['dependant_var'] == 1])

    # combined = pd.concat([train_x, train_y], axis = 1)
    to_sample = train_initial[train_initial['dependant_var'] == 0].reset_index(drop = True)
    pos_class = train_initial[train_initial['dependant_var'] == 1].reset_index(drop = True)

    selected_dataset = pd.DataFrame()
    # # for i in categories:
    # # df_category = data[data['Unit of assessment number'] == i].reset_index(drop = True)
    
    ind = list(range(len(to_sample)))

    rng = np.random.default_rng(seed=101)
    randomly_selected = list(rng.choice(ind, n_to_select, replace=False))
    
    # #randomly_selected = np.random.choice(ind, len(target_images))

    for i in randomly_selected:
        selected_dataset = pd.concat([selected_dataset, to_sample[to_sample.index == i]], axis = 0)

    selected_dataset = pd.concat([selected_dataset, pos_class], axis = 0, ignore_index = True)

    # selected_dataset_X = selected_dataset[[f'distance_to_median_of_uoa_{uoa}']]
    # selected_dataset_y = selected_dataset['dependant_var']

    # X, y = shuffle(selected_dataset_X, selected_dataset_y)
    
    # return (X, y)
    selected_dataset = shuffle(selected_dataset)
    return (selected_dataset)

def select_random(X_train_initial, y_train_initial):
    """
    Downsamples the negative class (0) in the training data to match 
    the size of the positive class (1). Applied ONLY to training data.
    """
    # Combine X and y for easier shuffling and splitting
    combined = pd.concat([X_train_initial.reset_index(drop=True), 
                          y_train_initial.reset_index(drop=True)], axis=1)

    # Determine the count of the minority class (Positive Class, 1)
    n_to_select = combined[combined['dependant_var'] == 1].shape[0]

    # Separate classes
    pos_class = combined[combined['dependant_var'] == 1].copy()
    neg_class = combined[combined['dependant_var'] == 0].copy()

    # Downsample the majority class (Negative Class, 0)
    # Using sample() is cleaner than the manual indexing approach
    # Set random_state for reproducibility
    neg_downsampled = neg_class.sample(n=n_to_select, replace=False, random_state=101)

    # Combine the balanced dataset and shuffle
    selected_dataset = pd.concat([pos_class, neg_downsampled], axis=0)
    selected_dataset = shuffle(selected_dataset, random_state=42)

    # Re-split X and y
    X_resampled = selected_dataset[X_train_initial.columns]
    y_resampled = selected_dataset['dependant_var']
    
    return (X_resampled, y_resampled)

def weighted_knn_uoa_scores(X_train, y_train, X_new, k, weighting, epsilon=1e-6):
    # SBERT embeddings are already L2-normalized, so no need to normalize again
    X_train_norm = normalize(np.asarray(X_train, dtype=np.float32), norm="l2")  # Ensure training data is normalized
    X_new_norm = normalize(np.asarray(X_new, dtype=np.float32), norm="l2")  # Ensure new data is normalized
    sim_matrix = np.dot(X_new_norm, X_train_norm.T)

    # 2. Get Top K Neighbors
    neighbour_indices = np.argpartition(-sim_matrix, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(sim_matrix.shape[0])[:, None]
    neighbour_sims = sim_matrix[row_idx, neighbour_indices]
    
    # Sort them properly
    order = np.argsort(-neighbour_sims, axis=1)
    neighbour_indices = np.take_along_axis(neighbour_indices, order, axis=1)
    neighbour_sims = np.take_along_axis(neighbour_sims, order, axis=1)

    # 3. Weights
    weights = similarities_to_weights(neighbour_sims, weighting, k, epsilon)

    # 4. MULTICLASS AGGREGATION (The Fix)
    unique_uoas = sorted(np.unique(y_train))
    num_queries = X_new.shape[0]
    # Initialize a matrix: [Number of Samples] x [Number of Unique UoAs]
    scores_per_uoa = np.zeros((num_queries, len(unique_uoas)))

    # Get the actual labels for the neighbours
    # If y_train is a Series, use .iloc; if array, use indexing
    y_train_array = np.array(y_train)
    neighbour_labels = y_train_array[neighbour_indices]

    for i, uoa in enumerate(unique_uoas):
        # Create a mask: Where does this neighbour match the current UoA?
        mask = (neighbour_labels == uoa)
        # Sum weights where the mask is True, divide by total weights for that row
        scores_per_uoa[:, i] = np.sum(weights * mask, axis=1) / np.sum(weights, axis=1)

    return scores_per_uoa, unique_uoas, neighbour_indices


def similarities_to_weights(neighbour_sims, w, k, epsilon):
    if w == "uniform":
        weights = np.ones_like(neighbour_sims)
    elif w == "inverse_distance":
        weights = 1.0 / (1.0 - neighbour_sims + epsilon)
    elif w == "inverse_distance_squared":
        weights = 1.0 / ((1.0 - neighbour_sims + epsilon) ** 2)
    elif w == "exponential_similarity":
        alpha = 20.0 # Standard alpha used in the exponential similarity function
        weights = np.exp(alpha * neighbour_sims)
    elif w == "rank":
        ranks = np.arange(1, k + 1, dtype=np.float32)
        # weights = 1.0 / ranks[None, :]
        weights = np.tile(1.0 / ranks, (neighbour_sims.shape[0], 1))

    return weights

