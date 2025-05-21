import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans # For type hinting
from sklearn.mixture import GaussianMixture # For type hinting
import logging

logger = logging.getLogger(__name__)

def calculate_silhouette_score(data: pd.DataFrame, labels: np.ndarray) -> float:
    """
    Calculates the Silhouette Score for a given dataset and cluster labels.

    The Silhouette Score measures how similar an object is to its own cluster
    (cohesion) compared to other clusters (separation). The score ranges from -1 to +1,
    where a high value indicates that the object is well matched to its own cluster
    and poorly matched to neighboring clusters.

    Args:
        data: Pandas DataFrame containing the features used for clustering.
        labels: Numpy array of cluster assignments for each data point.

    Returns:
        The silhouette score as a float. Returns 0.0 if the score cannot be
        calculated (e.g., if there's only one cluster or an error occurs).
    """
    n_labels = len(np.unique(labels))
    if n_labels < 2:
        logger.warning(
            f"Silhouette score cannot be calculated with less than 2 clusters. Found {n_labels} unique labels. "
            "Returning 0.0."
        )
        return 0.0
    if len(data) != len(labels):
        logger.error(
            f"Data and labels have different lengths ({len(data)} vs {len(labels)}). "
            "Silhouette score cannot be calculated. Returning 0.0."
        )
        return 0.0
    try:
        score = silhouette_score(data, labels)
        return score
    except ValueError as e:
        logger.error(f"Error calculating silhouette score: {e}. Returning 0.0.")
        return 0.0

def calculate_davies_bouldin_score(data: pd.DataFrame, labels: np.ndarray) -> float:
    """
    Calculates the Davies-Bouldin Index for a given dataset and cluster labels.

    The Davies-Bouldin Index is a measure of clustering quality, where a lower
    score indicates better clustering. It signifies the average similarity ratio
    of each cluster with its most similar cluster.

    Args:
        data: Pandas DataFrame containing the features used for clustering.
        labels: Numpy array of cluster assignments for each data point.

    Returns:
        The Davies-Bouldin score as a float. Returns 0.0 if the score
        cannot be calculated (e.g., if there's only one cluster or an error occurs).
        Note: A score of 0.0 from this function due to an error or single cluster
        should be interpreted carefully, as a true Davies-Bouldin score of 0 would imply perfect clustering.
    """
    n_labels = len(np.unique(labels))
    if n_labels < 2:
        logger.warning(
            f"Davies-Bouldin score cannot be calculated with less than 2 clusters. Found {n_labels} unique labels. "
            "Returning 0.0."
        )
        return 0.0
    if len(data) != len(labels):
        logger.error(
            f"Data and labels have different lengths ({len(data)} vs {len(labels)}). "
            "Davies-Bouldin score cannot be calculated. Returning 0.0."
        )
        return 0.0
    try:
        score = davies_bouldin_score(data, labels)
        return score
    except ValueError as e:
        logger.error(f"Error calculating Davies-Bouldin score: {e}. Returning 0.0.")
        return 0.0

def calculate_model_native_score(model: any, data: pd.DataFrame, model_type: str) -> float:
    """
    Calculates a native scoring metric for a given fitted clustering model.

    The metric depends on the type of model:
    - 'kmeans': Returns the model's inertia (sum of squared distances to closest centroid).
    - 'pam': Returns the model's inertia if available (e.g., KMedoids).
    - 'gmm': Returns the model's Bayesian Information Criterion (BIC).

    Args:
        model: The fitted clustering model object.
        data: The input data (pd.DataFrame). This is required for GMM's BIC calculation.
        model_type: A string indicating the type of model ('kmeans', 'pam', 'gmm').

    Returns:
        The native score of the model (float). Returns np.nan if the model_type
        is unknown, the score cannot be determined, or an error occurs.
    """
    model_type = model_type.lower()
    try:
        if model_type == 'kmeans':
            if hasattr(model, 'inertia_'):
                return float(model.inertia_)
            else:
                logger.warning("KMeans model does not have an 'inertia_' attribute. Returning np.nan.")
                return np.nan
        elif model_type == 'pam':
            if hasattr(model, 'inertia_'):
                return float(model.inertia_)
            else:
                logger.warning(
                    "PAM model does not have an 'inertia_' attribute. "
                    "This might be expected for some PAM implementations. Returning np.nan."
                )
                return np.nan
        elif model_type == 'gmm':
            if hasattr(model, 'bic') and callable(model.bic):
                return float(model.bic(data))
            else:
                logger.warning("GMM model does not have a callable 'bic' method. Returning np.nan.")
                return np.nan
        else:
            logger.error(f"Unknown model_type '{model_type}' provided. Cannot calculate native score. Returning np.nan.")
            return np.nan
    except Exception as e:
        logger.error(f"Error calculating native score for model_type '{model_type}': {e}. Returning np.nan.")
        return np.nan
