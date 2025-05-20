import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import logging

try:
    from sklearn_extra.cluster import KMedoids
except ModuleNotFoundError:
    KMedoids = None  # Placeholder if not found

logger = logging.getLogger(__name__)

def perform_kmeans(data: pd.DataFrame, n_clusters: int, random_state: int = None) -> tuple[KMeans, np.ndarray]:
    """
    Performs K-Means clustering on the given data.

    Args:
        data: Pandas DataFrame containing the data to cluster.
        n_clusters: The number of clusters to form.
        random_state: Determines random number generation for centroid initialization.
                      Use an int to make the randomness deterministic.

    Returns:
        A tuple containing:
            - model: The fitted KMeans model object.
            - labels: Cluster labels for each data point.
    """
    model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state)
    model.fit(data)
    return model, model.labels_

def perform_pam(data: pd.DataFrame, n_clusters: int, random_state: int = None) -> tuple[any, np.ndarray]:
    """
    Performs Partitioning Around Medoids (PAM) clustering using KMedoids.

    Note: This function requires the 'scikit-learn-extra' package.

    Args:
        data: Pandas DataFrame containing the data to cluster.
        n_clusters: The number of clusters to form.
        random_state: Determines random number generation for medoid initialization.
                      Use an int to make the randomness deterministic.

    Returns:
        A tuple containing:
            - model: The fitted KMedoids model object, or None if scikit-learn-extra is not installed.
            - labels: Cluster labels for each data point, or an empty numpy array if scikit-learn-extra is not installed.
    """
    if KMedoids is None:
        logger.error("scikit-learn-extra is not installed. PAM clustering cannot be performed. "
                     "Please install it by running: pip install scikit-learn-extra")
        return None, np.array([])
    try:
        model = KMedoids(n_clusters=n_clusters, method='pam', init='k-medoids++', random_state=random_state)
        model.fit(data)
        return model, model.labels_
    except TypeError: # KMedoids might be None if import failed initially and wasn't caught by the first check (less likely with current setup)
        logger.error("KMedoids model could not be initialized, possibly due to scikit-learn-extra not being installed. "
                     "PAM clustering cannot be performed. Please install it by running: pip install scikit-learn-extra")
        return None, np.array([])


def perform_gmm(data: pd.DataFrame, n_components: int, random_state: int = None) -> tuple[GaussianMixture, np.ndarray]:
    """
    Performs Gaussian Mixture Model (GMM) clustering on the given data.

    Args:
        data: Pandas DataFrame containing the data to cluster.
        n_components: The number of mixture components (clusters).
        random_state: Determines random number generation for initialization.
                      Use an int to make the randomness deterministic.

    Returns:
        A tuple containing:
            - model: The fitted GaussianMixture model object.
            - labels: Predicted cluster labels for each data point.
    """
    model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
    model.fit(data)
    labels = model.predict(data)
    return model, labels
