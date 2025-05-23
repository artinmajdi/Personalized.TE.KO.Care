import pandas as pd
import numpy as np
from typing import Any


from tekoa import logger
from tekoa.utils.clustering_algorithms import perform_kmeans, perform_pam, perform_gmm
from tekoa.utils.clustering_validation import (
    calculate_silhouette_score,
    calculate_davies_bouldin_score,
    calculate_model_native_score
)


class ClusteringManager:
    """
    Manages the execution of different clustering algorithms and stores their results.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the ClusteringManager with the dataset to be clustered.

        Args:
            data: Pandas DataFrame containing the data (e.g., FAMD components) to be used for clustering.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
        self.data = data
        self.results = {}  # Stores models, labels, and scores

    def run_clustering_pipeline(self, algorithm_type: str, k_list: list[int], random_state: int = None):
        """
        Runs a specified clustering algorithm for a list of k values (number of clusters),
        calculates validation metrics, and stores the results.

        Args:
            algorithm_type: The type of clustering algorithm to run.
                            Supported types: 'kmeans', 'pam', 'gmm'.
            k_list: A list of integers representing the number of clusters (or components for GMM) to try.
            random_state: Optional integer for random number generation reproducibility.
        """
        algorithm_type = algorithm_type.lower()
        if algorithm_type not in ['kmeans', 'pam', 'gmm']:
            logger.error(f"Unsupported algorithm_type: {algorithm_type}. Supported types are 'kmeans', 'pam', 'gmm'.")
            return

        logger.info(f"Running clustering pipeline for algorithm: {algorithm_type} with k_list: {k_list}")
        self.results.setdefault(algorithm_type, {})

        for k in k_list:
            if k <= 0:
                logger.warning(f"Number of clusters (k) must be positive. Skipping k={k} for {algorithm_type}.")
                continue

            logger.info(f"Running {algorithm_type} with k={k}")
            model = None
            labels = None

            if algorithm_type == 'kmeans':
                model, labels = perform_kmeans(self.data, n_clusters=k, random_state=random_state)
            elif algorithm_type == 'pam':
                model, labels = perform_pam(self.data, n_clusters=k, random_state=random_state)
                if model is None: # PAM might fail if scikit-learn-extra is not installed
                    logger.warning(f"PAM model was not created for k={k} (likely due to missing dependency). Skipping metrics calculation.")
                    self.results[algorithm_type][k] = {'model': None, 'labels': np.array([]), 'silhouette': np.nan, 'davies_bouldin': np.nan, 'native_score': np.nan}
                    continue
            elif algorithm_type == 'gmm':
                model, labels = perform_gmm(self.data, n_components=k, random_state=random_state)

            if model is not None and labels is not None and len(labels) > 0:
                # Ensure labels are not all the same for silhouette and davies-bouldin
                if len(np.unique(labels)) < 2:
                     s_score = 0.0 # As per our validation functions' behavior
                     db_score = 0.0 # As per our validation functions' behavior
                     logger.warning(f"Only one cluster found for {algorithm_type} with k={k}. Silhouette and Davies-Bouldin set to 0.")
                else:
                    s_score = calculate_silhouette_score(self.data, labels)
                    db_score = calculate_davies_bouldin_score(self.data, labels)

                n_score = calculate_model_native_score(model, self.data, model_type=algorithm_type)

                self.results[algorithm_type][k] = {
                    'model': model,
                    'labels': labels,
                    'silhouette': s_score,
                    'davies_bouldin': db_score,
                    'native_score': n_score
                }
                logger.info(f"Stored results for {algorithm_type} with k={k}: Silhouette={s_score:.3f}, DB={db_score:.3f}, Native={n_score:.3f}")
            elif model is None and algorithm_type != 'pam': # PAM already handled
                 logger.error(f"Model for {algorithm_type} with k={k} was not created. Results not stored.")
            elif labels is None or len(labels) == 0 and algorithm_type != 'pam':
                 logger.error(f"Labels for {algorithm_type} with k={k} were not generated. Results not stored.")


    def get_clustering_results(self, algorithm_type: str) -> dict:
        """
        Retrieves all stored results for a specified clustering algorithm.

        Args:
            algorithm_type: The type of algorithm ('kmeans', 'pam', 'gmm').

        Returns:
            A dictionary containing models, labels, and scores for different k values
            for the specified algorithm. Returns an empty dict if the algorithm_type is not found.
        """
        algorithm_type = algorithm_type.lower()
        return self.results.get(algorithm_type, {})

    def get_labels(self, algorithm_type: str, n_clusters: int) -> np.ndarray | None:
        """
        Retrieves the cluster labels for a specific algorithm and number of clusters.

        Args:
            algorithm_type: The type of algorithm ('kmeans', 'pam', 'gmm').
            n_clusters: The number of clusters for which to retrieve labels.

        Returns:
            A numpy array of cluster labels, or None if not found.
        """
        algorithm_type = algorithm_type.lower()
        if algorithm_type in self.results and n_clusters in self.results[algorithm_type]:
            return self.results[algorithm_type][n_clusters].get('labels')
        logger.warning(f"Labels not found for algorithm '{algorithm_type}' and n_clusters={n_clusters}.")
        return None

    def get_model(self, algorithm_type: str, n_clusters: int) -> Any | None:
        """
        Retrieves the fitted model object for a specific algorithm and number of clusters.

        Args:
            algorithm_type: The type of algorithm ('kmeans', 'pam', 'gmm').
            n_clusters: The number of clusters for which to retrieve the model.

        Returns:
            The fitted model object, or None if not found.
        """
        algorithm_type = algorithm_type.lower()
        if algorithm_type in self.results and n_clusters in self.results[algorithm_type]:
            return self.results[algorithm_type][n_clusters].get('model')
        logger.warning(f"Model not found for algorithm '{algorithm_type}' and n_clusters={n_clusters}.")
        return None
