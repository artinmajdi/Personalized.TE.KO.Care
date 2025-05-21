import unittest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from unittest.mock import Mock, patch # patch can be used for more complex scenarios

from tekoa.utils.clustering_validation import (
    calculate_silhouette_score,
    calculate_davies_bouldin_score,
    calculate_model_native_score
)
import logging

# Get the logger used in the module to be tested
validation_logger = logging.getLogger('tekoa.utils.clustering_validation')

class TestClusteringValidation(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.test_data = pd.DataFrame({
            'X': [1, 1, 0, 5, 5, 6, 7, 7, 8],
            'Y': [0, 1, 0, 4, 5, 4, 6, 7, 6]
        })
        # Define some sample labels
        self.labels_3_clusters = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.labels_1_cluster = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.n_samples = len(self.test_data)

    def test_calculate_silhouette_score(self):
        """Test Silhouette Score calculation."""
        score = calculate_silhouette_score(self.test_data, self.labels_3_clusters)
        self.assertIsInstance(score, float)
        self.assertTrue(-1 <= score <= 1)

        with self.assertLogs(logger=validation_logger, level='WARNING') as cm:
            score_1_cluster = calculate_silhouette_score(self.test_data, self.labels_1_cluster)
        self.assertEqual(score_1_cluster, 0.0)
        self.assertTrue(
            any("Silhouette score cannot be calculated with less than 2 clusters" in message for message in cm.output)
        )

    def test_calculate_davies_bouldin_score(self):
        """Test Davies-Bouldin Score calculation."""
        score = calculate_davies_bouldin_score(self.test_data, self.labels_3_clusters)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

        with self.assertLogs(logger=validation_logger, level='WARNING') as cm:
            score_1_cluster = calculate_davies_bouldin_score(self.test_data, self.labels_1_cluster)
        self.assertEqual(score_1_cluster, 0.0)
        self.assertTrue(
            any("Davies-Bouldin score cannot be calculated with less than 2 clusters" in message for message in cm.output)
        )

    def test_calculate_model_native_score_kmeans(self):
        """Test native score for KMeans (inertia)."""
        mock_kmeans_model = Mock(spec=KMeans)
        mock_kmeans_model.inertia_ = 123.45
        score = calculate_model_native_score(mock_kmeans_model, self.test_data, model_type='kmeans')
        self.assertEqual(score, 123.45)

    def test_calculate_model_native_score_pam_with_inertia(self):
        """Test native score for PAM with inertia."""
        mock_pam_model = Mock() # Generic mock
        mock_pam_model.inertia_ = 99.0
        score = calculate_model_native_score(mock_pam_model, self.test_data, model_type='pam')
        self.assertEqual(score, 99.0)

    def test_calculate_model_native_score_pam_without_inertia(self):
        """Test native score for PAM without inertia."""
        # Create a mock that will not have 'inertia_' unless explicitly set.
        # Using spec=[] means it has no attributes by default.
        mock_pam_model_no_inertia = Mock(spec=[])

        with self.assertLogs(logger=validation_logger, level='WARNING') as cm:
            score = calculate_model_native_score(mock_pam_model_no_inertia, self.test_data, model_type='pam')

        self.assertTrue(np.isnan(score))
        self.assertTrue(
            any("PAM model does not have an 'inertia_' attribute" in message for message in cm.output)
        )

    def test_calculate_model_native_score_gmm(self):
        """Test native score for GMM (BIC)."""
        mock_gmm_model = Mock(spec=GaussianMixture)
        # Mock the bic method, ensuring it's callable
        mock_gmm_model.bic = Mock(return_value=250.75)

        score = calculate_model_native_score(mock_gmm_model, self.test_data, model_type='gmm')
        self.assertEqual(score, 250.75)
        mock_gmm_model.bic.assert_called_once_with(self.test_data)

    def test_calculate_model_native_score_unknown(self):
        """Test native score for an unknown model type."""
        mock_model = Mock()
        with self.assertLogs(logger=validation_logger, level='ERROR') as cm:
            score = calculate_model_native_score(mock_model, self.test_data, model_type='unknown_type')
        self.assertTrue(np.isnan(score))
        self.assertTrue(
            any("Unknown model_type 'unknown_type' provided" in message for message in cm.output)
        )

if __name__ == '__main__':
    logging.basicConfig() # Configure basic logging for tests run directly
    validation_logger.setLevel(logging.INFO) # Ensure it captures necessary levels
    unittest.main()
