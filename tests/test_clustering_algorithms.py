import unittest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import logging

# Attempt to import KMedoids for the PAM test logic
try:
    from sklearn_extra.cluster import KMedoids
    SKLEARN_EXTRA_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_EXTRA_AVAILABLE = False

from te_koa.utils.clustering_algorithms import perform_kmeans, perform_pam, perform_gmm

# Configure logger for capturing messages in tests
# Get the logger used in the module to be tested
clustering_logger = logging.getLogger('te_koa.utils.clustering_algorithms')

class TestClusteringAlgorithms(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.test_data = pd.DataFrame({
            'X': [1, 1, 0, 5, 5, 6],
            'Y': [0, 1, 0, 4, 5, 4]
        })
        self.n_samples = len(self.test_data)

    def test_perform_kmeans(self):
        """Test K-Means clustering."""
        model, labels = perform_kmeans(self.test_data, n_clusters=2, random_state=0)
        self.assertIsInstance(model, KMeans)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), self.n_samples)
        self.assertEqual(len(np.unique(labels)), 2)

    def test_perform_gmm(self):
        """Test Gaussian Mixture Model (GMM) clustering."""
        model, labels = perform_gmm(self.test_data, n_components=2, random_state=0)
        self.assertIsInstance(model, GaussianMixture)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), self.n_samples)
        # GMM might merge components, so unique labels could be less than n_components
        self.assertLessEqual(len(np.unique(labels)), 2)
        self.assertGreaterEqual(len(np.unique(labels)), 1) # Should find at least one cluster


    def test_perform_pam(self):
        """Test Partitioning Around Medoids (PAM) clustering."""
        if not SKLEARN_EXTRA_AVAILABLE:
            # Configure a handler to capture logs specifically for this test case
            # This is to avoid interference with global logging configuration
            # and to ensure we only assert logs for this specific scenario.
            with self.assertLogs(logger=clustering_logger, level='ERROR') as cm:
                model, labels = perform_pam(self.test_data, n_clusters=2, random_state=0)
                self.assertIsNone(model)
                self.assertIsInstance(labels, np.ndarray) # Should be an empty array
                self.assertEqual(len(labels), 0)
                self.assertTrue(
                    any("scikit-learn-extra is not installed" in message for message in cm.output) or
                    any("KMedoids model could not be initialized" in message for message in cm.output)
                )
        else:
            # If sklearn-extra IS available, perform basic checks
            # No error log is expected here
            # Temporarily disable logging capture for the success case or ensure logger level is high enough
            # to not capture INFO/DEBUG if any were added.
            # For this test, we only care about the ERROR case above.
            
            # Ensure no error logs are generated when sklearn-extra is available
            # We can do this by checking that no ERROR messages are logged during the call.
            with self.assertNoLogs(logger=clustering_logger, level='ERROR'):
                model, labels = perform_pam(self.test_data, n_clusters=2, random_state=0)
            
            self.assertIsNotNone(model)
            self.assertIsInstance(model, KMedoids)
            self.assertIsInstance(labels, np.ndarray)
            self.assertEqual(len(labels), self.n_samples)
            self.assertEqual(len(np.unique(labels)), 2)

if __name__ == '__main__':
    # Ensure the logger is configured to capture messages if tests are run directly
    # This is mainly for the PAM test when sklearn-extra is not installed
    logging.basicConfig() 
    clustering_logger.setLevel(logging.INFO) # Ensure it captures ERROR and INFO
    unittest.main()
