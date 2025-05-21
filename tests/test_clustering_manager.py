import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call # Added call for checking call arguments

from tekoa.utils.clustering_manager import ClusteringManager
import logging

# Suppress logging output during tests for cleaner test results
logging.disable(logging.CRITICAL)

class TestClusteringManager(unittest.TestCase):

    def setUp(self):
        """Set up common test data and manager instance."""
        self.test_famd_components = pd.DataFrame({
            'FAMD_1': np.random.rand(20),
            'FAMD_2': np.random.rand(20)
        })
        self.manager = ClusteringManager(self.test_famd_components)
        self.k_list = [2, 3]
        self.random_state = 0

    def _mock_clustering_outputs(self, n_clusters, data=None, random_state=None, n_components=None, init=None, n_init=None, method=None, covariance_type=None):
        """
        Helper to create mock model and labels.
        Accepts various args to be flexible with different perform_* functions.
        The actual clustering function arguments are used by the mock call,
        but n_clusters is what we need for label generation here.
        If n_components is passed (for GMM), use that for n_clusters.
        """
        k_for_labels = n_clusters if n_components is None else n_components
        mock_model = MagicMock()
        mock_model.name = f'mock_model_k{k_for_labels}'
        # Ensure labels are within the valid range for the number of clusters
        mock_labels = np.random.randint(0, k_for_labels, size=len(self.test_famd_components))
        return mock_model, mock_labels

    @patch('tekoa.utils.clustering_manager.calculate_model_native_score')
    @patch('tekoa.utils.clustering_manager.calculate_davies_bouldin_score')
    @patch('tekoa.utils.clustering_manager.calculate_silhouette_score')
    @patch('tekoa.utils.clustering_manager.perform_gmm')
    @patch('tekoa.utils.clustering_manager.perform_pam')
    @patch('tekoa.utils.clustering_manager.perform_kmeans')
    def test_run_clustering_pipeline_kmeans_success(self, mock_perform_kmeans, mock_perform_pam, mock_perform_gmm,
                                                 mock_calc_silhouette, mock_calc_davies_bouldin, mock_calc_native_score):
        # Configure mocks
        mock_perform_kmeans.side_effect = self._mock_clustering_outputs
        mock_calc_silhouette.return_value = 0.5
        mock_calc_davies_bouldin.return_value = 0.6
        mock_calc_native_score.return_value = 100.0

        # Run pipeline
        self.manager.run_clustering_pipeline('kmeans', self.k_list, random_state=self.random_state)

        # Assertions for perform_kmeans
        expected_kmeans_calls = [
            call(self.test_famd_components, n_clusters=k, random_state=self.random_state) for k in self.k_list
        ]
        mock_perform_kmeans.assert_has_calls(expected_kmeans_calls, any_order=False)
        self.assertEqual(mock_perform_kmeans.call_count, len(self.k_list))

        # Assertions for metric calculations
        self.assertEqual(mock_calc_silhouette.call_count, len(self.k_list))
        self.assertEqual(mock_calc_davies_bouldin.call_count, len(self.k_list))
        self.assertEqual(mock_calc_native_score.call_count, len(self.k_list))

        # Check results structure
        self.assertIn('kmeans', self.manager.results)
        for k in self.k_list:
            self.assertIn(k, self.manager.results['kmeans'])
            result_k = self.manager.results['kmeans'][k]
            self.assertTrue(hasattr(result_k['model'], 'name')) # Check if it's our mock model
            self.assertEqual(result_k['model'].name, f'mock_model_k{k}')
            self.assertEqual(len(result_k['labels']), len(self.test_famd_components))
            self.assertEqual(result_k['silhouette'], 0.5)
            self.assertEqual(result_k['davies_bouldin'], 0.6)
            self.assertEqual(result_k['native_score'], 100.0)

        # Assert other algorithms not called
        mock_perform_pam.assert_not_called()
        mock_perform_gmm.assert_not_called()

    @patch('tekoa.utils.clustering_manager.calculate_model_native_score')
    @patch('tekoa.utils.clustering_manager.calculate_davies_bouldin_score')
    @patch('tekoa.utils.clustering_manager.calculate_silhouette_score')
    @patch('tekoa.utils.clustering_manager.perform_gmm')
    @patch('tekoa.utils.clustering_manager.perform_pam')
    @patch('tekoa.utils.clustering_manager.perform_kmeans')
    def test_run_clustering_pipeline_pam_success(self, mock_perform_kmeans, mock_perform_pam, mock_perform_gmm,
                                              mock_calc_silhouette, mock_calc_davies_bouldin, mock_calc_native_score):
        mock_perform_pam.side_effect = self._mock_clustering_outputs
        mock_calc_silhouette.return_value = 0.4
        mock_calc_davies_bouldin.return_value = 0.7
        mock_calc_native_score.return_value = 110.0

        self.manager.run_clustering_pipeline('pam', self.k_list, random_state=self.random_state)

        expected_pam_calls = [
            call(self.test_famd_components, n_clusters=k, random_state=self.random_state) for k in self.k_list
        ]
        mock_perform_pam.assert_has_calls(expected_pam_calls, any_order=False)
        self.assertEqual(mock_perform_pam.call_count, len(self.k_list))

        self.assertEqual(mock_calc_silhouette.call_count, len(self.k_list))
        self.assertEqual(mock_calc_davies_bouldin.call_count, len(self.k_list))
        self.assertEqual(mock_calc_native_score.call_count, len(self.k_list))

        self.assertIn('pam', self.manager.results)
        for k in self.k_list:
            self.assertIn(k, self.manager.results['pam'])
            result_k = self.manager.results['pam'][k]
            self.assertEqual(result_k['model'].name, f'mock_model_k{k}')
            self.assertEqual(len(result_k['labels']), len(self.test_famd_components))
            self.assertEqual(result_k['silhouette'], 0.4)
            self.assertEqual(result_k['davies_bouldin'], 0.7)
            self.assertEqual(result_k['native_score'], 110.0)

        mock_perform_kmeans.assert_not_called()
        mock_perform_gmm.assert_not_called()

    @patch('tekoa.utils.clustering_manager.calculate_model_native_score')
    @patch('tekoa.utils.clustering_manager.calculate_davies_bouldin_score')
    @patch('tekoa.utils.clustering_manager.calculate_silhouette_score')
    @patch('tekoa.utils.clustering_manager.perform_gmm')
    @patch('tekoa.utils.clustering_manager.perform_pam')
    @patch('tekoa.utils.clustering_manager.perform_kmeans')
    def test_run_clustering_pipeline_gmm_success(self, mock_perform_kmeans, mock_perform_pam, mock_perform_gmm,
                                               mock_calc_silhouette, mock_calc_davies_bouldin, mock_calc_native_score):
        mock_perform_gmm.side_effect = self._mock_clustering_outputs
        mock_calc_silhouette.return_value = 0.3
        mock_calc_davies_bouldin.return_value = 0.8
        mock_calc_native_score.return_value = 120.0 # e.g. BIC

        self.manager.run_clustering_pipeline('gmm', self.k_list, random_state=self.random_state)

        expected_gmm_calls = [
            call(self.test_famd_components, n_components=k, random_state=self.random_state) for k in self.k_list
        ]
        mock_perform_gmm.assert_has_calls(expected_gmm_calls, any_order=False)
        self.assertEqual(mock_perform_gmm.call_count, len(self.k_list))

        self.assertEqual(mock_calc_silhouette.call_count, len(self.k_list))
        self.assertEqual(mock_calc_davies_bouldin.call_count, len(self.k_list))
        self.assertEqual(mock_calc_native_score.call_count, len(self.k_list))

        self.assertIn('gmm', self.manager.results)
        for k in self.k_list:
            self.assertIn(k, self.manager.results['gmm'])
            result_k = self.manager.results['gmm'][k]
            self.assertEqual(result_k['model'].name, f'mock_model_k{k}')
            self.assertEqual(len(result_k['labels']), len(self.test_famd_components))
            self.assertEqual(result_k['silhouette'], 0.3)
            self.assertEqual(result_k['davies_bouldin'], 0.8)
            self.assertEqual(result_k['native_score'], 120.0)

        mock_perform_kmeans.assert_not_called()
        mock_perform_pam.assert_not_called()

    @patch('tekoa.utils.clustering_manager.calculate_model_native_score')
    @patch('tekoa.utils.clustering_manager.calculate_davies_bouldin_score')
    @patch('tekoa.utils.clustering_manager.calculate_silhouette_score')
    @patch('tekoa.utils.clustering_manager.perform_pam')
    def test_run_clustering_pipeline_pam_failure(self, mock_perform_pam, mock_calc_silhouette,
                                             mock_calc_davies_bouldin, mock_calc_native_score):
        # Configure mock_perform_pam to simulate failure
        mock_perform_pam.return_value = (None, np.array([]))
        test_k_val = 2

        self.manager.run_clustering_pipeline('pam', [test_k_val], random_state=self.random_state)

        mock_perform_pam.assert_called_once_with(self.test_famd_components, n_clusters=test_k_val, random_state=self.random_state)

        # Metrics functions should not be called if model is None
        mock_calc_silhouette.assert_not_called()
        mock_calc_davies_bouldin.assert_not_called()
        mock_calc_native_score.assert_not_called()

        self.assertIn('pam', self.manager.results)
        self.assertIn(test_k_val, self.manager.results['pam'])
        result_k = self.manager.results['pam'][test_k_val]

        self.assertIsNone(result_k['model'])
        self.assertEqual(len(result_k['labels']), 0)
        self.assertTrue(np.isnan(result_k['silhouette']))
        self.assertTrue(np.isnan(result_k['davies_bouldin']))
        self.assertTrue(np.isnan(result_k['native_score']))

    def test_get_clustering_results(self):
        dummy_results = {'model': MagicMock(), 'labels': np.array([0, 1]), 'silhouette': 0.1}
        self.manager.results = {'kmeans': {2: dummy_results}}

        retrieved_results = self.manager.get_clustering_results('kmeans')
        self.assertEqual(retrieved_results, {'kmeans': {2: dummy_results}}['kmeans']) # get_clustering_results returns inner dict

        empty_results = self.manager.get_clustering_results('non_existent_algo')
        self.assertEqual(empty_results, {})

    def test_get_labels(self):
        dummy_labels = np.array([0, 1, 0, 1])
        self.manager.results = {'kmeans': {2: {'labels': dummy_labels}}}

        retrieved_labels = self.manager.get_labels('kmeans', 2)
        np.testing.assert_array_equal(retrieved_labels, dummy_labels)

        none_labels = self.manager.get_labels('kmeans', 99) # Non-existent k
        self.assertIsNone(none_labels)

        none_labels_algo = self.manager.get_labels('non_existent', 2) # Non-existent algo
        self.assertIsNone(none_labels_algo)


    def test_get_model(self):
        dummy_model = MagicMock()
        dummy_model.name = "my_dummy_gmm"
        self.manager.results = {'gmm': {3: {'model': dummy_model}}}

        retrieved_model = self.manager.get_model('gmm', 3)
        self.assertEqual(retrieved_model.name, "my_dummy_gmm")

        none_model = self.manager.get_model('gmm', 99) # Non-existent k
        self.assertIsNone(none_model)

        none_model_algo = self.manager.get_model('non_existent', 3) # Non-existent algo
        self.assertIsNone(none_model_algo)

if __name__ == '__main__':
    logging.disable(logging.NOTSET) # Re-enable logging if run directly for debugging
    unittest.main()
