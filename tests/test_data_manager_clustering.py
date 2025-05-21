import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from tekoa.visualization.app_refactored_claude_components.data_manager import DataManager
from tekoa.utils.clustering_manager import ClusteringManager # For type checking if needed by mocks
from tekoa.utils.dimensionality_reducer import DimensionalityReducer # For type checking if needed by mocks

class TestDataManagerClustering(unittest.TestCase):

    def setUp(self):
        self.sample_raw_data = pd.DataFrame({
            'col1': np.random.rand(20),
            'col2': np.random.rand(20),
            'col3': ['cat{}'.format(i % 3) for i in range(20)]
        })

        self.patcher_st = patch('tekoa.visualization.app_refactored_claude_components.data_manager.st', new_callable=MagicMock)
        self.mock_st_global = self.patcher_st.start()
        self.addCleanup(self.patcher_st.stop)

        self.mock_st_global.session_state = {'processed_data': self.sample_raw_data.copy(), 'pipeline_results': {}}

        self.data_manager = DataManager()

        self.k_list = [2, 3]
        self.optimal_famd_components = 3
        self.mock_famd_components_df = pd.DataFrame(
            np.random.rand(len(self.sample_raw_data), self.optimal_famd_components),
            columns=[f'FAMD_{i+1}' for i in range(self.optimal_famd_components)]
        )

    def test_initialize_clustering_manager(self):
        result = self.data_manager.initialize_clustering_manager(self.mock_famd_components_df)
        self.assertTrue(result)
        self.assertIsInstance(self.data_manager.clustering_manager, ClusteringManager)
        self.assertTrue(self.data_manager.clustering_manager.data.equals(self.mock_famd_components_df))

        result_empty = self.data_manager.initialize_clustering_manager(pd.DataFrame())
        self.assertFalse(result_empty)
        self.assertIsNone(self.data_manager.clustering_manager)

        result_none = self.data_manager.initialize_clustering_manager(None)
        self.assertFalse(result_none)
        self.assertIsNone(self.data_manager.clustering_manager)

    @patch('tekoa.visualization.app_refactored_claude_components.data_manager.ClusteringManager')
    @patch('tekoa.visualization.app_refactored_claude_components.data_manager.DimensionalityReducer')
    @patch.object(DataManager, 'transform_data') # Patching DataManager's own method
    def test_run_all_clustering_analyses_success_no_rerun(self, mock_dm_transform_data, MockDimensionalityReducerClass, MockClusteringManagerClass):
        # This test ensures that if FAMD results are current, DataManager.perform_dimensionality_reduction is not called again.

        mock_dr_instance = MockDimensionalityReducerClass.return_value
        # Configure the DimensionalityReducer instance for the "no rerun" path:
        # 1. Its internal data matches the current processed data
        mock_dr_instance.data = self.mock_st_global.session_state['processed_data']
        # 2. It has FAMD results with the correct number of components
        mock_dr_instance.famd_results = {
            'transformed_data': pd.DataFrame(np.random.rand(len(self.sample_raw_data), self.optimal_famd_components))
        }

        mock_dm_transform_data.return_value = self.mock_famd_components_df
        mock_cm_instance = MockClusteringManagerClass.return_value

        # Spy on DataManager.perform_dimensionality_reduction to ensure it's NOT called
        with patch.object(self.data_manager, 'perform_dimensionality_reduction') as mock_dm_perform_dr_spy:
            success = self.data_manager.run_all_clustering_analyses(
                self.k_list,
                optimal_famd_components_count=self.optimal_famd_components,
                random_state=0
            )
            self.assertTrue(success)
            mock_dm_perform_dr_spy.assert_not_called() # Key: FAMD re-run was not needed

        mock_dm_transform_data.assert_called_once_with(method='famd', n_components=self.optimal_famd_components)
        MockClusteringManagerClass.assert_called_once_with(self.mock_famd_components_df)
        self.assertEqual(mock_cm_instance.run_clustering_pipeline.call_count, 3)
        mock_cm_instance.run_clustering_pipeline.assert_any_call(algorithm_type='kmeans', k_list=self.k_list, random_state=0)
        self.assertIn('clustering', self.mock_st_global.session_state['pipeline_results'])


    @patch('tekoa.visualization.app_refactored_claude_components.data_manager.ClusteringManager')
    @patch.object(DataManager, 'transform_data')
    @patch.object(DataManager, 'perform_dimensionality_reduction')
    @patch('tekoa.visualization.app_refactored_claude_components.data_manager.DimensionalityReducer') # To control the instance created by initialize_dimensionality_reducer
    def test_run_all_clustering_analyses_famd_rerun_needed(self, MockDimensionalityReducerClass, mock_dm_perform_dr, mock_dm_transform_data, MockClusteringManagerClass):
        # Configure the DimensionalityReducer instance that will be created to trigger a rerun
        mock_dr_instance = MockDimensionalityReducerClass.return_value
        mock_dr_instance.famd_results = None # This will cause 'needs_famd_rerun' to be true
        mock_dr_instance.data = self.mock_st_global.session_state['processed_data']


        mock_dm_perform_dr.return_value = {'method': 'famd', 'n_components': self.optimal_famd_components}
        mock_dm_transform_data.return_value = self.mock_famd_components_df
        mock_cm_instance = MockClusteringManagerClass.return_value

        success = self.data_manager.run_all_clustering_analyses(
            self.k_list,
            optimal_famd_components_count=self.optimal_famd_components,
            random_state=0
        )
        self.assertTrue(success)

        mock_dm_perform_dr.assert_called_once_with(
            method='famd',
            variables=None,
            n_components=self.optimal_famd_components
        )
        mock_dm_transform_data.assert_called_once_with(method='famd', n_components=self.optimal_famd_components)
        MockClusteringManagerClass.assert_called_once_with(self.mock_famd_components_df)
        self.assertEqual(mock_cm_instance.run_clustering_pipeline.call_count, 3)

    @patch('tekoa.visualization.app_refactored_claude_components.data_manager.DimensionalityReducer')
    def test_run_all_clustering_analyses_no_famd_optimal_components(self, MockDimensionalityReducerClass):
        mock_dr_instance = MockDimensionalityReducerClass.return_value
        mock_dr_instance.optimal_components = None
        mock_dr_instance.famd_results = None

        # Ensure pipeline_results also doesn't have it from a previous run
        self.mock_st_global.session_state['pipeline_results'] = {}


        result = self.data_manager.run_all_clustering_analyses(self.k_list, optimal_famd_components_count=None)
        self.assertFalse(result)
        self.mock_st_global.error.assert_any_call("FAMD optimal components not determined. Please run FAMD (and determine optimal components) on the Dimensionality Reduction page first.")

    @patch('tekoa.visualization.app_refactored_claude_components.data_manager.DimensionalityReducer')
    @patch.object(DataManager, 'perform_dimensionality_reduction', return_value=True)
    @patch.object(DataManager, 'transform_data', return_value=None)
    def test_run_all_clustering_analyses_transform_fails(self, mock_transform_data, mock_perform_dr, MockDimReducer):
        # Mock DimReducer instance attributes for the path leading up to transform_data
        mock_dr_instance = MockDimReducer.return_value
        mock_dr_instance.data = self.mock_st_global.session_state['processed_data']
        mock_dr_instance.famd_results = {'transformed_data': "dummy_data_not_none"} # Make it seem like FAMD ran

        result = self.data_manager.run_all_clustering_analyses(self.k_list, optimal_famd_components_count=self.optimal_famd_components)
        self.assertFalse(result)
        self.mock_st_global.error.assert_called_with("Could not retrieve FAMD components after transform_data. Ensure FAMD was run successfully.")

    def test_get_all_clustering_results(self):
        test_results = {'kmeans': {'k2': 'test_val'}}
        self.mock_st_global.session_state['pipeline_results']['clustering'] = test_results
        results = self.data_manager.get_all_clustering_results()
        self.assertEqual(results, test_results)

        self.mock_st_global.session_state['pipeline_results']['clustering'] = {}
        self.assertEqual(self.data_manager.get_all_clustering_results(), {})

        del self.mock_st_global.session_state['pipeline_results']['clustering']
        self.assertEqual(self.data_manager.get_all_clustering_results(), {})

    def test_get_cluster_labels_and_model_for_run_with_manager(self):
        self.data_manager.clustering_manager = MagicMock(spec=ClusteringManager)
        mock_labels = np.array([0,0,1,1])
        mock_model_obj = MagicMock()
        self.data_manager.clustering_manager.get_labels.return_value = mock_labels
        self.data_manager.clustering_manager.get_model.return_value = mock_model_obj

        labels = self.data_manager.get_cluster_labels_for_run('kmeans', 2)
        model = self.data_manager.get_cluster_model_for_run('kmeans', 2)

        self.data_manager.clustering_manager.get_labels.assert_called_with('kmeans', 2)
        np.testing.assert_array_equal(labels, mock_labels)
        self.data_manager.clustering_manager.get_model.assert_called_with('kmeans', 2)
        self.assertEqual(model, mock_model_obj)

    def test_get_cluster_labels_and_model_for_run_no_manager(self):
        self.data_manager.clustering_manager = None
        labels = self.data_manager.get_cluster_labels_for_run('kmeans', 2)
        model = self.data_manager.get_cluster_model_for_run('kmeans', 2)
        self.assertIsNone(labels)
        self.assertIsNone(model)

    @patch('tekoa.visualization.app_refactored_claude_components.data_manager.characterize_phenotypes')
    def test_characterize_selected_phenotypes_success(self, mock_characterize_phenotypes_util):
        # self.mock_st_global is available from setUp
        # self.data_manager is available from setUp

        algo_type = 'kmeans'
        n_clusters = 2
        variables_to_compare = ['Age', 'Outcome']

        # Mock inputs that characterize_selected_phenotypes will try to fetch
        mock_original_data = pd.DataFrame({
            'Age': [20, 25, 30, 35, 40, 45],
            'Outcome': [1, 0, 1, 0, 1, 0],
            'OtherVar': ['A', 'B', 'A', 'B', 'A', 'B']
        })
        mock_labels = np.array([0, 0, 1, 1, 0, 1])

        # Configure DataManager's internal getters
        self.data_manager.get_original_data = MagicMock(return_value=mock_original_data)
        self.data_manager.get_cluster_labels_for_run = MagicMock(return_value=mock_labels)

        # Configure the mock utility function
        mock_char_df = pd.DataFrame({
            'Variable': variables_to_compare,
            'TestType': ['ANOVA', 'Chi-Square'], # Example
            'PValue': [0.01, 0.04]
            # Minimal df for testing DataManager's role, not the util's full output
        })
        mock_characterize_phenotypes_util.return_value = mock_char_df

        # Call the method
        result_df = self.data_manager.characterize_selected_phenotypes(algo_type, n_clusters, variables_to_compare)

        # Assertions
        self.data_manager.get_original_data.assert_called_once()
        self.data_manager.get_cluster_labels_for_run.assert_called_once_with(algo_type, n_clusters)
        mock_characterize_phenotypes_util.assert_called_once_with(
            original_data=mock_original_data,
            labels=mock_labels,
            variables_to_compare=variables_to_compare
        )
        self.assertIsNotNone(result_df)
        pd.testing.assert_frame_equal(result_df, mock_char_df)

        # Check storage in session state
        # Ensure the path exists before asserting
        self.assertIn('clustering', self.mock_st_global.session_state['pipeline_results'])
        self.assertIn(algo_type, self.mock_st_global.session_state['pipeline_results']['clustering'])
        self.assertIn(n_clusters, self.mock_st_global.session_state['pipeline_results']['clustering'][algo_type])

        expected_storage_path = self.mock_st_global.session_state['pipeline_results']['clustering'][algo_type][n_clusters]['characterization_results']
        pd.testing.assert_frame_equal(expected_storage_path, mock_char_df)

    def test_characterize_selected_phenotypes_no_labels(self):
        # self.mock_st_global and self.data_manager from setUp

        self.data_manager.get_original_data = MagicMock(return_value=pd.DataFrame({'Age': [1,2]}))
        self.data_manager.get_cluster_labels_for_run = MagicMock(return_value=None) # Simulate no labels

        result_df = self.data_manager.characterize_selected_phenotypes('kmeans', 2, ['Age'])

        self.assertIsNone(result_df)
        self.mock_st_global.error.assert_called_with("Cluster labels for kmeans (k=2) not found. Please ensure clustering was run successfully.")

    def test_characterize_selected_phenotypes_no_variables(self):
        # self.mock_st_global and self.data_manager from setUp
        mock_original_data = pd.DataFrame({'Age': [20, 25, 30, 35, 40, 45]})
        mock_labels = np.array([0, 0, 1, 1, 0, 1])
        self.data_manager.get_original_data = MagicMock(return_value=mock_original_data)
        self.data_manager.get_cluster_labels_for_run = MagicMock(return_value=mock_labels)

        result_df = self.data_manager.characterize_selected_phenotypes('kmeans', 2, []) # Empty list of variables

        self.assertIsNotNone(result_df)
        self.assertTrue(result_df.empty)
        self.assertListEqual(list(result_df.columns), ['Variable', 'TestType', 'Statistic', 'PValue', 'CorrectedPValue', 'RejectNullFDR', 'EffectSize'])
        self.mock_st_global.warning.assert_called_with("Please select at least one variable to characterize.")


if __name__ == '__main__':
    unittest.main()
