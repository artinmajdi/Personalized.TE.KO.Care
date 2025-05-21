import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

# Mock streamlit before DataManager is imported, as DataManager might use st.* at import time
# or in its __init__ if it calls _initialize_session_state.
mock_st_session_state = MagicMock()
mock_st_error = MagicMock()

# Simulate a basic structure for session_state.pipeline_results
mock_st_session_state.pipeline_results = {}
mock_st_session_state.get = lambda key, default: getattr(mock_st_session_state, key, default)

# It's important to set up setdefault on the pipeline_results dictionary itself
# if the code directly calls setdefault on st.session_state.pipeline_results
# However, DataManager calls st.session_state.setdefault('clustering', {})
# which means we need to mock setdefault on mock_st_session_state directly.
def session_state_setdefault(key, default_value):
    if not hasattr(mock_st_session_state, key):
        setattr(mock_st_session_state, key, default_value)
    return getattr(mock_st_session_state, key)

mock_st_session_state.setdefault = session_state_setdefault


@patch('streamlit.session_state', new=mock_st_session_state)
@patch('streamlit.error', new=mock_st_error)
def data_manager_instance_creator():
    from tekoa.visualization.data_manager import DataManager # Import here after st is mocked
    dm = DataManager()
    # Simulate initial data loading for relevant attributes
    dm.data = pd.DataFrame({
        'numeric_col1': np.random.rand(20),
        'numeric_col2': np.random.rand(20),
        'numeric_col3': np.random.rand(20), # For PCA to have enough features
        'categorical_col1': ['A'] * 10 + ['B'] * 10
    })
    # Mock get_data to return a copy of this, consistent with how DataManager works
    dm.get_data = MagicMock(return_value=dm.data.copy())

    # Directly mock the attributes that would be initialized by other methods
    dm.dimensionality_reducer = MagicMock()
    dm.dimensionality_reducer.numeric_vars = ['numeric_col1', 'numeric_col2', 'numeric_col3']
    dm.dimensionality_reducer.categorical_vars = ['categorical_col1']
    
    # Ensure the data attribute of dimensionality_reducer is set for FAMD calls if it checks it
    dm.dimensionality_reducer.data = dm.data.copy()


    dm.clustering_manager = MagicMock()
    return dm

@pytest.fixture
def data_manager_instance(mocker):
    # Reset mocks for each test to ensure test isolation
    mock_st_session_state.reset_mock()
    mock_st_error.reset_mock()
    # Ensure pipeline_results is reset for each test run that uses the fixture
    mock_st_session_state.pipeline_results = {}
    
    # Mock the DataManager's internal initializers for its components
    # This prevents the actual DimensionalityReducer/ClusteringManager from being created
    mocker.patch('tekoa.visualization.data_manager.DimensionalityReducer', return_value=MagicMock(spec=True))
    mocker.patch('tekoa.visualization.data_manager.ClusteringManager', return_value=MagicMock(spec=True))

    dm = data_manager_instance_creator()
    
    # Re-assign the mocked reducer and clusterer after DataManager.__init__ might have run
    # (though with the patches above, it should use the MagicMocks)
    # We need to ensure the instance dm.dimensionality_reducer is the one we configure.
    # The patches on the class should mean dm.dimensionality_reducer (if created by init_dim_reducer)
    # would be a MagicMock. We just need to ensure it has numeric_vars.
    
    # If initialize_dimensionality_reducer is called by the method under test,
    # it will use the patched DimensionalityReducer.
    # Let's ensure the instance used by the method has the properties we need.
    # We can mock the initialize methods on DataManager itself.
    
    # The current DataManager.run_all_clustering_analyses calls self.initialize_dimensionality_reducer()
    # This method in DataManager sets self.dimensionality_reducer.
    # So, we need to control what this self.dimensionality_reducer becomes.
    # The easiest is to let it be created (it will be a MagicMock due to the class patch)
    # and then configure that MagicMock instance within each test *after* initialize_dimensionality_reducer is called.
    # OR, we can mock initialize_dimensionality_reducer itself.

    # For simplicity, let's mock initialize_dimensionality_reducer to do nothing,
    # and we will use the dm.dimensionality_reducer we already set up in data_manager_instance_creator.
    mocker.patch.object(dm, 'initialize_dimensionality_reducer', side_effect=lambda: None)
    # Do the same for initialize_clustering_manager if it's called before run_clustering_pipeline
    mocker.patch.object(dm, 'initialize_clustering_manager', side_effect=lambda data: True) # Simulate success


    return dm


# Test Case 1: FAMD success
@patch('streamlit.session_state', new=mock_st_session_state)
@patch('streamlit.error', new=mock_st_error)
def test_run_all_clustering_analyses_famd_success(data_manager_instance, mocker):
    dm = data_manager_instance

    # Configure the mocked dimensionality_reducer on the dm instance
    mock_famd_execution_results = {'some_famd_metric': 'value'} # Simulate what perform_famd might return
    dm.dimensionality_reducer.perform_famd.return_value = mock_famd_execution_results
    
    mock_famd_transformed_df = pd.DataFrame({'famd_comp1': [1,2,3], 'famd_comp2': [4,5,6]})
    
    # Mock DataManager's own transform_data method for this test
    # This is what run_all_clustering_analyses calls internally after perform_dimensionality_reduction
    mocker.patch.object(dm, 'transform_data', return_value=mock_famd_transformed_df)
    
    # Configure the mocked clustering_manager
    dm.clustering_manager.run_clustering_pipeline.return_value = None # It doesn't return anything
    dm.clustering_manager.get_clustering_results.return_value = {'kmeans_metric': 123}


    result = dm.run_all_clustering_analyses(k_list=[2,3], optimal_famd_components_count=2)

    assert result is True
    # perform_dimensionality_reduction is called by run_all_clustering_analyses
    # This internal method then calls dm.dimensionality_reducer.perform_famd
    # So we check the call on dm.dimensionality_reducer.perform_famd
    dm.dimensionality_reducer.perform_famd.assert_called_once_with(n_components=2)
    dm.dimensionality_reducer.perform_pca.assert_not_called()
    
    dm.transform_data.assert_called_once_with(method='famd', n_components=2)
    dm.initialize_clustering_manager.assert_called_once_with(mock_famd_transformed_df)
    assert dm.clustering_manager.run_clustering_pipeline.call_count == 3 # kmeans, pam, gmm

    assert 'clustering' in mock_st_session_state.pipeline_results
    assert 'kmeans' in mock_st_session_state.pipeline_results['clustering']


# Test Case 2: FAMD fails, PCA success
@patch('streamlit.session_state', new=mock_st_session_state)
@patch('streamlit.error', new=mock_st_error)
def test_run_all_clustering_analyses_famd_fails_pca_succeeds(data_manager_instance, mocker):
    dm = data_manager_instance

    # Mock FAMD failure: perform_famd raises error, or returns results that lead to transform_data returning None/empty
    dm.dimensionality_reducer.perform_famd.side_effect = ValueError("FAMD failed intentionally")
    
    # Mock PCA success
    mock_pca_execution_results = {'some_pca_metric': 'value'}
    dm.dimensionality_reducer.perform_pca.return_value = mock_pca_execution_results
    
    mock_pca_transformed_df = pd.DataFrame({'pca_comp1': [10,20,30], 'pca_comp2': [40,50,60]})

    # dm.transform_data needs to be configured for both FAMD (fail) and PCA (success) paths
    def transform_data_side_effect(method, n_components):
        if method == 'famd':
            return pd.DataFrame() # Simulate FAMD transform returning empty
        elif method == 'pca':
            return mock_pca_transformed_df
        return None
    mocker.patch.object(dm, 'transform_data', side_effect=transform_data_side_effect)

    dm.clustering_manager.run_clustering_pipeline.return_value = None
    dm.clustering_manager.get_clustering_results.return_value = {'pca_kmeans_metric': 456}

    # dm.dimensionality_reducer.numeric_vars has 3 vars. data has 20 samples.
    # PCA n_components will be min(2, 3, 19) = 2 if optimal_famd_components_count is 2
    expected_pca_n_components = 2

    result = dm.run_all_clustering_analyses(k_list=[2,3], optimal_famd_components_count=2)

    assert result is True
    dm.dimensionality_reducer.perform_famd.assert_called_once_with(n_components=2)
    dm.dimensionality_reducer.perform_pca.assert_called_once_with(variables=None, n_components=expected_pca_n_components, standardize=True)
    
    dm.transform_data.assert_any_call(method='famd', n_components=2)
    dm.transform_data.assert_any_call(method='pca', n_components=expected_pca_n_components)
    
    dm.initialize_clustering_manager.assert_called_once_with(mock_pca_transformed_df)
    assert dm.clustering_manager.run_clustering_pipeline.call_count == 3

    assert 'clustering' in mock_st_session_state.pipeline_results
    assert 'kmeans' in mock_st_session_state.pipeline_results['clustering']


# Test Case 3: Both FAMD and PCA fail
@patch('streamlit.session_state', new=mock_st_session_state)
@patch('streamlit.error', new=mock_st_error) # mock_st_error is already our global mock
def test_run_all_clustering_analyses_famd_pca_fail(data_manager_instance, mocker, caplog):
    dm = data_manager_instance
    caplog.set_level("ERROR") # Capture ERROR level logs

    # Mock FAMD failure
    dm.dimensionality_reducer.perform_famd.side_effect = ValueError("FAMD failed intentionally for test")
    # Mock PCA failure
    dm.dimensionality_reducer.perform_pca.side_effect = ValueError("PCA failed intentionally for test")

    # Mock transform_data to return None/empty for both methods
    def transform_data_side_effect(method, n_components):
        return pd.DataFrame() # Empty DataFrame
    mocker.patch.object(dm, 'transform_data', side_effect=transform_data_side_effect)

    # optimal_famd_components_count = 2. numeric_vars=3. samples=20.
    # expected_pca_n_components = min(2,3,19) = 2
    expected_pca_n_components = 2

    result = dm.run_all_clustering_analyses(k_list=[2,3], optimal_famd_components_count=2)

    assert result is False
    dm.dimensionality_reducer.perform_famd.assert_called_once_with(n_components=2)
    dm.dimensionality_reducer.perform_pca.assert_called_once_with(variables=None, n_components=expected_pca_n_components, standardize=True)
    
    dm.transform_data.assert_any_call(method='famd', n_components=2)
    dm.transform_data.assert_any_call(method='pca', n_components=expected_pca_n_components)
    
    mock_st_error.assert_called_once_with("Clustering cannot proceed as dimensionality reduction (FAMD and PCA) failed to produce usable components.")
    dm.initialize_clustering_manager.assert_not_called()
    
    assert "Both FAMD and PCA failed to produce usable components. Clustering will be skipped." in caplog.text


# Additional test: FAMD fails, PCA succeeds, optimal_famd_components_count is None
@patch('streamlit.session_state', new=mock_st_session_state)
@patch('streamlit.error', new=mock_st_error)
def test_run_all_clustering_analyses_famd_fails_pca_succeeds_no_optimal_count(data_manager_instance, mocker):
    dm = data_manager_instance

    dm.dimensionality_reducer.perform_famd.side_effect = ValueError("FAMD failed")
    
    mock_pca_execution_results = {'some_pca_metric': 'value'}
    dm.dimensionality_reducer.perform_pca.return_value = mock_pca_execution_results
    
    mock_pca_transformed_df = pd.DataFrame({'pca_comp1': [10,20,30], 'pca_comp2': [40,50,60]})

    def transform_data_side_effect(method, n_components):
        if method == 'famd':
            # This might not even be called if perform_famd fails early in the actual DataManager.perform_dim_reduction
            # Or it might be called and return empty. Let's assume it returns empty.
            return pd.DataFrame() 
        elif method == 'pca':
            return mock_pca_transformed_df
        return None
    mocker.patch.object(dm, 'transform_data', side_effect=transform_data_side_effect)

    dm.clustering_manager.run_clustering_pipeline.return_value = None
    dm.clustering_manager.get_clustering_results.return_value = {'pca_kmeans_metric': 789}

    # When optimal_famd_components_count is None:
    # PCA n_components = min(10, len(numeric_vars), len(data)-1)
    # numeric_vars = 3, data_len = 20. So, min(10, 3, 19) = 3
    expected_pca_n_components = 3

    result = dm.run_all_clustering_analyses(k_list=[2,3], optimal_famd_components_count=None)

    assert result is True
    # FAMD is attempted. Since optimal_famd_components_count is None, the FAMD call inside
    # perform_dimensionality_reduction might not happen if it strictly requires n_components.
    # The current code for run_all_clustering_analyses:
    # if actual_optimal_famd_components_count is None: raises error for FAMD.
    # So, perform_famd should not be called with n_components=None if it's strict.
    # Let's check the call to perform_famd itself.
    # The current structure calls perform_dimensionality_reduction(method='famd', n_components=actual_optimal_famd_components_count)
    # If actual_optimal_famd_components_count is None, FAMD part raises ValueError.
    # So, dm.dimensionality_reducer.perform_famd will not be called.
    # dm.dimensionality_reducer.perform_famd.assert_not_called() # This depends on internal logic of perform_dimensionality_reduction
    # Let's trace: run_all_clustering_analyses gets actual_optimal_famd_components_count = None.
    # Then it tries FAMD. Inside the try block for FAMD:
    # if actual_optimal_famd_components_count is None: logger.error; raise ValueError.
    # So, dm.perform_dimensionality_reduction(method='famd'...) is NOT called with n_components=None.
    # The ValueError is caught. Then PCA path is taken.
    
    # So, perform_famd on the reducer mock should not be called.
    dm.dimensionality_reducer.perform_famd.assert_not_called()


    dm.dimensionality_reducer.perform_pca.assert_called_once_with(variables=None, n_components=expected_pca_n_components, standardize=True)
    
    # transform_data for FAMD might not be called if perform_famd itself isn't called or fails very early
    # Based on the logic, if FAMD part raises ValueError before calling self.perform_dimensionality_reduction('famd'...),
    # then self.transform_data('famd'...) also won't be called.
    # Let's check that transform_data was called for PCA.
    dm.transform_data.assert_called_once_with(method='pca', n_components=expected_pca_n_components)
    
    dm.initialize_clustering_manager.assert_called_once_with(mock_pca_transformed_df)
    assert dm.clustering_manager.run_clustering_pipeline.call_count == 3
```
