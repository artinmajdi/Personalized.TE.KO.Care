"""
Unit tests for Phase-II (phenotyping) methods in DataManager.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

# Mock streamlit before importing DataManager
# This is a common pattern: mock global dependencies first.
mock_st = MagicMock()
mock_st.session_state = {}

# Patch streamlit where it's imported by the modules under test
patches = {
    'streamlit': mock_st,
    'streamlit.logger': MagicMock(), # if logger is used directly via st.logger
}

# Apply patches using a context manager or globally if careful
# For pytest, it's often cleaner to use fixtures or autouse=True for module-level mocks.
# However, direct patching like this is also common.
# We need to ensure these modules see the mocked streamlit when they are imported.
# One way is to patch sys.modules.
# For simplicity in this environment, we'll assume DataManager can be imported after this mock setup.
# Or, more robustly, use pytest-mock's mocker fixture.

# If DataManager is in tekoa.visualization.data_manager
# We need to ensure that when data_manager.py does 'import streamlit as st', it gets our mock_st.
# This is typically handled by pytest-mock's mocker fixture by patching the specific module.
# For now, we'll proceed assuming direct patching or mocker fixture in actual test execution.

from tekoa.visualization.data_manager import DataManager
from tekoa.configuration.params import DatasetNames # Assuming this is needed

# Mocking sklearn placeholders used in DataManager
# These mocks will return predefined values to make tests deterministic.
MockKMeans = MagicMock()
mock_silhouette_score = MagicMock()
mock_silhouette_samples = MagicMock()

@pytest.fixture(autouse=True) # Automatically use this fixture for all tests
def global_mocks(mocker):
    """Apply global mocks for streamlit and sklearn components."""
    mocker.patch('streamlit.session_state', new_callable=lambda: {}) # Reset session_state for each test
    mocker.patch('streamlit.spinner', MagicMock()) # Mock spinner context manager
    mocker.patch('tekoa.visualization.data_manager.st', new=mock_st) # Ensure DataManager sees the mocked st

    # Patching the sklearn imports within data_manager.py
    mocker.patch('tekoa.visualization.data_manager.KMeans', new=MockKMeans)
    mocker.patch('tekoa.visualization.data_manager.silhouette_score', new=mock_silhouette_score)
    mocker.patch('tekoa.visualization.data_manager.silhouette_samples', new=mock_silhouette_samples)
    mocker.patch('tekoa.logger.info', MagicMock()) # Mock logger calls
    mocker.patch('tekoa.logger.warning', MagicMock())
    mocker.patch('tekoa.logger.error', MagicMock())


@pytest.fixture
def data_manager_instance(mocker):
    """Fixture to create a DataManager instance with minimal setup."""
    # Mock DataLoader within DataManager if its methods are called indirectly or during init
    # For these specific tests, we might not need a full DataLoader mock if we set data attributes directly.
    dm = DataManager()
    # Initialize some basic attributes that might be expected
    dm.data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    dm.dictionary = pd.DataFrame({'Variable': ['feature1', 'feature2'], 'Description': ['Desc1', 'Desc2']})
    mock_st.session_state = {} # Ensure session_state is clean for each test using this fixture
    return dm

@pytest.fixture
def sample_processed_data():
    """Sample processed data for phenotyping."""
    return pd.DataFrame({
        'feat1': np.random.rand(20),
        'feat2': np.random.rand(20),
        'feat3': np.random.rand(20)
    }, index=[f'patient_{i}' for i in range(20)])


# --- Test Cases ---

def test_perform_auto_phenotyping_specified_k(data_manager_instance, sample_processed_data, mocker):
    dm = data_manager_instance
    mock_st.session_state = {} # Start with a clean session state

    # Configure mocks for sklearn placeholders
    expected_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    MockKMeans.return_value.fit_predict.return_value = expected_labels
    mock_silhouette_score.return_value = 0.5
    mock_silhouette_samples.return_value = np.random.rand(len(sample_processed_data))

    clustering_params = {'n_clusters': 2, 'method': 'kmeans', 'random_state': 42}
    results = dm.perform_auto_phenotyping(sample_processed_data, clustering_params)

    MockKMeans.assert_called_once_with(n_clusters=2, random_state=42, n_init='auto')
    MockKMeans.return_value.fit_predict.assert_called_once_with(sample_processed_data)
    mock_silhouette_score.assert_called_once_with(sample_processed_data, expected_labels)

    assert 'phenotyping_results' in mock_st.session_state
    assert mock_st.session_state['phenotyping_results']['n_clusters'] == 2
    assert mock_st.session_state['phenotyping_results']['silhouette_score'] == 0.5
    assert np.array_equal(mock_st.session_state['phenotyping_results']['cluster_labels'], expected_labels)
    assert 'phenotyping' in mock_st.session_state['pipeline_results']
    assert mock_st.session_state['pipeline_results']['phenotyping']['n_clusters'] == 2

def test_perform_auto_phenotyping_auto_k(data_manager_instance, sample_processed_data, mocker):
    dm = data_manager_instance
    mock_st.session_state = {}

    # Mock behavior for auto-k detection (e.g., trying k=2,3,4,5)
    # For k=2
    labels_k2 = np.array([0]*10 + [1]*10)
    score_k2 = 0.6
    # For k=3 (assume this is better)
    labels_k3 = np.array([0]*7 + [1]*7 + [2]*6)
    score_k3 = 0.7
    # For k=4
    labels_k4 = np.array([0]*5 + [1]*5 + [2]*5 + [3]*5)
    score_k4 = 0.55
     # For k=5
    labels_k5 = np.array([0]*4 + [1]*4 + [2]*4 + [3]*4 + [4]*4)
    score_k5 = 0.5


    mock_kmeans_instance_k2 = MagicMock()
    mock_kmeans_instance_k2.fit_predict.return_value = labels_k2
    mock_kmeans_instance_k3 = MagicMock()
    mock_kmeans_instance_k3.fit_predict.return_value = labels_k3
    mock_kmeans_instance_k4 = MagicMock()
    mock_kmeans_instance_k4.fit_predict.return_value = labels_k4
    mock_kmeans_instance_k5 = MagicMock()
    mock_kmeans_instance_k5.fit_predict.return_value = labels_k5


    MockKMeans.side_effect = [
        mock_kmeans_instance_k2, mock_kmeans_instance_k3, mock_kmeans_instance_k4, mock_kmeans_instance_k5
    ]
    mock_silhouette_score.side_effect = [score_k2, score_k3, score_k4, score_k5, score_k3] # score_k3 for final call after best k is chosen

    # Samples for the final call after choosing k=3
    mock_silhouette_samples.return_value = np.random.rand(len(sample_processed_data))


    clustering_params = {'method': 'kmeans', 'random_state': 42} # n_clusters is None
    results = dm.perform_auto_phenotyping(sample_processed_data, clustering_params)

    assert MockKMeans.call_count == 4 # Called for k=2,3,4,5
    # Check that the best k (k=3) was chosen
    assert results['n_clusters'] == 3
    assert results['silhouette_score'] == score_k3
    assert np.array_equal(results['cluster_labels'], labels_k3)
    assert 'phenotyping_results' in mock_st.session_state
    assert mock_st.session_state['phenotyping_results']['n_clusters'] == 3

def test_perform_auto_phenotyping_empty_data(data_manager_instance):
    dm = data_manager_instance
    mock_st.session_state = {}
    empty_df = pd.DataFrame()
    results = dm.perform_auto_phenotyping(empty_df)
    assert results == {}
    assert 'phenotyping_results' not in mock_st.session_state


def test_get_silhouette_plot_data_valid(data_manager_instance):
    dm = data_manager_instance
    mock_st.session_state['phenotyping_results'] = {
        'silhouette_values': np.array([0.1, 0.2, 0.3]),
        'cluster_labels': np.array([0, 1, 0]),
        'n_clusters': 2,
        'silhouette_score': 0.2
    }
    plot_data = dm.get_silhouette_plot_data()
    assert plot_data is not None
    assert np.array_equal(plot_data['silhouette_values'], np.array([0.1, 0.2, 0.3]))
    assert plot_data['n_clusters'] == 2

def test_get_silhouette_plot_data_missing_results(data_manager_instance):
    dm = data_manager_instance
    mock_st.session_state = {} # No phenotyping_results
    plot_data = dm.get_silhouette_plot_data()
    assert plot_data is None

    mock_st.session_state['phenotyping_results'] = {} # Incomplete results
    plot_data = dm.get_silhouette_plot_data()
    assert plot_data is None


def test_get_phenotype_radar_chart_data(data_manager_instance, sample_processed_data):
    dm = data_manager_instance
    labels = np.array([0]*10 + [1]*10) # Two clusters
    data_with_clusters = sample_processed_data.copy()
    data_with_clusters['cluster_labels'] = labels

    # Test for phenotype 0
    radar_data_p0 = dm.get_phenotype_radar_chart_data(0, data_with_clusters)
    assert radar_data_p0 is not None
    assert len(radar_data_p0) == 3 # feat1, feat2, feat3
    for col in ['feat1', 'feat2', 'feat3']:
        assert np.isclose(radar_data_p0[col], sample_processed_data[data_with_clusters['cluster_labels'] == 0][col].mean())

    # Test with specific features
    radar_data_p1_specific = dm.get_phenotype_radar_chart_data(1, data_with_clusters, features=['feat1', 'feat3'])
    assert radar_data_p1_specific is not None
    assert len(radar_data_p1_specific) == 2
    assert 'feat2' not in radar_data_p1_specific

    # Test invalid phenotype ID
    assert dm.get_phenotype_radar_chart_data(99, data_with_clusters) is None
    # Test with data missing 'cluster_labels'
    assert dm.get_phenotype_radar_chart_data(0, sample_processed_data) is None # No 'cluster_labels'

def test_get_patients_for_phenotype(data_manager_instance, sample_processed_data):
    dm = data_manager_instance
    labels = np.array([0]*5 + [1]*8 + [0]*7) # Phenotype 0 has 12, Phenotype 1 has 8
    data_with_clusters = sample_processed_data.copy()
    data_with_clusters['cluster_labels'] = labels

    patients_p0 = dm.get_patients_for_phenotype(0, data_with_clusters)
    assert patients_p0 is not None
    assert len(patients_p0) == 12
    expected_indices_p0 = sample_processed_data.index[labels == 0]
    assert all(idx in patients_p0.index for idx in expected_indices_p0)


    patients_p1 = dm.get_patients_for_phenotype(1, data_with_clusters)
    assert patients_p1 is not None
    assert len(patients_p1) == 8

    # Test invalid phenotype ID
    patients_p99 = dm.get_patients_for_phenotype(99, data_with_clusters)
    assert patients_p99 is not None
    assert patients_p99.empty

    # Test with data missing 'cluster_labels'
    assert dm.get_patients_for_phenotype(0, sample_processed_data) is None


def test_save_and_get_phenotype_name(data_manager_instance):
    dm = data_manager_instance
    mock_st.session_state = {} # Clean session state

    # Test default name
    assert dm.get_phenotype_name(0) == "Phenotype 0"
    assert dm.get_phenotype_name(1) == "Phenotype 1"

    dm.save_phenotype_name(0, "High Pain Group")
    assert 'phenotype_names' in mock_st.session_state
    assert mock_st.session_state['phenotype_names'][0] == "High Pain Group"
    assert dm.get_phenotype_name(0) == "High Pain Group"

    # Check another phenotype still has default name
    assert dm.get_phenotype_name(1) == "Phenotype 1"

    dm.save_phenotype_name(1, "Low Pain Group")
    assert mock_st.session_state['phenotype_names'][1] == "Low Pain Group"
    assert dm.get_phenotype_name(1) == "Low Pain Group"


def test_get_phenotype_stability_data(data_manager_instance, sample_processed_data):
    dm = data_manager_instance
    mock_st.session_state = {} # Clean session state

    # Test when phenotyping_results are missing
    assert dm.get_phenotype_stability_data() is None

    # Setup mock phenotyping_results (as done by perform_auto_phenotyping)
    mock_st.session_state['phenotyping_results'] = {
        'cluster_labels': np.array([0, 1, 0, 1]),
        'n_clusters': 2,
        'silhouette_score': 0.5,
        'silhouette_values': np.array([0.1,0.6,0.2,0.7]),
        'params': {'method': 'kmeans', 'n_clusters': 2}
    }
    dm.processed_data = sample_processed_data # Ensure dm has processed_data

    # The actual stability calculation is mocked in DataManager,
    # so this test mainly ensures the flow and session state updates.
    # The mocked np.random.rand will be called.
    with patch('numpy.random.rand', return_value=np.array([0.8, 0.85])) as mock_rand, \
         patch('numpy.random.normal', return_value=np.array([0.45]*50)) as mock_normal:

        stability_results = dm.get_phenotype_stability_data()

        assert stability_results is not None
        assert 'avg_jaccard_score' in stability_results
        assert np.isclose(stability_results['avg_jaccard_score'], 0.825) # (0.8+0.85)/2
        assert 'pipeline_results' in mock_st.session_state
        assert 'phenotyping' in mock_st.session_state['pipeline_results']
        assert 'stability' in mock_st.session_state['pipeline_results']['phenotyping']
        assert np.isclose(mock_st.session_state['pipeline_results']['phenotyping']['stability']['avg_jaccard_score'], 0.825)
        mock_rand.assert_called_once() # Called once for jaccard scores
        # mock_normal should be called n_clusters times
        assert mock_normal.call_count == mock_st.session_state['phenotyping_results']['n_clusters']

def test_perform_auto_phenotyping_single_cluster_fallback(data_manager_instance, sample_processed_data, mocker):
    dm = data_manager_instance
    mock_st.session_state = {}

    # Mock KMeans to produce only one cluster label
    single_cluster_labels = np.zeros(len(sample_processed_data), dtype=int)
    MockKMeans.return_value.fit_predict.return_value = single_cluster_labels
    
    # silhouette_score would raise error if only one cluster, so it shouldn't be called directly
    # or if called, it should be handled. DataManager handles this.
    # mock_silhouette_score should not be called if len(np.unique(labels)) < 2
    # mock_silhouette_samples will be called but its result for single cluster is zeros
    mock_silhouette_samples.return_value = np.zeros(len(sample_processed_data))


    clustering_params = {'n_clusters': 1, 'method': 'kmeans'} # Force 1 cluster
    results = dm.perform_auto_phenotyping(sample_processed_data, clustering_params)

    MockKMeans.assert_called_once_with(n_clusters=1, random_state=None, n_init='auto') # random_state is None if not in params
    mock_silhouette_score.assert_not_called() # Crucial: silhouette_score is ill-defined for 1 cluster

    assert results['n_clusters'] == 1
    assert results['silhouette_score'] == -1 # Default indicator for problematic silhouette
    assert np.array_equal(results['cluster_labels'], single_cluster_labels)
    assert np.all(results['silhouette_values'] == 0) # Should be all zeros

    assert 'phenotyping_results' in mock_st.session_state
    assert mock_st.session_state['phenotyping_results']['silhouette_score'] == -1
    assert 'phenotyping' in mock_st.session_state['pipeline_results']
    assert mock_st.session_state['pipeline_results']['phenotyping']['silhouette_score'] == -1
```
