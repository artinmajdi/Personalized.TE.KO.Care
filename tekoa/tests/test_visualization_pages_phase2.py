"""
Unit tests for Phase-II Page Components in tekoa/visualization/pages.py
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call

# Import the page classes to be tested
from tekoa.visualization.pages import (
    AutoPhenotypingPage,
    PhenotypeExplorerPage,
    ValidationDashboardPage
)
# Import DataManager to mock it, not to use its real implementation
from tekoa.visualization.data_manager import DataManager


@pytest.fixture(autouse=True)
def global_page_mocks(mocker):
    """Apply global mocks for streamlit and other utilities for all page tests."""
    mock_st_session_state = {}

    mocker.patch('streamlit.session_state', mock_st_session_state)
    # It's often easier to mock specific st functions as needed,
    # but for a broad sweep like this, a dictionary of mocks can work.
    mock_st_functions = {
        'header': mocker.MagicMock(),
        'subheader': mocker.MagicMock(),
        'markdown': mocker.MagicMock(),
        'button': mocker.MagicMock(return_value=False), # Default to not clicked
        'spinner': mocker.MagicMock(),
        'info': mocker.MagicMock(),
        'warning': mocker.MagicMock(),
        'error': mocker.MagicMock(),
        'success': mocker.MagicMock(),
        'metric': mocker.MagicMock(),
        'selectbox': mocker.MagicMock(),
        'sidebar': mocker.MagicMock(), # Container for sidebar elements
        'pyplot': mocker.MagicMock(),
        'plotly_chart': mocker.MagicMock(),
        'expander': mocker.MagicMock(), # Context manager
        'dataframe': mocker.MagicMock(),
        'download_button': mocker.MagicMock(),
        'columns': mocker.MagicMock(return_value=(MagicMock(), MagicMock())), # Default to 2 columns
        'number_input': mocker.MagicMock(),
        'checkbox': mocker.MagicMock(return_value=False), # Default to not checked
        'progress': mocker.MagicMock(),
        'multiselect': mocker.MagicMock(return_value=[]), # Default to empty selection
        'text_input': mocker.MagicMock(return_value=""),
        'file_uploader': mocker.MagicMock(return_value=None),
        'rerun': mocker.MagicMock(),
        'stop': mocker.MagicMock(),
        'set_page_config': mocker.MagicMock(), # For app.py, but good to have if pages use it
        'image': mocker.MagicMock(),
    }
    
    # Mock sidebar elements by attaching them to the sidebar mock
    mock_st_functions['sidebar'].selectbox = mocker.MagicMock()
    mock_st_functions['sidebar'].text_input = mocker.MagicMock(return_value="")
    mock_st_functions['sidebar'].button = mocker.MagicMock(return_value=False)
    mock_st_functions['sidebar'].markdown = mocker.MagicMock()
    mock_st_functions['sidebar'].subheader = mocker.MagicMock()
    mock_st_functions['sidebar'].multiselect = mocker.MagicMock(return_value=[])

    # Enable context management for st.spinner and st.expander
    mock_st_functions['spinner'].__enter__ = mocker.MagicMock(return_value=None)
    mock_st_functions['spinner'].__exit__ = mocker.MagicMock(return_value=None)
    mock_st_functions['expander'].__enter__ = mocker.MagicMock(return_value=None)
    mock_st_functions['expander'].__exit__ = mocker.MagicMock(return_value=None)


    for func_name, mock_func in mock_st_functions.items():
        mocker.patch(f'streamlit.{func_name}', mock_func)
        # Also patch for pages.py if it imports st directly
        mocker.patch(f'tekoa.visualization.pages.st.{func_name}', mock_func, create=True)


    mocker.patch('matplotlib.pyplot', new_callable=MagicMock)
    mocker.patch('plotly.express', new_callable=MagicMock)
    mocker.patch('plotly.graph_objects.Figure', new_callable=MagicMock) # For radar chart in PhenotypeExplorerPage
    mocker.patch('tekoa.visualization.pages.Figure', new_callable=MagicMock) # Specifically for _create_radar_chart

    mocker.patch('tekoa.logger.info', MagicMock())
    mocker.patch('tekoa.logger.warning', MagicMock())
    mocker.patch('tekoa.logger.error', MagicMock())
    
    # Ensure session_state is a clean dict for each test
    # This is crucial because session_state persists across function calls in a real app
    # Re-patching 'streamlit.session_state' to a new dictionary for each test
    mocker.patch('streamlit.session_state', new_callable=dict)
    mocker.patch('tekoa.visualization.pages.st.session_state', new_callable=dict, create=True)


@pytest.fixture
def mock_data_manager(mocker):
    """Fixture to create a MagicMock for DataManager."""
    dm = mocker.MagicMock(spec=DataManager)
    dm.get_data.return_value = pd.DataFrame({'col1': [1,2], 'col2': [3,4]}) # Default non-empty data
    dm.perform_auto_phenotyping.return_value = {} # Default empty results
    dm.get_silhouette_plot_data.return_value = None
    dm.get_phenotype_name.side_effect = lambda x: f"Phenotype {x}" # Default name
    dm.save_phenotype_name.return_value = None
    dm.get_phenotype_radar_chart_data.return_value = {}
    dm.get_patients_for_phenotype.return_value = pd.DataFrame()
    dm.get_phenotype_stability_data.return_value = {}
    # Ensure session_state is accessible if DataManager tries to use it directly (it shouldn't ideally)
    # dm.st_session_state = streamlit.session_state # This would be problematic, ensure DM uses passed session_state or its own
    return dm

# --- AutoPhenotypingPage Tests ---

class TestAutoPhenotypingPage:

    def test_render_no_processed_data(self, mock_data_manager, global_page_mocks, mocker):
        """Test render when processed_data is None."""
        # Override default mock for this test
        mocker.patch('tekoa.visualization.pages.st.session_state', {'processed_data': None})

        AutoPhenotypingPage.render(mock_data_manager)
        
        streamlit.warning.assert_called_once_with(
            "Processed data is not available. Please complete the data preparation pipeline first."
        )
        streamlit.header.assert_called_once_with("Auto-Phenotyping: Discovering Patient Subgroups")

    def test_render_initial_state_with_data(self, mock_data_manager, global_page_mocks, mocker):
        """Test initial rendering when processed_data is available but no phenotyping done yet."""
        mocker.patch('tekoa.visualization.pages.st.session_state', {
            'processed_data': pd.DataFrame({'a': [1]}),
            # 'phenotyping_results': None # Implicitly None or not present
        })
        
        streamlit.button.return_value = False # Ensure "Find Phenotypes" is not clicked

        AutoPhenotypingPage.render(mock_data_manager)

        streamlit.header.assert_called_with("Auto-Phenotyping: Discovering Patient Subgroups")
        streamlit.markdown.assert_any_call(mocker.ANY) # For intro text
        streamlit.subheader.assert_called_with("Clustering Configuration")
        streamlit.selectbox.assert_called_once() # Clustering method
        streamlit.number_input.assert_called_once() # Number of clusters
        streamlit.button.assert_any_call("🚀 Find Phenotypes", use_container_width=True)
        streamlit.info.assert_called_with("Click 'Find Phenotypes' to start the auto-phenotyping process.")


    def test_render_find_phenotypes_button_click_success(self, mock_data_manager, global_page_mocks, mocker):
        """Test clicking 'Find Phenotypes' and successful phenotyping."""
        mocker.patch('tekoa.visualization.pages.st.session_state', {
            'processed_data': pd.DataFrame({'a': [1, 2, 3]}),
        })
        streamlit.button.configure_mock(side_effect=lambda label, **kwargs: label == "🚀 Find Phenotypes")
        
        mock_data_manager.get_data.return_value = pd.DataFrame({'a': [1, 2, 3]})
        phenotyping_output = {
            'cluster_labels': np.array([0, 1, 0]),
            'n_clusters': 2,
            'silhouette_score': 0.75,
            'silhouette_values': np.array([0.1, 0.8, 0.2])
        }
        mock_data_manager.perform_auto_phenotyping.return_value = phenotyping_output
        mock_data_manager.get_silhouette_plot_data.return_value = phenotyping_output # For simplicity

        AutoPhenotypingPage.render(mock_data_manager)

        mock_data_manager.perform_auto_phenotyping.assert_called_once()
        streamlit.spinner.__enter__.assert_called_once()
        streamlit.success.assert_called_with("Auto-phenotyping complete! Found 2 potential phenotypes.")
        
        # Check if results are displayed
        streamlit.metric.assert_any_call("Optimal Number of Clusters (Phenotypes)", 2)
        streamlit.metric.assert_any_call("Average Silhouette Score", "0.750")
        streamlit.pyplot.assert_called_once() # For silhouette plot

    def test_render_find_phenotypes_button_click_failure(self, mock_data_manager, global_page_mocks, mocker):
        """Test clicking 'Find Phenotypes' and phenotyping fails."""
        mocker.patch('tekoa.visualization.pages.st.session_state', {
            'processed_data': pd.DataFrame({'a': [1, 2, 3]}),
        })
        streamlit.button.configure_mock(side_effect=lambda label, **kwargs: label == "🚀 Find Phenotypes")
        
        mock_data_manager.get_data.return_value = pd.DataFrame({'a': [1, 2, 3]})
        mock_data_manager.perform_auto_phenotyping.return_value = {} # Empty dict indicates failure

        AutoPhenotypingPage.render(mock_data_manager)

        mock_data_manager.perform_auto_phenotyping.assert_called_once()
        streamlit.error.assert_called_with("Auto-phenotyping failed or returned no results. Check logs for details.")
        streamlit.pyplot.assert_not_called() # Plot should not be shown on failure

    def test_render_silhouette_plot_error(self, mock_data_manager, global_page_mocks, mocker):
        """Test error handling if silhouette plot data is bad or plotting fails."""
        mocker.patch('tekoa.visualization.pages.st.session_state', {
            'processed_data': pd.DataFrame({'a': [1]}),
            'phenotyping_results': { # Valid results to trigger plot section
                'n_clusters': 2, 'silhouette_score': 0.5, 'cluster_labels': np.array([0,1]),
                'silhouette_values': np.array([0.2, 0.8])
            }
        })
        # Mock get_silhouette_plot_data to return something that might cause an issue, or mock plt.figure to raise error
        mock_data_manager.get_silhouette_plot_data.return_value = {'n_clusters': 2} # Incomplete data
        
        # Or, to simulate error during plotting itself:
        mocker.patch('tekoa.visualization.pages.plt.subplots', side_effect=Exception("Plotting error"))

        AutoPhenotypingPage.render(mock_data_manager)
        streamlit.error.assert_any_call("Could not generate silhouette plot: Plotting error")


# --- PhenotypeExplorerPage Tests ---
class TestPhenotypeExplorerPage:

    def test_render_no_phenotyping_results(self, mock_data_manager, global_page_mocks, mocker):
        mocker.patch('tekoa.visualization.pages.st.session_state', {}) # No 'phenotyping_results'
        
        streamlit.button.return_value = False # Go to auto-phenotyping not clicked

        PhenotypeExplorerPage.render(mock_data_manager)
        streamlit.warning.assert_called_with(
            "Phenotyping has not been performed yet. Please go to the 'Auto-Phenotyping' page first."
        )
        streamlit.button.assert_called_with("Go to Auto-Phenotyping")

    def test_render_with_results_select_phenotype_and_name(self, mock_data_manager, global_page_mocks, mocker):
        pheno_results = {
            'n_clusters': 2, 
            'cluster_labels': np.array([0, 1, 0, 1]),
            'silhouette_score': 0.5
        }
        mocker.patch('tekoa.visualization.pages.st.session_state', {
            'phenotyping_results': pheno_results,
            'processed_data': pd.DataFrame({'f1': [1,2,3,4], 'f2': [5,6,7,8]})
        })
        
        mock_data_manager.get_data.return_value = pd.DataFrame({'f1': [1,2,3,4], 'f2': [5,6,7,8]})
        
        # Simulate selecting phenotype 0
        streamlit.sidebar.selectbox.return_value = 0
        # Simulate typing a new name and clicking save
        streamlit.sidebar.text_input.return_value = "New Pheno Name"
        # Make the "Save Name" button specific to the selected phenotype return True
        streamlit.sidebar.button.side_effect = lambda label, **kwargs: label == "Save Name" and kwargs.get('key') == "save_name_0"


        mock_data_manager.get_phenotype_name.side_effect = lambda x: "Phenotype 0" if x == 0 else "Phenotype 1"
        
        # Mock radar data
        radar_data_dict = {'f1': 0.5, 'f2': 0.8}
        mock_data_manager.get_phenotype_radar_chart_data.return_value = radar_data_dict
        
        # Mock patient data
        mock_data_manager.get_patients_for_phenotype.return_value = pd.DataFrame({'patient_id': ['p1', 'p3']})

        PhenotypeExplorerPage.render(mock_data_manager)

        streamlit.sidebar.selectbox.assert_called_once()
        mock_data_manager.get_phenotype_name.assert_any_call(0) # Called for selectbox format_func and display
        
        streamlit.sidebar.text_input.assert_called_with("Edit Phenotype Name", value="Phenotype 0")
        streamlit.sidebar.button.assert_any_call("Save Name", key="save_name_0")
        mock_data_manager.save_phenotype_name.assert_called_once_with(0, "New Pheno Name")
        streamlit.sidebar.success.assert_called_once() # For name saved
        streamlit.rerun.assert_called_once() # After saving name

        # This part might not be reached if rerun is called. For testing, we might need to control rerun or test in stages.
        # For now, assuming we can check calls before rerun:
        mock_data_manager.get_phenotype_radar_chart_data.assert_called_with(
            phenotype_id=0, 
            processed_data_with_clusters=mocker.ANY, # Check dataframe content if crucial
            features=['f1', 'f2']
        )
        # Check if _create_radar_chart was called (mocked as Figure for the page)
        tekoa.visualization.pages.Figure.assert_called_once() 
        
        streamlit.expander.assert_any_call("View Patient List")
        mock_data_manager.get_patients_for_phenotype.assert_called_with(phenotype_id=0, processed_data_with_clusters=mocker.ANY)
        streamlit.dataframe.assert_called_with(pd.DataFrame({'patient_id': ['p1', 'p3']}))

    # More tests for PhenotypeExplorerPage: radar chart error, no numeric cols, comparison section

# --- ValidationDashboardPage Tests ---
class TestValidationDashboardPage:

    def test_render_no_phenotyping_results(self, mock_data_manager, global_page_mocks, mocker):
        mocker.patch('tekoa.visualization.pages.st.session_state', {})
        ValidationDashboardPage.render(mock_data_manager)
        streamlit.warning.assert_called_with(
            "Phenotyping has not been performed yet. Please go to the 'Auto-Phenotyping' page first."
        )

    def test_render_with_results_calculate_stability(self, mock_data_manager, global_page_mocks, mocker):
        pheno_results = {'n_clusters': 3, 'silhouette_score': 0.6}
        mocker.patch('tekoa.visualization.pages.st.session_state', {
            'phenotyping_results': pheno_results,
            'validation_checklist': {} # Let it be initialized by the page
        })
        
        # Simulate "Calculate/Refresh Stability Metrics" button click
        streamlit.button.side_effect = lambda label, **kwargs: label == "🔄 Calculate/Refresh Stability Metrics"

        stability_output = {'avg_jaccard_score': 0.85, 'silhouette_bootstrap_distributions': [[0.5,0.6]]}
        mock_data_manager.get_phenotype_stability_data.return_value = stability_output

        ValidationDashboardPage.render(mock_data_manager)

        mock_data_manager.get_phenotype_stability_data.assert_called_once_with(pheno_results)
        streamlit.success.assert_called_with("Stability metrics updated.")
        
        # Check if stability metrics are displayed
        streamlit.metric.assert_any_call(label="Average Jaccard Index", value="0.850", delta="Highly Stable", delta_color="normal")
        streamlit.progress.assert_any_call(85)
        streamlit.plotly_chart.assert_called_once() # For bootstrap histogram

    def test_render_checklist_interaction(self, mock_data_manager, global_page_mocks, mocker):
        # Initialize session_state for the checklist items
        # The _initialize_checklist_state method in the page class should handle this,
        # so we start with an empty session_state or one that might be partially filled.
        initial_checklist_item_key = "item_distinct_interpretable"
        initial_checklist_item_label = "Phenotypes are distinct and interpretable."
        
        # Patch session_state to be a dictionary that we can inspect
        # The global_page_mocks already does this, so st.session_state is a dict
        
        # Simulate phenotyping results being present
        mocker.patch('tekoa.visualization.pages.st.session_state', {
            'phenotyping_results': {'n_clusters': 2},
            # 'validation_checklist' will be populated by _initialize_checklist_state
        })

        # Simulate one checkbox being checked by the user
        streamlit.checkbox.side_effect = lambda label, value, key: True if key == f"chk_{initial_checklist_item_key}" else False
        
        ValidationDashboardPage.render(mock_data_manager)

        # Verify all checkboxes were rendered (check for one specific call)
        streamlit.checkbox.assert_any_call(initial_checklist_item_label, value=False, key=f"chk_{initial_checklist_item_key}")
        
        # Verify checklist completion is displayed
        # Based on the side_effect, 1 item should be checked
        total_items = 5 # Assuming 5 default items
        streamlit.progress.assert_any_call(int((1/total_items)*100))
        streamlit.markdown.assert_any_call(f"**Checklist Completion: 1/{total_items} ({(1/total_items)*100:.0f}%)**")
        
        # Check that the session_state was updated by the checkbox call
        assert streamlit.session_state['validation_checklist'][initial_checklist_item_key]['checked'] is True

    def test_render_export_report_button(self, mock_data_manager, global_page_mocks, mocker):
        mocker.patch('tekoa.visualization.pages.st.session_state', {
            'phenotyping_results': {'n_clusters': 2},
            'stability_data': {'avg_jaccard_score': 0.8},
            'validation_checklist': {"item_distinct_interpretable": {"label": "L", "checked": True}}
        })
        
        streamlit.button.side_effect = lambda label, **kwargs: label == "📄 Export Phenotype Report (Placeholder)"

        # Mock json.dumps (or pd.io.json.dumps as used in page)
        mocker.patch('pandas.io.json.dumps', return_value="{}")


        ValidationDashboardPage.render(mock_data_manager)

        streamlit.download_button.assert_called_once_with(
            label="Download Phenotype Report (JSON)",
            data="{}",
            file_name="phenotype_validation_report.json",
            mime="application/json"
        )
        streamlit.success.assert_called_with("Phenotype report (JSON) prepared for download.")

```
