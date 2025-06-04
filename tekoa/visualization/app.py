"""
Enhanced TE-KOA-C Dataset Dashboard.

This module provides a comprehensive dashboard for the TE-KOA-C dataset,
implementing all Phase I components including:
- Data loading and exploration
- Missing data analysis and imputation
- Variable screening
- Dimensionality reduction
- Data quality enhancement
- Treatment group analysis
"""

import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from tekoa import logger

from tekoa.visualization.data_manager import DataManager
from tekoa.visualization.ui_utils import apply_custom_css
from tekoa.visualization.pages import (
    HeaderComponent,
    SidebarComponent,
    OverviewPage,
    DataExplorerPage,
    DictionaryPage,
    MissingDataPage,
    ScreeningPage,
    DimensionalityPage,
    QualityPage,
    TreatmentGroupsPage,
    PipelinePage
)


from tekoa.visualization.phase2_pages import ClusteringPage, ValidationPage, CharacterizationPage


class Dashboard:
    """Enhanced dashboard for the TE-KOA-C clinical research dataset."""

    def __init__(self):
        """Initialize the TE-KOA dashboard component."""
        logger.info("Initializing Dashboard...")
        self.data_manager = DataManager()

    def run(self):
        """Render the TE-KOA-C dashboard."""
        # Set page config
        st.set_page_config(
            page_title="TE-KOA Clinical Research Dashboard",
            page_icon="🦵",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Apply custom CSS
        apply_custom_css()

        # Sidebar navigation (handles upload UI and triggers reruns for loading)
        SidebarComponent.render(self.data_manager)

        # Load data if not already loaded AND an uploaded file is available in session state
        # This block loads data based on st.session_state.uploaded_file set by SidebarComponent
        if self.data_manager.data is None or self.data_manager.dictionary is None:
            uploaded_file_in_session = st.session_state.get('uploaded_file')

            if uploaded_file_in_session is not None:
                with st.spinner(f"Loading dataset from {uploaded_file_in_session.name}..."):
                    # Explicitly pass the uploaded file to DataManager's load_data method.
                    # This ensures DataManager uses this specific file.
                    if not self.data_manager.load_data(uploaded_file_obj=uploaded_file_in_session):
                        st.error(
                            f"Failed to load the uploaded dataset: {uploaded_file_in_session.name}. "
                            "Please ensure it's a valid Excel file with 'Sheet1' for data and 'dictionary' for the data dictionary."
                        )
                        # Render header even on failure, to show the 'X' status, then stop.
                        HeaderComponent.render(self.data_manager)
                        st.stop()
                    else:
                        st.success(f"Successfully loaded dataset from: {uploaded_file_in_session.name}")
            # else:
                # No data loaded and no file has been uploaded yet.
                # The sidebar component already provides the UI for uploading.
                # Header will be rendered next, reflecting no data.
                pass

        # Display header (data_manager.data should be up-to-date for this run)
        HeaderComponent.render(self.data_manager)

        # Get current page from session state
        current_page = st.session_state.get('current_page', 'Overview')

        # If data is still None (e.g., no file uploaded, or initial state and no default data loaded)
        if self.data_manager.data is None:
            # HeaderComponent has already rendered, showing 'Data Loaded: X'
            # Display info message and stop further page rendering for this run.
            st.info("Welcome! Please upload an Excel dataset using the sidebar to begin analysis.")
            return # Stop rendering pages if no data

        # Render current page
        if current_page == 'Overview':
            OverviewPage.render(self.data_manager)

        elif current_page == 'Data Explorer':
            DataExplorerPage.render(self.data_manager)

        elif current_page == 'Data Dictionary':
            DictionaryPage.render(self.data_manager)

        # Phase 1 pages -- Data Preparation & Dimensionality Reduction
        elif current_page == 'Missing Data & Imputation':
            MissingDataPage.render(self.data_manager)

        elif current_page == 'Variable Screening':
            ScreeningPage.render(self.data_manager)

        elif current_page == 'Dimensionality Reduction':
            DimensionalityPage.render(self.data_manager)

        elif current_page == 'Data Quality':
            QualityPage.render(self.data_manager)

        elif current_page == 'Treatment Groups':
            TreatmentGroupsPage.render(self.data_manager)

        # Phase 2 pages -- Phenotype Discovery
        elif current_page == 'Clustering Analysis':
            ClusteringPage.render(self.data_manager)

        elif current_page == 'Cluster Validation':
            ValidationPage.render(self.data_manager)

        elif current_page == 'Phenotype Characterization':
            CharacterizationPage.render(self.data_manager)

        # Pipeline & Export
        elif current_page == 'Pipeline & Export':
            PipelinePage.render(self.data_manager)

        else:
            st.warning(f"Unknown page: {current_page}")


def main():
    """Main entry point for the TE-KOA dashboard."""
    dashboard = Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
