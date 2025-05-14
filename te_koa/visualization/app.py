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

import logging
import streamlit as st
from typing import Optional

from te_koa.visualization.app_refactored_claude_components.data_manager import DataManager
from te_koa.visualization.app_refactored_claude_components.ui_utils import apply_custom_css
from te_koa.visualization.app_refactored_claude_components.pages import (
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Dashboard:
    """Enhanced dashboard for the TE-KOA-C clinical research dataset."""

    def __init__(self):
        """Initialize the TE-KOA dashboard component."""
        self.data_manager = DataManager()

    def render(self):
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

        # Display header
        HeaderComponent.render(self.data_manager)

        # Sidebar navigation
        SidebarComponent.render(self.data_manager)

        # Load data if not already loaded AND an uploaded file is available in session state
        if self.data_manager.data is None or self.data_manager.dictionary is None:
            uploaded_file_in_session = st.session_state.get('uploaded_file')

            if uploaded_file_in_session is not None:
                with st.spinner(f"Loading dataset from {uploaded_file_in_session.name}..."):
                    # Explicitly pass the uploaded file to DataManager's load_data method.
                    # This ensures DataManager uses this specific file.
                    if not self.data_manager.load_data():
                        st.error(
                            f"Failed to load the uploaded dataset: {uploaded_file_in_session.name}. "
                            "Please ensure it's a valid Excel file with 'Sheet1' for data and 'dictionary' for the data dictionary."
                        )
                        # Consider stopping or allowing user to upload a new file.
                        # For now, st.stop() prevents further rendering errors if pages expect data.
                        st.stop()
                    else:
                        st.success(f"Successfully loaded dataset from: {uploaded_file_in_session.name}")
            # else:
                # No data loaded and no file has been uploaded yet.
                # The sidebar component (SidebarComponent.render above) already provides
                # the UI for uploading. Individual pages should handle the 'no data' state gracefully
                # or display prompts if they are active and data is missing.
                # Example: if st.session_state.get('current_page', 'Overview') == 'Overview':
                # st.info("Welcome! Please upload an Excel dataset using the sidebar to begin analysis.")
                pass # Allow app to render; sidebar will show upload options.

        # Get current page from session state
        current_page = st.session_state.get('current_page', 'Overview')

        if self.data_manager.data is None:
            st.info("Welcome! Please upload an Excel dataset using the sidebar to begin analysis.")
            return

        # Render current page
        if current_page == 'Overview':
            OverviewPage.render(self.data_manager)

        elif current_page == 'Data Explorer':
            DataExplorerPage.render(self.data_manager)

        elif current_page == 'Data Dictionary':
            DictionaryPage.render(self.data_manager)

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

        elif current_page == 'Pipeline & Export':
            PipelinePage.render(self.data_manager)

        else:
            st.warning(f"Unknown page: {current_page}")


def main():
    """Main entry point for the TE-KOA dashboard."""
    dashboard = Dashboard()
    dashboard.render()


if __name__ == "__main__":
    main()
