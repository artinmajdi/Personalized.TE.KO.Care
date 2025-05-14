import logging
import streamlit as st
from te_koa.visualization.app_refactored_gemini_components import DataManager, ModelManager, Plotter, UIHelpers
from te_koa.visualization.app_refactored_gemini_components.page_renderers import (
    AboutPage,
    EDAPage,
    FeatureImportancePage,
    PredictiveModellingPage,
    PersonalizedPredictionPage
)

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AppOrchestrator:
    """
    Main application class that orchestrates the different components
    of the Streamlit dashboard.
    """
    def __init__(self):
        """
        Initializes all managers and page renderer instances.
        """
        # Initialize managers
        self.data_manager = DataManager()
        self.model_manager = ModelManager(data_manager=self.data_manager) # ModelManager might need data_manager for feature names
        self.plotter = Plotter(data_manager=self.data_manager) # Plotter needs data_manager for feature definitions

        # Initialize page renderers
        self.pages = {
            "About": AboutPage(),
            "Exploratory Data Analysis (EDA)": EDAPage(self.data_manager, self.plotter),
            "Feature Importance": FeatureImportancePage(self.data_manager, self.model_manager, self.plotter),
            "Predictive Modelling": PredictiveModellingPage(self.data_manager, self.model_manager, self.plotter),
            "Personalized Prediction": PersonalizedPredictionPage(self.data_manager, self.model_manager, self.plotter)
        }

    def run(self):
        """
        Runs the Streamlit application: renders the sidebar and the selected page.
        """
        st.set_page_config(
            page_title            = "Personalized TE KOA Care",
            page_icon             = "🔬",
            layout                = "wide",
            initial_sidebar_state = "expanded",
        )

        selected_page_name = UIHelpers.render_sidebar()

        # Render the selected page
        if selected_page_name in self.pages:
            # Before rendering any page that uses models or data, ensure they are loaded
            # (or attempt to load them). Managers handle this internally with caching.
            # For example, accessing self.model_manager.models will trigger loading if not already done.
            # Similarly for self.data_manager.get_data().

            # Log which page is being rendered (optional)
            logger.info(f"Rendering page: {selected_page_name}")

            page_renderer = self.pages[selected_page_name]
            try:
                page_renderer.render()
            except Exception as e:
                st.error(f"An error occurred while rendering the page '{selected_page_name}': {e}")
                st.exception(e) # Shows the full traceback in the app for debugging
        else:
            st.error("Page not found. Please select a valid page from the sidebar.")


if __name__ == "__main__":
    # Run the application
    orchestrator = AppOrchestrator()
    orchestrator.run()
