# te_koa/visualization/refactored_app/ui_helpers.py
import streamlit as st

class UIHelpers:
    """
    Contains helper functions for creating common UI elements like the sidebar.
    """
    @staticmethod
    def render_sidebar():
        """
        Renders the sidebar navigation and returns the selected page.
        """
        st.sidebar.title("🩺 Personalized TE KOA Care")
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            """
            Welcome to the Personalized Therapeutic Exercise for Knee Osteoarthritis (TE KOA) Care dashboard.
            This tool helps in understanding and predicting outcomes for KOA patients.
            """
        )
        st.sidebar.markdown("---")

        # Initialize session state for selected page if it doesn't exist
        if 'selected_page' not in st.session_state:
            st.session_state.selected_page = "About"

        # Navigation options
        navigation_options = [
            "About",
            "Exploratory Data Analysis (EDA)",
            "Feature Importance",
            "Predictive Modelling",
            "Personalized Prediction"
        ]

        # Use radio buttons for navigation, updating session_state on change
        def update_page():
            st.session_state.selected_page = st.session_state._sidebar_selection

        st.sidebar.radio(
            "Navigate Sections:",
            options=navigation_options,
            key="_sidebar_selection", # Use a temporary key for the widget
            on_change=update_page,
            index=navigation_options.index(st.session_state.selected_page) # Set current selection
        )

        st.sidebar.markdown("---")
        st.sidebar.info(
            "This dashboard is based on research data and predictive models. "
            "Consult with healthcare professionals for medical advice."
        )
        st.sidebar.markdown(
            """
            <div style="text-align: center;">
                <small>Powered by Streamlit & PyData</small>
            </div>
            """, unsafe_allow_html=True
        )
        return st.session_state.selected_page


    @staticmethod
    def display_patient_info(patient_data_series, feature_definitions):
        """
        Displays patient information in a structured way.
        Args:
            patient_data_series (pd.Series): Series containing patient data.
            feature_definitions (dict): Dictionary with feature descriptions.
        """
        st.subheader("Patient Information:")

        # Create two columns for better layout
        col1, col2 = st.columns(2)

        # Alternate adding items to columns
        data_items = list(patient_data_series.items())
        for i, (feature, value) in enumerate(data_items):
            description = feature_definitions.get(feature, {}).get("description", feature)
            unit = feature_definitions.get(feature, {}).get("unit", "")
            display_value = f"{value} {unit}".strip()

            if i % 2 == 0:
                with col1:
                    st.markdown(f"**{description}:** `{display_value}`")
            else:
                with col2:
                    st.markdown(f"**{description}:** `{display_value}`")
        st.markdown("---")


    @staticmethod
    def display_prediction_results(prediction, probability, model_name):
        """
        Displays the prediction results.
        Args:
            prediction (int): The predicted class (0 or 1).
            probability (float): The probability of the positive class.
            model_name (str): The name of the model used for prediction.
        """
        st.subheader(f"Prediction Results ({model_name}):")

        # Define outcome based on your target variable's meaning
        # Assuming 1 is "Poor Outcome" and 0 is "Good Outcome" (adjust if different)
        outcome_map = params.TARGET_CATEGORY_MAP # {0: "Good Outcome", 1: "Poor Outcome"}

        predicted_outcome_label = outcome_map.get(prediction, f"Class {prediction}")

        if prediction == 1: # Assuming 1 is the "at-risk" or "poor" outcome
            st.error(f"**Predicted Outcome:** {predicted_outcome_label}")
        else:
            st.success(f"**Predicted Outcome:** {predicted_outcome_label}")

        st.metric(
            label=f"Probability of '{outcome_map.get(1, 'Poor Outcome')}'", # Probability of the positive class
            value=f"{probability:.2%}"
        )

        # Confidence level (can be simple or more complex)
        confidence = "High" if probability > 0.8 or probability < 0.2 else "Medium" if probability > 0.6 or probability < 0.4 else "Low"
        st.write(f"**Confidence in Prediction:** {confidence}")
        st.markdown("---")

