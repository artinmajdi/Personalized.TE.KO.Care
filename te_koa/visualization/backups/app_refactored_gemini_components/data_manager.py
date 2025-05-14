# te_koa/visualization/refactored_app/data_manager.py
import pandas as pd
import streamlit as st
from te_koa.configurations import params
from te_koa.io.data_loader import DataLoader

class DataManager:
    """
    Handles loading, preprocessing, and providing access to data.
    """
    def __init__(self):
        """
        Initializes the DataManager and loads the necessary data.
        """
        self.data_loader = DataLoader()
        self._load_data()

    @st.cache_data(ttl=3600) # Cache data for 1 hour
    def _load_data(_self): # Note: st.cache_data methods use _self or self as first arg
        """
        Loads the main dataset and data dictionary.
        The underscore for 'self' is a convention for Streamlit caching.
        """
        try:
            raw_data = _self.data_loader.load_data()
            data_dictionary = _self.data_loader.load_dictionary()

            # Basic preprocessing if needed (example, can be expanded)
            processed_data = raw_data.copy()
            # Example: Convert columns to numeric if they are not, based on dictionary or known types
            # This part needs to be adapted from your original app's preprocessing logic
            # For now, we assume data is mostly clean as per original structure.

            # Identify categorical and numerical features based on params or dictionary
            _self.all_features = [col for col in processed_data.columns if col != params.TARGET_VARIABLE]

            # Simple inference of categorical/numerical - can be improved using data_dictionary
            _self.numerical_features = processed_data[_self.all_features].select_dtypes(include=['number']).columns.tolist()
            _self.categorical_features = processed_data[_self.all_features].select_dtypes(exclude=['number']).columns.tolist()

            return processed_data, data_dictionary
        except FileNotFoundError as e:
            st.error(f"Error loading data: {e}. Please check file paths in configurations.")
            return pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            st.error(f"An unexpected error occurred while loading data: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def get_data(self):
        """Returns the processed dataset."""
        data, _ = self._load_data()
        return data

    def get_data_dictionary(self):
        """Returns the data dictionary."""
        _, dictionary = self._load_data()
        return dictionary

    def get_feature_definitions(self):
        """Returns feature definitions from params."""
        return params.FEATURE_DEFINITIONS

    def get_target_variable(self):
        """Returns the name of the target variable."""
        return params.TARGET_VARIABLE

    def get_all_features_list(self):
        """Returns the list of all feature names."""
        data, _ = self._load_data()
        if data.empty:
            return []
        return [col for col in data.columns if col != params.TARGET_VARIABLE]

    def get_numerical_features(self):
        """Returns list of numerical features."""
        self._load_data() # Ensure features are identified
        return self.numerical_features

    def get_categorical_features(self):
        """Returns list of categorical features."""
        self._load_data() # Ensure features are identified
        return self.categorical_features

    def get_X_y(self):
        """
        Splits the data into features (X) and target (y).
        """
        data = self.get_data()
        if data.empty or params.TARGET_VARIABLE not in data.columns:
            return pd.DataFrame(), pd.Series(dtype='float64')

        X = data.drop(columns=[params.TARGET_VARIABLE])
        y = data[params.TARGET_VARIABLE]
        return X, y

    def get_patient_input_template(self):
        """
        Creates a template DataFrame for patient input based on feature types.
        """
        X, _ = self.get_X_y()
        if X.empty:
            return pd.DataFrame()

        # Create a template with one row of NaNs or default values
        # This helps in creating input fields later
        input_template = pd.DataFrame(columns=X.columns)
        # For simplicity, we'll use a dictionary for default values or types
        # This should be more robust in a real scenario, possibly using the data dictionary

        # Example: Initialize with appropriate types or placeholders
        # For now, an empty DataFrame with correct columns is a start
        return input_template
