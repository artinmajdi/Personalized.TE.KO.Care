# te_koa/visualization/refactored_app/model_manager.py
import logging
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from te_koa.configurations import params
import shap # Ensure shap is installed

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Handles loading machine learning models, preprocessors, SHAP explainers,
    and making predictions.
    """
    MODEL_MAPPING = {
        "LGBM": "LightGBM",
        "RF": "Random Forest",
        "MLP": "MLP Classifier",
        "Stacking": "Stacking Classifier"
    }

    def __init__(self, data_manager):
        """
        Initializes the ModelManager.
        Args:
            data_manager (DataManager): An instance of DataManager to get feature info.
        """
        self.data_manager    = data_manager
        self.models          = self._load_models()
        self.preprocessor    = self._load_preprocessor()
        self.shap_explainers = self._load_shap_explainers()
        self.feature_names   = self._get_transformed_feature_names()


    @st.cache_resource(ttl=3600) # Cache resource for 1 hour
    def _load_models(_self):
        """Loads all trained models specified in params.MODEL_PATHS."""
        models = {}
        for model_key, model_path in params.MODEL_PATHS.items():
            try:
                models[model_key] = joblib.load(model_path)
                # st.success(f"Successfully loaded model: {_self.MODEL_MAPPING.get(model_key, model_key)}")
            except FileNotFoundError:
                st.error(f"Model file not found for {model_key} at {model_path}. This model will be unavailable.")
                models[model_key] = None
            except Exception as e:
                st.error(f"Error loading model {model_key}: {e}")
                models[model_key] = None
        return models

    @st.cache_resource(ttl=3600)
    def _load_preprocessor(_self):
        """Loads the preprocessor."""
        try:
            preprocessor = joblib.load(params.PREPROCESSOR_PATH)
            # st.success("Successfully loaded preprocessor.")
            return preprocessor
        except FileNotFoundError:
            st.error(f"Preprocessor file not found at {params.PREPROCESSOR_PATH}. Predictions may fail.")
            return None
        except Exception as e:
            st.error(f"Error loading preprocessor: {e}")
            return None

    @st.cache_resource(ttl=3600)
    def _load_shap_explainers(_self):
        """Loads SHAP explainers for available models."""
        shap_explainers = {}
        # Ensure models are loaded before trying to load SHAP explainers
        if not _self.models:
            _self.models = _self._load_models()

        for model_key, explainer_path in params.SHAP_EXPLAINER_PATHS.items():
            if _self.models.get(model_key): # Only load explainer if model exists
                try:
                    shap_explainers[model_key] = joblib.load(explainer_path)
                    # st.success(f"Successfully loaded SHAP explainer for: {_self.MODEL_MAPPING.get(model_key, model_key)}")
                except FileNotFoundError:
                    st.warning(f"SHAP explainer not found for {model_key} at {explainer_path}. SHAP plots for this model will be unavailable.")
                    shap_explainers[model_key] = None
                except Exception as e:
                    st.error(f"Error loading SHAP explainer for {model_key}: {e}")
                    shap_explainers[model_key] = None
            else:
                shap_explainers[model_key] = None # Model itself is missing
        return shap_explainers

    def _get_transformed_feature_names(self):
        """
        Gets feature names after preprocessing.
        Relies on the preprocessor being fitted and having a `get_feature_names_out` method
        or similar, or by inspecting transformers.
        """
        if not self.preprocessor:
            # Fallback to original features if preprocessor is not loaded
            # This might not be accurate for SHAP plots if features are transformed.
            st.warning("Preprocessor not loaded. Using original feature names. SHAP plots might not be accurate.")
            return self.data_manager.get_all_features_list()

        try:
            # Attempt to get feature names from the preprocessor
            # This is highly dependent on the structure of your preprocessor
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                # For ColumnTransformer with named transformers
                return self.preprocessor.get_feature_names_out()
            elif hasattr(self.preprocessor, 'transformers_'):
                # More complex scenario: iterate through transformers
                # This is a placeholder and needs to be adapted to your specific preprocessor
                feature_names = []
                for name, trans, columns in self.preprocessor.transformers_:
                    if trans == 'drop' or trans == 'passthrough':
                        continue
                    if hasattr(trans, 'get_feature_names_out'):
                        feature_names.extend(trans.get_feature_names_out(columns))
                    else: # Fallback for simple transformers
                        feature_names.extend(columns)
                return feature_names
            else:
                # If preprocessor doesn't have a method, try transforming dummy data
                # This is a common pattern but might be slow or fail.
                X_sample, _ = self.data_manager.get_X_y()
                if not X_sample.empty:
                    X_transformed_sample = self.preprocessor.transform(X_sample.head(1))
                    if isinstance(X_transformed_sample, np.ndarray):
                        return [f"feature_{i}" for i in range(X_transformed_sample.shape[1])]
                    elif isinstance(X_transformed_sample, pd.DataFrame):
                        return X_transformed_sample.columns.tolist()
        except Exception as e:
            st.warning(f"Could not determine transformed feature names: {e}. Using original feature names.")

        # Fallback if names can't be determined
        return self.data_manager.get_all_features_list()


    def get_available_models(self):
        """Returns a dictionary of available (loaded) models."""
        return {key: model for key, model in self.models.items() if model is not None}

    def get_model_display_name(self, model_key):
        """Returns the display name for a model key."""
        return self.MODEL_MAPPING.get(model_key, model_key)

    def predict(self, model_key, input_data_df):
        """
        Makes predictions using the specified model.
        Args:
            model_key (str): Key of the model to use (e.g., "LGBM").
            input_data_df (pd.DataFrame): DataFrame with input features.
        Returns:
            tuple: (predictions, probabilities) or (None, None) if error.
        """
        model = self.models.get(model_key)
        if model is None or self.preprocessor is None:
            st.error(f"Model '{model_key}' or preprocessor not available for prediction.")
            return None, None

        try:
            # Ensure input_data_df has columns in the same order as training data
            X_train_cols = self.data_manager.get_all_features_list()
            input_data_df_reordered = input_data_df[X_train_cols]

            processed_input = self.preprocessor.transform(input_data_df_reordered)
            predictions = model.predict(processed_input)
            probabilities = model.predict_proba(processed_input)[:, 1] # Probability of positive class
            return predictions, probabilities
        except Exception as e:
            st.error(f"Error during prediction with {model_key}: {e}")
            return None, None

    def get_shap_values(self, model_key, input_data_df):
        """
        Calculates SHAP values for the given input data and model.
        Args:
            model_key (str): Key of the model.
            input_data_df (pd.DataFrame): DataFrame with input features.
        Returns:
            shap.Explanation or None: SHAP values or None if error/unavailable.
        """
        explainer = self.shap_explainers.get(model_key)
        model = self.models.get(model_key)

        if explainer is None or model is None or self.preprocessor is None:
            # st.warning(f"SHAP explainer or model for '{model_key}' not available.")
            return None

        try:
            # Ensure input_data_df has columns in the same order as training data
            X_train_cols = self.data_manager.get_all_features_list()
            input_data_df_reordered = input_data_df[X_train_cols]

            processed_input = self.preprocessor.transform(input_data_df_reordered)

            # SHAP explainers might expect a DataFrame or NumPy array
            # If preprocessor outputs NumPy, convert to DataFrame with feature names
            if isinstance(processed_input, np.ndarray) and self.feature_names:
                 if processed_input.shape[1] == len(self.feature_names):
                    processed_input_df = pd.DataFrame(processed_input, columns=self.feature_names)
                 else: # Mismatch in feature count
                    st.warning(f"Feature name count ({len(self.feature_names)}) mismatch with processed data columns ({processed_input.shape[1]}). SHAP might be inaccurate.")
                    # Attempt to use generic feature names if mismatch
                    processed_input_df = pd.DataFrame(processed_input, columns=[f"feature_{i}" for i in range(processed_input.shape[1])])

            elif isinstance(processed_input, pd.DataFrame):
                 processed_input_df = processed_input
            else: # Fallback for other types, may need adjustment
                 processed_input_df = pd.DataFrame(processed_input)


            if isinstance(explainer, shap.TreeExplainer) or isinstance(explainer, shap.KernelExplainer) or isinstance(explainer, shap.DeepExplainer):
                 shap_values = explainer(processed_input_df) # New SHAP API
                 # For older SHAP versions, it might be explainer.shap_values(processed_input_df)
                 # If shap_values is a list (e.g. for multi-class), take the one for the positive class
                 if isinstance(shap_values, list) and len(shap_values) > 1:
                     # Assuming binary classification, index 1 is for the positive class
                     # This might need adjustment based on your specific SHAP explainer output
                     return shap_values[1]
                 return shap_values

            else: # Fallback for other explainer types or if direct call fails
                 shap_values_raw = explainer.shap_values(processed_input_df)
                 # Construct an Explanation object if not already
                 # This depends on the output of explainer.shap_values()
                 if isinstance(shap_values_raw, np.ndarray): # Common for TreeExplainer.shap_values
                     return shap.Explanation(
                         values=shap_values_raw,
                         base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else None, # Add base_values
                         data=processed_input_df,
                         feature_names=processed_input_df.columns.tolist()
                     )
                 elif isinstance(shap_values_raw, list): # Common for Kernel/Deep for multi-output
                      # Assuming binary classification, positive class is often index 1
                     return shap.Explanation(
                         values=shap_values_raw[1], # Adjust index if necessary
                         base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1 else explainer.expected_value,
                         data=processed_input_df,
                         feature_names=processed_input_df.columns.tolist()
                     )
                 return shap_values_raw # If already an Explanation object or other format

        except Exception as e:
            st.error(f"Error calculating SHAP values for {model_key}: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def get_model_performance_metrics(self, model_key, X_test, y_test):
        """
        Calculates and returns performance metrics for a given model on test data.
        Args:
            model_key (str): The key of the model.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels for the test set.
        Returns:
            dict: A dictionary of performance metrics, or None if error.
        """
        model = self.models.get(model_key)
        if model is None or self.preprocessor is None:
            return None

        try:
            processed_X_test = self.preprocessor.transform(X_test)
            y_pred = model.predict(processed_X_test)
            y_proba = model.predict_proba(processed_X_test)[:, 1]

            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-score": f1_score(y_test, y_pred, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, y_proba)
            }
            cm = confusion_matrix(y_test, y_pred)
            return metrics, cm, y_pred, y_proba
        except Exception as e:
            st.error(f"Error calculating performance metrics for {model_key}: {e}")
            return None, None, None, None

    def get_feature_importance(self, model_key):
        """
        Retrieves feature importance from a model.
        Note: This is a simplified version. Actual implementation depends on model type.
        For tree-based models, it's usually `feature_importances_`.
        For linear models, it's `coef_`.
        For SHAP-based importance, it would involve averaging SHAP values.
        """
        model = self.models.get(model_key)
        if model is None:
            return None

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Ensure feature names are available and match the importances length
            # These should be the names *after* preprocessing
            feature_names = self.feature_names
            if len(importances) == len(feature_names):
                return pd.Series(importances, index=feature_names).sort_values(ascending=False)
            else:
                st.warning(f"Mismatch between feature importance length ({len(importances)}) and feature name count ({len(feature_names)}) for model {model_key}.")
                # Fallback to generic feature names if mismatch
                return pd.Series(importances, index=[f"feature_{i}" for i in range(len(importances))]).sort_values(ascending=False)

        elif hasattr(model, 'coef_'): # For linear models
            importances = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            feature_names = self.feature_names
            if len(importances) == len(feature_names):
                return pd.Series(importances, index=feature_names).sort_values(ascending=False, key=abs) # Sort by absolute value for coefficients
            else:
                st.warning(f"Mismatch between coefficient length ({len(importances)}) and feature name count ({len(feature_names)}) for model {model_key}.")
                return pd.Series(importances, index=[f"feature_{i}" for i in range(len(importances))]).sort_values(ascending=False, key=abs)

        # Fallback: if model is StackingClassifier, try getting from final_estimator
        elif hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'feature_importances_'):
            importances = model.final_estimator_.feature_importances_
            # Getting feature names for the StackingClassifier's final estimator can be tricky
            # It depends on whether passthrough=True for some base estimators.
            # For now, we assume the number of features matches self.feature_names
            # This might need adjustment based on your StackingClassifier setup.
            feature_names = self.feature_names
            if len(importances) == len(feature_names):
                 return pd.Series(importances, index=feature_names).sort_values(ascending=False)
            else: # If not, try to get input features to final_estimator
                try:
                    # This is a guess; actual number of features depends on base estimators' outputs
                    num_meta_features = importances.shape[0]
                    meta_feature_names = [f"meta_feature_{i}" for i in range(num_meta_features)]
                    return pd.Series(importances, index=meta_feature_names).sort_values(ascending=False)
                except Exception:
                    st.warning(f"Could not determine feature names for StackingClassifier's final estimator for model {model_key}.")
                    return pd.Series(importances).sort_values(ascending=False)


        st.warning(f"Feature importance not directly available for model type: {type(model).__name__} for model {model_key}. Consider using SHAP values for importance.")
        return None

