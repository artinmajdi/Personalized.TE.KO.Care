# te_koa/visualization/refactored_app/page_renderers.py
import streamlit as st
import pandas as pd
import numpy as np
from te_koa.visualization.app_refactored_gemini_components.plotter import Plotter
from te_koa.visualization.app_refactored_gemini_components.data_manager import DataManager
from te_koa.visualization.app_refactored_gemini_components.model_manager import ModelManager
from te_koa.visualization.app_refactored_gemini_components.ui_helpers import UIHelpers
from te_koa.configurations import params
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import shap

class AboutPage:
    """Renders the 'About' page."""
    def render(self):
        st.title("🔬 About the Personalized TE KOA Care Dashboard")
        st.markdown("---")

        st.header("Purpose")
        st.markdown(
            """
            This interactive dashboard provides insights into the factors influencing outcomes for individuals
            with Knee Osteoarthritis (KOA) undergoing Therapeutic Exercise (TE). It leverages machine learning
            models trained on clinical data to:

            - Explore relationships within the dataset (Exploratory Data Analysis).
            - Identify key predictors of TE outcomes (Feature Importance).
            - Evaluate the performance of different predictive models.
            - Offer personalized outcome predictions based on individual patient characteristics.
            """
        )

        st.header("Dataset")
        st.markdown(
            f"""
            The analysis is based on the `te_koa_R01_only_RCT_data.xlsx` dataset.
            - **Target Variable:** `{params.TARGET_VARIABLE}` ({params.FEATURE_DEFINITIONS.get(params.TARGET_VARIABLE, {}).get('description', 'Outcome of TE')}).
              - `{params.TARGET_CATEGORY_MAP.get(0, "Class 0")}`
              - `{params.TARGET_CATEGORY_MAP.get(1, "Class 1")}`
            - For detailed descriptions of each variable, please refer to the data dictionary or the EDA section.
            """
        )
        # You can add more details about data source, ethics, etc.

        st.header("Methodology")
        st.markdown(
            """
            1.  **Data Preprocessing:** Includes handling missing values, encoding categorical features, and scaling numerical features.
            2.  **Exploratory Data Analysis (EDA):** Visualizing distributions, correlations, and relationships.
            3.  **Model Training:** Several machine learning models (e.g., LightGBM, Random Forest, MLP, Stacking Classifier) were trained to predict the TE outcome.
            4.  **Model Evaluation:** Models were evaluated using metrics like F1-score, ROC AUC, Precision, and Recall on a held-out test set.
            5.  **Feature Importance:** Techniques like SHAP (SHapley Additive exPlanations) and model-specific importances are used to understand predictor influence.
            6.  **Personalized Prediction:** Users can input hypothetical patient data to receive a prediction and an explanation of the prediction.
            """
        )

        st.header("How to Use")
        st.markdown(
            """
            - Use the **sidebar** to navigate between different sections of the dashboard.
            - **Exploratory Data Analysis (EDA):** View summary statistics and visualizations of the dataset.
            - **Feature Importance:** See which factors are most influential in predicting outcomes.
            - **Predictive Modelling:** Compare the performance of different ML models.
            - **Personalized Prediction:** Enter patient details to get a tailored outcome prediction and explanation.
            """
        )

        st.header("Disclaimer")
        st.warning(
            """
            This tool is for informational and research purposes only. It is **not a substitute for professional medical advice, diagnosis, or treatment.**
            Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
            The predictions made by this tool are based on patterns in the data and are not guaranteed to be accurate for every individual.
            """
        )
        st.markdown("---")
        st.markdown("<p style='text-align: center;'>Developed by Artin Majdi</p>", unsafe_allow_html=True)


class EDAPage:
    """Renders the 'Exploratory Data Analysis' page."""
    def __init__(self, data_manager: DataManager, plotter: Plotter):
        self.data_manager = data_manager
        self.plotter = plotter
        self.data = self.data_manager.get_data()
        self.data_dictionary = self.data_manager.get_data_dictionary()

    def render(self):
        st.title("📊 Exploratory Data Analysis (EDA)")
        st.markdown("---")

        if self.data.empty:
            st.error("Dataset could not be loaded. EDA cannot be performed.")
            return

        st.header("Dataset Overview")
        st.markdown(f"**Number of Patients (Rows):** `{self.data.shape[0]}`")
        st.markdown(f"**Number of Variables (Columns):** `{self.data.shape[1]}`")

        st.subheader("First 5 Rows of the Dataset:")
        st.dataframe(self.data.head())

        st.subheader("Dataset Summary Statistics:")
        st.dataframe(self.data.describe(include='all').T)

        st.subheader("Data Dictionary / Variable Descriptions:")
        if not self.data_dictionary.empty:
            # Display a searchable and sortable view of the dictionary
            st.dataframe(self.data_dictionary, height=300)
        else:
            st.info("Data dictionary is not available.")

        st.markdown("---")
        st.header("Visualizations")

        # Target Variable Distribution
        st.subheader("Target Variable Distribution")
        target_var = self.data_manager.get_target_variable()
        fig_target = self.plotter.plot_target_distribution(self.data, target_var)
        st.plotly_chart(fig_target, use_container_width=True)

        # Demographic Distributions
        st.subheader("Demographic Distributions")
        col1, col2 = st.columns(2)
        with col1:
            fig_age = self.plotter.plot_age_distribution(self.data)
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            fig_gender = self.plotter.plot_gender_distribution(self.data) # Assumes 'GENDER' or 'SEX' column
            st.plotly_chart(fig_gender, use_container_width=True)

        fig_bmi = self.plotter.plot_bmi_distribution(self.data) # Assumes 'BMI' column
        st.plotly_chart(fig_bmi, use_container_width=True)


        # Correlation Matrix for Numerical Features
        st.subheader("Correlation Matrix")
        numerical_features = self.data_manager.get_numerical_features()
        if numerical_features:
            fig_corr = self.plotter.plot_correlation_matrix(self.data, numerical_features)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No numerical features identified for correlation matrix.")

        # Distribution of selected features
        st.subheader("Distribution of Selected Features")
        all_features = self.data_manager.get_all_features_list()
        if all_features:
            selected_feature = st.selectbox(
                "Select a feature to view its distribution:",
                options=all_features,
                index=0, # Default to the first feature
                format_func=lambda x: self.plotter._get_feature_label(x)
            )
            if selected_feature:
                if selected_feature in self.data_manager.get_numerical_features():
                    fig_dist = px.histogram(self.data, x=selected_feature,
                                            title=f"Distribution of {self.plotter._get_feature_label(selected_feature)}",
                                            labels={selected_feature: self.plotter._get_feature_label(selected_feature)})
                    st.plotly_chart(fig_dist, use_container_width=True)
                elif selected_feature in self.data_manager.get_categorical_features():
                    counts = self.data[selected_feature].value_counts().reset_index()
                    counts.columns = [selected_feature, 'count']
                    fig_dist = px.bar(counts, x=selected_feature, y='count',
                                      title=f"Distribution of {self.plotter._get_feature_label(selected_feature)}",
                                      labels={selected_feature: self.plotter._get_feature_label(selected_feature)})
                    st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No features available for distribution plotting.")


class FeatureImportancePage:
    """Renders the 'Feature Importance' page."""
    def __init__(self, data_manager: DataManager, model_manager: ModelManager, plotter: Plotter):
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.plotter = plotter
        self.available_models = self.model_manager.get_available_models()

    def render(self):
        st.title("🎯 Feature Importance Analysis")
        st.markdown("---")

        if not self.available_models:
            st.error("No models loaded. Feature importance cannot be displayed.")
            return

        model_options = {key: self.model_manager.get_model_display_name(key) for key in self.available_models.keys()}
        selected_model_key = st.selectbox(
            "Select a Model to View Feature Importance:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )

        if not selected_model_key:
            return

        st.header(f"Feature Importance for: {model_options[selected_model_key]}")

        # Method 1: Model-specific feature importance (e.g., Gini importance for trees)
        st.subheader("Model-Specific Feature Importance")
        model_importance = self.model_manager.get_feature_importance(selected_model_key)

        if model_importance is not None and not model_importance.empty:
            # Convert Series to DataFrame for plotter
            importance_df = model_importance.reset_index()
            importance_df.columns = ['feature', 'importance']
            fig_model_imp = self.plotter.plot_feature_importance(importance_df, top_n=20)
            st.plotly_chart(fig_model_imp, use_container_width=True)
        else:
            st.info(f"Model-specific feature importance is not available or could not be computed for {model_options[selected_model_key]}.")

        # Method 2: SHAP Feature Importance (Summary Plot)
        st.subheader("SHAP Value Summary (Global Importance)")
        shap_explainer = self.model_manager.shap_explainers.get(selected_model_key)

        if shap_explainer is not None:
            # We need some data to compute SHAP values for a summary plot
            # Usually, this is done on a test set or a representative sample of the training set
            # For simplicity, let's try to use a sample from the full dataset if available
            X, y = self.data_manager.get_X_y()

            if X.empty:
                st.warning("Data not available to generate SHAP summary plot.")
                return

            # SHAP can be slow on large datasets, so sample if necessary
            sample_size = min(100, len(X)) # Adjust sample size as needed
            if len(X) > sample_size:
                X_sample = X.sample(sample_size, random_state=params.RANDOM_STATE)
            else:
                X_sample = X

            # Preprocess the sample data
            if self.model_manager.preprocessor:
                try:
                    X_sample_processed = self.model_manager.preprocessor.transform(X_sample)

                    # Convert to DataFrame with correct feature names if preprocessor outputs numpy
                    feature_names_after_transform = self.model_manager.feature_names
                    if isinstance(X_sample_processed, np.ndarray):
                        if X_sample_processed.shape[1] == len(feature_names_after_transform):
                             X_sample_processed_df = pd.DataFrame(X_sample_processed, columns=feature_names_after_transform, index=X_sample.index)
                        else:
                            st.warning(f"SHAP Summary: Mismatch between transformed feature count ({X_sample_processed.shape[1]}) and expected feature names ({len(feature_names_after_transform)}). Plot may be mislabeled.")
                            X_sample_processed_df = pd.DataFrame(X_sample_processed, index=X_sample.index) # Generic names
                    elif isinstance(X_sample_processed, pd.DataFrame):
                        X_sample_processed_df = X_sample_processed
                    else:
                        st.error("Processed data for SHAP is not in an expected format (NumPy array or DataFrame).")
                        return

                    # Calculate SHAP values for the sample
                    # For TreeExplainer, shap_values directly on processed data
                    # For KernelExplainer, it might need the model's predict_proba function

                    # Using the pre-loaded explainer's logic from ModelManager
                    # The ModelManager's get_shap_values handles the explainer type
                    shap_values_sample = self.model_manager.get_shap_values(selected_model_key, X_sample) # Pass original X_sample

                    if shap_values_sample is not None:
                        # The plotter's SHAP summary function expects SHAP values and the *processed* data
                        # Ensure X_sample_processed_df is aligned with what the SHAP explainer saw.
                        # If get_shap_values returned an Explanation object, it might contain its own data and feature names.

                        # If shap_values_sample is an Explanation object, it might contain its own data.
                        # We need to ensure the `features` argument to summary_plot is consistent.
                        # The `plot_shap_summary_plot` in Plotter should handle this.
                        fig_shap_summary = self.plotter.plot_shap_summary_plot(
                            shap_values_sample,
                            X_sample_processed_df, # This is the data corresponding to shap_values_sample
                            plot_type="bar" # or "dot", "violin"
                        )
                        if fig_shap_summary:
                            st.pyplot(fig_shap_summary) # SHAP plots are often Matplotlib
                        else:
                            st.info("Could not generate SHAP summary plot.")
                    else:
                        st.info(f"SHAP values could not be computed for {model_options[selected_model_key]} on the sample data.")

                except Exception as e:
                    st.error(f"Error generating SHAP summary plot: {e}")
                    import traceback
                    st.error(traceback.format_exc())
            else:
                st.warning("Preprocessor not available. Cannot generate SHAP summary plot requiring processed data.")
        else:
            st.info(f"SHAP explainer not available for {model_options[selected_model_key]}. Cannot display SHAP-based importance.")

        st.markdown("---")
        st.markdown(
            """
            **Note on Feature Importance:**
            - **Model-Specific Importance:** Often based on how much a feature contributes to reducing impurity (e.g., Gini importance in Random Forests) or the magnitude of coefficients (in linear models).
            - **SHAP Values (SHapley Additive exPlanations):** A game theory approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions. SHAP values show the average impact of each feature on the model's prediction magnitude.
            """
        )


class PredictiveModellingPage:
    """Renders the 'Predictive Modelling' page."""
    def __init__(self, data_manager: DataManager, model_manager: ModelManager, plotter: Plotter):
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.plotter = plotter
        self.available_models = self.model_manager.get_available_models()

        # Load and split data for evaluation (cached)
        self.X_test, self.y_test, self.X_train, self.y_train = self._get_test_data()

    @st.cache_data(ttl=3600) # Cache for 1 hour
    def _get_test_data(_self): # _self for st.cache_data
        X, y = _self.data_manager.get_X_y()
        if X.empty or y.empty:
            st.warning("Data not available for model evaluation.")
            return pd.DataFrame(), pd.Series(dtype='float64'), pd.DataFrame(), pd.Series(dtype='float64')

        # Split data if not already split (this is for consistent evaluation)
        # In a real scenario, you'd use pre-defined train/test splits from your ML pipeline
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=params.TEST_SIZE, random_state=params.RANDOM_STATE, stratify=y
            )
            return X_test, y_test, X_train, y_train # Return train too if needed later
        except Exception as e:
            st.error(f"Error splitting data: {e}. Using full dataset for evaluation (not recommended).")
            return X, y, X, y


    def render(self):
        st.title("⚙️ Predictive Modelling Performance")
        st.markdown("---")

        if not self.available_models:
            st.error("No models loaded. Model performance cannot be displayed.")
            return

        if self.X_test.empty or self.y_test.empty:
            st.error("Test data not available. Cannot evaluate models.")
            return

        st.header("Model Performance Metrics on Test Set")
        st.markdown(
            f"""
            The following metrics are calculated on a held-out test set
            (test size: {params.TEST_SIZE*100}%, random state: {params.RANDOM_STATE}).
            Target Variable: `{params.TARGET_VARIABLE}` ({params.TARGET_CATEGORY_MAP[1]} vs {params.TARGET_CATEGORY_MAP[0]})
            """
        )

        performance_data = {}
        roc_curves_data = {}
        pr_curves_data = {}
        all_metrics_dfs = []

        for model_key, model_instance in self.available_models.items():
            model_display_name = self.model_manager.get_model_display_name(model_key)
            metrics, _, y_pred, y_proba = self.model_manager.get_model_performance_metrics(
                model_key, self.X_test, self.y_test
            )
            if metrics and y_proba is not None:
                performance_data[model_display_name] = metrics

                # For table display
                metrics_df = pd.DataFrame([metrics], index=[model_display_name])
                all_metrics_dfs.append(metrics_df)

                # ROC Curve data
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                roc_auc_val = auc(fpr, tpr)
                roc_curves_data[model_display_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc_val}

                # Precision-Recall Curve data
                precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
                avg_precision = average_precision_score(self.y_test, y_proba)
                pr_curves_data[model_display_name] = {'precision': precision, 'recall': recall, 'auc': avg_precision}


        if performance_data:
            # Combined Metrics Table
            st.subheader("Performance Metrics Overview")
            combined_metrics_df = pd.concat(all_metrics_dfs)
            st.dataframe(combined_metrics_df.style.format("{:.3f}"))

            # Bar chart comparison
            fig_perf_comp = self.plotter.plot_model_performance_comparison(performance_data)
            st.plotly_chart(fig_perf_comp, use_container_width=True)
        else:
            st.info("No performance metrics could be calculated for the loaded models.")

        # ROC and PR Curves
        if roc_curves_data or pr_curves_data:
            st.subheader("Model Evaluation Curves")
            col1, col2 = st.columns(2)
            with col1:
                if roc_curves_data:
                    fig_roc = self.plotter.plot_roc_auc_curves(roc_curves_data)
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info("ROC curve data not available.")
            with col2:
                if pr_curves_data:
                    fig_pr = self.plotter.plot_precision_recall_curves(pr_curves_data)
                    st.plotly_chart(fig_pr, use_container_width=True)
                else:
                    st.info("Precision-Recall curve data not available.")

        # Confusion Matrix for a selected model
        st.subheader("Confusion Matrix")
        model_options = {key: self.model_manager.get_model_display_name(key) for key in self.available_models.keys()}
        selected_cm_model_key = st.selectbox(
            "Select Model for Confusion Matrix:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            key="cm_model_select"
        )

        if selected_cm_model_key:
            _, cm, _, _ = self.model_manager.get_model_performance_metrics(
                selected_cm_model_key, self.X_test, self.y_test
            )
            if cm is not None:
                class_names = [params.TARGET_CATEGORY_MAP.get(0, "Class 0"), params.TARGET_CATEGORY_MAP.get(1, "Class 1")]
                fig_cm = self.plotter.plot_confusion_matrix(cm, class_names,
                                                           title=f"Confusion Matrix for {model_options[selected_cm_model_key]}")
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.warning(f"Could not generate confusion matrix for {model_options[selected_cm_model_key]}.")


class PersonalizedPredictionPage:
    """Renders the 'Personalized Prediction' page."""
    def __init__(self, data_manager: DataManager, model_manager: ModelManager, plotter: Plotter):
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.plotter = plotter
        self.available_models = self.model_manager.get_available_models()
        self.feature_definitions = self.data_manager.get_feature_definitions()
        self.all_features_list = self.data_manager.get_all_features_list()
        self.numerical_features = self.data_manager.get_numerical_features()
        self.categorical_features = self.data_manager.get_categorical_features()
        self.X_train_sample, _ = self.data_manager.get_X_y() # Used for input defaults/ranges


    def _get_input_fields(self):
        """Creates input fields for patient data."""
        st.subheader("Enter Patient Information:")

        # Use session state to store input values to persist them
        if 'patient_input_data' not in st.session_state:
            st.session_state.patient_input_data = {}

        input_data = {}

        # Intelligent defaults or ranges based on training data
        # For categorical, use unique values from training data
        # For numerical, use min/max/mean from training data

        # Layout in columns for better readability
        num_cols = 2 # Or 3, depending on number of features
        cols = st.columns(num_cols)
        col_idx = 0

        for feature in self.all_features_list:
            current_column = cols[col_idx % num_cols]
            label = self.feature_definitions.get(feature, {}).get("description", feature)
            unit = self.feature_definitions.get(feature, {}).get("unit", "")
            full_label = f"{label} ({unit})" if unit else label

            # Get current value from session state or default
            current_value = st.session_state.patient_input_data.get(feature)

            if feature in self.categorical_features:
                unique_values = sorted(list(self.X_train_sample[feature].dropna().unique()))
                # If current_value is not in unique_values (e.g. first run, or data changed), set a default
                default_cat_index = 0
                if current_value in unique_values:
                    default_cat_index = unique_values.index(current_value)
                elif unique_values: # Set to first option if current_value invalid
                    current_value = unique_values[0]

                if unique_values:
                    input_data[feature] = current_column.selectbox(
                        full_label, options=unique_values,
                        index=default_cat_index,
                        key=f"input_{feature}" # Unique key for each widget
                    )
                else:
                    input_data[feature] = current_column.text_input(full_label, value="", key=f"input_{feature}")
                    current_column.warning(f"No unique values found for {feature} in sample data. Please enter manually.")

            elif feature in self.numerical_features:
                min_val = float(self.X_train_sample[feature].min())
                max_val = float(self.X_train_sample[feature].max())
                mean_val = float(self.X_train_sample[feature].mean())

                # Set default for numerical input
                default_num_value = mean_val
                if current_value is not None:
                    try:
                        default_num_value = float(current_value)
                        # Clamp to min/max if out of bounds from previous entry
                        default_num_value = max(min(default_num_value, max_val), min_val)
                    except ValueError:
                        default_num_value = mean_val

                input_data[feature] = current_column.number_input(
                    full_label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_num_value,
                    step= (max_val - min_val) / 100 if max_val > min_val else 0.01, # Dynamic step
                    format="%.2f" if pd.api.types.is_float_dtype(self.X_train_sample[feature]) else "%d",
                    key=f"input_{feature}"
                )
            else: # Fallback for unknown types (should not happen if features correctly categorized)
                input_data[feature] = current_column.text_input(full_label, value=str(current_value) if current_value is not None else "", key=f"input_{feature}")

            col_idx += 1

        # Update session state with new inputs when button is pressed or automatically
        st.session_state.patient_input_data = input_data.copy()
        return pd.DataFrame([input_data])


    def render(self):
        st.title(" Personalized Prediction & Explanation")
        st.markdown("---")

        if not self.available_models:
            st.error("No models loaded. Personalized prediction is unavailable.")
            return

        if not self.all_features_list:
            st.error("Feature list not available. Cannot create input form.")
            return

        # Model Selection
        model_options = {key: self.model_manager.get_model_display_name(key) for key in self.available_models.keys()}
        selected_model_key = st.selectbox(
            "Select a Predictive Model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            key="prediction_model_select"
        )

        # Patient Data Input Fields
        patient_input_df = self._get_input_fields()

        # Prediction Button
        if st.button("Get Prediction", type="primary", key="get_prediction_button"):
            if patient_input_df.isnull().values.any():
                st.error("Please fill in all patient information fields.")
            else:
                # Display entered info for confirmation
                UIHelpers.display_patient_info(patient_input_df.iloc[0], self.feature_definitions)

                # Make Prediction
                prediction, probability = self.model_manager.predict(selected_model_key, patient_input_df)

                if prediction is not None and probability is not None:
                    UIHelpers.display_prediction_results(
                        prediction[0], # predict returns array
                        probability[0],# predict_proba returns array
                        model_options[selected_model_key]
                    )

                    # SHAP Explanations
                    st.subheader("Prediction Explanation (SHAP Values)")
                    shap_explainer_obj = self.model_manager.shap_explainers.get(selected_model_key)

                    if shap_explainer_obj:
                        # Get SHAP values for the single instance
                        # The model_manager.get_shap_values should handle preprocessing internally
                        shap_values_instance_explanation = self.model_manager.get_shap_values(
                            selected_model_key,
                            patient_input_df # Pass the original, unprocessed input DataFrame
                        )

                        if shap_values_instance_explanation is not None:
                            # The shap_values_instance_explanation should be a SHAP Explanation object
                            # or raw values that the plotter can handle.
                            # It's computed for the *processed* version of patient_input_df.

                            # We need the *processed* version of the input for some SHAP plots.
                            # ModelManager's preprocessor should be used.
                            if self.model_manager.preprocessor:
                                try:
                                    X_instance_processed_array = self.model_manager.preprocessor.transform(patient_input_df)

                                    # Convert to DataFrame with correct feature names
                                    feature_names_after_transform = self.model_manager.feature_names
                                    X_instance_processed_df = pd.DataFrame(X_instance_processed_array, columns=feature_names_after_transform)

                                    # Ensure the SHAP explanation object (if it is one) is for a single instance
                                    # SHAP explainers often return values for all instances passed.
                                    # If shap_values_instance_explanation is from explainer(processed_data),
                                    # and processed_data had one row, then shap_values_instance_explanation[0] gives the explanation for that row.

                                    single_instance_shap_for_plot = shap_values_instance_explanation
                                    if hasattr(shap_values_instance_explanation, 'values') and shap_values_instance_explanation.values.shape[0] > 1:
                                        # This means get_shap_values might have returned for multiple instances, even if we passed one.
                                        # Or it's (num_samples, num_features, num_classes) - handle this.
                                        # For plotting a single instance, we need the explanation for that instance.
                                        # If input was one row, then `explanation_object[0]` should be correct.
                                        # Assuming get_shap_values returns Explanation for the single row passed
                                        # or the raw SHAP values for that single row.
                                        # If Explanation object: explanation_obj[0] to get the Explanation for the first (only) sample.
                                        try:
                                            single_instance_shap_for_plot = shap_values_instance_explanation[0]
                                        except IndexError:
                                            st.warning("Could not isolate single instance from SHAP Explanation object. Plot might be incorrect.")
                                            pass # Use as is, plotter should handle if it's just values

                                    # 1. SHAP Force Plot (JS for interactivity, or Matplotlib)
                                    st.markdown("##### Contribution of each feature (Force Plot):")
                                    # The plotter needs the explainer (for base_value) and the SHAP values for the instance
                                    force_plot_js = self.plotter.plot_shap_individual_force_plot(
                                        explainer=shap_explainer_obj, # Pass the actual explainer object
                                        shap_values_instance=single_instance_shap_for_plot, # SHAP values/Explanation for the instance
                                        X_instance_processed_df=X_instance_processed_df, # Processed features for the instance
                                        matplotlib=False
                                    )
                                    if force_plot_js:
                                        shap.initjs() # Required for JS plots
                                        st.components.v1.html(force_plot_js.html(), height=200)
                                    else:
                                        st.info("Could not generate SHAP force plot.")

                                    # 2. SHAP Waterfall Plot
                                    st.markdown("##### Detailed feature impact (Waterfall Plot):")
                                    # Waterfall plot typically needs an Explanation object for a single instance.
                                    # Ensure single_instance_shap_for_plot is a suitable Explanation object.
                                    # If single_instance_shap_for_plot are raw numpy values, this might fail.
                                    # The plotter.plot_shap_waterfall_plot expects an Explanation object.

                                    # We might need to construct an Explanation object here if `get_shap_values`
                                    # returned raw values.
                                    # For now, assume `single_instance_shap_for_plot` is usable or is an Explanation object.
                                    # The plotter method should be robust.

                                    waterfall_fig = self.plotter.plot_shap_waterfall_plot(single_instance_shap_for_plot)
                                    if waterfall_fig:
                                        st.pyplot(waterfall_fig)
                                    else:
                                        st.info("Could not generate SHAP waterfall plot. Requires a SHAP Explanation object for the instance.")

                                except Exception as e_shap_plot:
                                    st.error(f"Error generating SHAP plots for instance: {e_shap_plot}")
                                    import traceback
                                    st.error(traceback.format_exc())
                            else:
                                st.warning("Preprocessor not available, cannot generate SHAP instance plots that require processed data.")
                        else:
                            st.info(f"SHAP values could not be computed for this prediction with {model_options[selected_model_key]}. Explanation unavailable.")
                    else:
                        st.info(f"SHAP explainer not available for {model_options[selected_model_key]}. Prediction explanation unavailable.")
                else:
                    st.error(f"Could not make a prediction with {model_options[selected_model_key]}. Please check model and preprocessor logs.")
