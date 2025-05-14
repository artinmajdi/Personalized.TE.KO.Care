# te_koa/visualization/refactored_app/plotter.py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
import shap
from sklearn.metrics import roc_curve, auc, precision_recall_curve

class Plotter:
    """
    Handles creation of all Plotly visualizations for the application.
    """
    def __init__(self, data_manager):
        """
        Initializes the Plotter.
        Args:
            data_manager (DataManager): An instance of DataManager.
        """
        self.data_manager = data_manager
        self.feature_definitions = data_manager.get_feature_definitions()

    def _get_feature_label(self, feature_name):
        """Gets a display-friendly label for a feature."""
        return self.feature_definitions.get(feature_name, {}).get("description", feature_name)

    def plot_age_distribution(self, df):
        """Plots the age distribution."""
        if 'AGE' not in df.columns:
            st.warning("AGE column not found in data for plotting distribution.")
            return go.Figure()
        fig = px.histogram(df, x="AGE", nbins=20, title="Age Distribution of Patients",
                           labels={"AGE": self._get_feature_label("AGE")})
        fig.update_layout(bargap=0.1)
        return fig

    def plot_bmi_distribution(self, df):
        """Plots the BMI distribution."""
        if 'BMI' not in df.columns:
            st.warning("BMI column not found in data for plotting distribution.")
            return go.Figure()
        fig = px.histogram(df, x="BMI", nbins=30, title="BMI Distribution of Patients",
                           labels={"BMI": self._get_feature_label("BMI")})
        fig.update_layout(bargap=0.1)
        return fig

    def plot_gender_distribution(self, df):
        """Plots the gender distribution."""
        gender_col = None
        # Try to find a gender column, common names: 'GENDER', 'SEX', 'Gender', 'Sex'
        for col_name in ['GENDER', 'SEX', 'Gender', 'Sex', 'gender', 'sex']:
            if col_name in df.columns:
                gender_col = col_name
                break

        if gender_col is None:
            st.warning("Gender column (e.g., 'GENDER', 'SEX') not found for plotting distribution.")
            return go.Figure()

        gender_counts = df[gender_col].value_counts().reset_index()
        gender_counts.columns = [gender_col, 'count']
        fig = px.pie(gender_counts, names=gender_col, values='count', title="Gender Distribution",
                     labels={gender_col: self._get_feature_label(gender_col)})
        return fig

    def plot_target_distribution(self, df, target_variable):
        """Plots the distribution of the target variable."""
        if target_variable not in df.columns:
            st.warning(f"Target variable '{target_variable}' not found for plotting distribution.")
            return go.Figure()

        target_counts = df[target_variable].value_counts().reset_index()
        target_counts.columns = [target_variable, 'count']
        fig = px.pie(target_counts, names=target_variable, values='count',
                     title=f"Distribution of Target: {self._get_feature_label(target_variable)}",
                     labels={target_variable: self._get_feature_label(target_variable)})
        return fig

    def plot_correlation_matrix(self, df, numerical_features):
        """Plots the correlation matrix for numerical features."""
        if not numerical_features:
            st.warning("No numerical features found to plot correlation matrix.")
            return go.Figure()

        corr_matrix = df[numerical_features].corr()
        fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                        title="Correlation Matrix of Numerical Features",
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(height=600)
        return fig

    def plot_feature_importance(self, importance_df, top_n=20):
        """
        Plots feature importances.
        Args:
            importance_df (pd.DataFrame): DataFrame with 'feature' and 'importance' columns.
            top_n (int): Number of top features to display.
        """
        if importance_df is None or importance_df.empty:
            st.info("Feature importance data is not available for this model or could not be computed.")
            return go.Figure()

        # Ensure columns are named correctly
        if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
             # Try to infer if it's a Series converted to DataFrame
            if importance_df.shape[1] == 1 and isinstance(importance_df.index, pd.Index): # Likely a Series
                importance_df = importance_df.reset_index()
                importance_df.columns = ['feature', 'importance']
            else:
                st.error("Importance DataFrame must have 'feature' and 'importance' columns.")
                return go.Figure()

        top_features = importance_df.nlargest(top_n, 'importance').sort_values('importance', ascending=True)

        fig = px.bar(top_features, x='importance', y='feature', orientation='h',
                     title=f"Top {min(top_n, len(top_features))} Feature Importances",
                     labels={'importance': 'Importance Score', 'feature': 'Feature'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=max(400, len(top_features)*25))
        return fig

    def plot_model_performance_comparison(self, performance_metrics_dict):
        """
        Plots a comparison of performance metrics (e.g., F1-score, ROC AUC) across models.
        Args:
            performance_metrics_dict (dict):
                {'ModelName1': {'Metric1': value, 'Metric2': value},
                 'ModelName2': {'Metric1': value, 'Metric2': value}}
        """
        if not performance_metrics_dict:
            st.info("No model performance data to compare.")
            return go.Figure()

        df_list = []
        for model_name, metrics in performance_metrics_dict.items():
            if metrics: # Ensure metrics are not None
                for metric_name, value in metrics.items():
                    df_list.append({'Model': model_name, 'Metric': metric_name, 'Score': value})

        if not df_list:
            st.info("Performance metrics are empty or invalid.")
            return go.Figure()

        perf_df = pd.DataFrame(df_list)

        # Select key metrics for bar chart comparison
        key_metrics = ['F1-score', 'ROC AUC', 'Precision', 'Recall', 'Accuracy']
        plot_df = perf_df[perf_df['Metric'].isin(key_metrics)]

        if plot_df.empty:
            st.info(f"No data for key metrics: {', '.join(key_metrics)}.")
            return go.Figure()

        fig = px.bar(plot_df, x='Metric', y='Score', color='Model', barmode='group',
                     title='Model Performance Comparison',
                     labels={'Score': 'Score', 'Metric': 'Performance Metric'})
        fig.update_layout(yaxis_range=[0,1]) # Scores typically between 0 and 1
        return fig

    def plot_roc_auc_curves(self, roc_data_dict, title="ROC AUC Curves"):
        """
        Plots ROC AUC curves for multiple models.
        Args:
            roc_data_dict (dict):
                {'ModelName1': {'fpr': array, 'tpr': array, 'auc': float}, ...}
        """
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        for model_name, data in roc_data_dict.items():
            if data and 'fpr' in data and 'tpr' in data and 'auc' in data:
                name = f"{model_name} (AUC={data['auc']:.3f})"
                fig.add_trace(go.Scatter(x=data['fpr'], y=data['tpr'], name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=600,
            title=title,
            legend_title_text='Models'
        )
        return fig

    def plot_precision_recall_curves(self, pr_data_dict, title="Precision-Recall Curves"):
        """
        Plots Precision-Recall curves for multiple models.
        Args:
            pr_data_dict (dict):
                {'ModelName1': {'precision': array, 'recall': array, 'auc': float_or_none}, ...}
                 auc for PR curve is Average Precision (AP)
        """
        fig = go.Figure()

        for model_name, data in pr_data_dict.items():
            if data and 'precision' in data and 'recall' in data:
                name = f"{model_name}"
                if 'auc' in data and data['auc'] is not None: # Average Precision
                     name = f"{model_name} (AP={data['auc']:.3f})"
                fig.add_trace(go.Scatter(x=data['recall'], y=data['precision'], name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=700, height=600,
            title=title,
            legend_title_text='Models',
            yaxis_range=[0,1.05],
            xaxis_range=[0,1.05]
        )
        return fig

    def plot_confusion_matrix(self, cm, class_names, title="Confusion Matrix"):
        """
        Plots a confusion matrix.
        Args:
            cm (np.array): Confusion matrix (e.g., from sklearn.metrics.confusion_matrix).
            class_names (list): List of class names (e.g., ['Negative', 'Positive']).
            title (str): Title of the plot.
        """
        fig = px.imshow(cm,
                        labels=dict(x="Predicted Label", y="True Label", color="Count"),
                        x=class_names,
                        y=class_names,
                        text_auto=True,
                        color_continuous_scale='Blues') # Or another scale like 'Viridis'
        fig.update_layout(title_text=title, title_x=0.5)
        return fig

    def plot_shap_summary_plot(self, shap_values, X_processed_df, plot_type="bar"):
        """
        Generates a SHAP summary plot.
        Args:
            shap_values (shap.Explanation or np.ndarray): SHAP values.
                                If np.ndarray, assumes it's for the positive class.
            X_processed_df (pd.DataFrame): The processed input data for which SHAP values were computed.
                                           Feature names should match those used by the SHAP explainer.
            plot_type (str): Type of SHAP summary plot ("bar", "dot", "violin", etc.).
        Returns:
            A Matplotlib figure object (SHAP plots directly use Matplotlib).
            Or None if error.
        """
        if shap_values is None:
            st.info("SHAP values are not available or could not be computed for this model.")
            return None

        try:
            # The SHAP library's plot functions often create their own figures.
            # We need to capture this figure to display it in Streamlit.
            # shap.summary_plot returns matplotlib.axes.Axes object, so we need to get current fig.
            import matplotlib.pyplot as plt

            # Create a new figure before calling shap.summary_plot
            # This helps in managing the figure object.
            fig, ax = plt.subplots()

            if isinstance(shap_values, np.ndarray): # If raw SHAP values array
                shap.summary_plot(shap_values, X_processed_df, plot_type=plot_type, show=False, feature_names=X_processed_df.columns, ax=ax)
            elif hasattr(shap_values, 'values') and hasattr(shap_values, 'data'): # If SHAP Explanation object
                # For summary_plot with Explanation objects, data is often implicitly used.
                # Ensure X_processed_df has the same columns as shap_values.feature_names if available
                # Or that shap_values.data is correctly aligned.
                # If shap_values.data is a numpy array, X_processed_df provides feature names.

                # If shap_values.values has more than 2 dimensions (e.g. multi-output models)
                # and we are interested in a specific output (e.g. class 1 for binary)
                vals_to_plot = shap_values.values
                if vals_to_plot.ndim > 2 and vals_to_plot.shape[-1] > 1: # e.g. (num_samples, num_features, num_classes)
                    vals_to_plot = vals_to_plot[..., params.SHAP_CLASS_INDEX_FOR_SUMMARY] # Assuming binary, class 1

                # If shap_values.data is None or not a DataFrame, use X_processed_df
                # The SHAP library can be inconsistent here.
                # We pass shap_values.values and features=X_processed_df
                # or shap_values directly if it's a well-formed Explanation object.

                # Check if X_processed_df columns match shap_values.feature_names
                # This is crucial for correct labeling.
                if hasattr(shap_values, 'feature_names') and list(X_processed_df.columns) != list(shap_values.feature_names):
                    st.warning("Mismatch between X_processed_df columns and shap_values.feature_names. Plot might be mislabeled.")
                    # Try to use shap_values.feature_names if available
                    # This assumes shap_values.values align with shap_values.feature_names
                    # And shap_values.data (if numpy) also aligns.
                    # It's safer if X_processed_df is already aligned.

                shap.summary_plot(vals_to_plot, features=X_processed_df, plot_type=plot_type, show=False, ax=ax)

            else:
                st.error("Unsupported SHAP values format for summary plot.")
                plt.close(fig) # Close the unused figure
                return None

            plt.tight_layout()
            return fig # Return the Matplotlib figure object
        except Exception as e:
            st.error(f"Error generating SHAP summary plot: {e}")
            import traceback
            st.error(traceback.format_exc())
            if 'fig' in locals() and fig is not None:
                plt.close(fig) # Ensure figure is closed on error
            return None


    def plot_shap_individual_force_plot(self, explainer, shap_values_instance, X_instance_processed_df, matplotlib=False):
        """
        Generates an individual SHAP force plot.
        Args:
            explainer (shap.Explainer): The SHAP explainer object (needed for expected_value).
            shap_values_instance (shap.Explanation or np.ndarray): SHAP values for a single instance.
                                                                If ndarray, it's the raw values.
            X_instance_processed_df (pd.DataFrame): The single processed instance data (1 row).
            matplotlib (bool): Whether to return a Matplotlib plot (True) or JS plot (False).
        Returns:
            A Matplotlib figure object or SHAP JS plot component.
        """
        if shap_values_instance is None or explainer is None:
            st.info("SHAP values or explainer not available for this instance.")
            return None

        try:
            # Expected value might be a single value or an array (for multi-output)
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                # Assuming binary classification, take the expected value for the positive class
                # This index might need to be params.SHAP_CLASS_INDEX_FOR_SUMMARY
                base_value = base_value[params.SHAP_CLASS_INDEX_FOR_SUMMARY] if len(base_value) > params.SHAP_CLASS_INDEX_FOR_SUMMARY else base_value[0]


            # If shap_values_instance is an Explanation object, it contains all necessary parts
            if hasattr(shap_values_instance, 'values') and hasattr(shap_values_instance, 'base_values') and hasattr(shap_values_instance, 'data'):
                # Ensure we are using the shap values for the correct class if multi-dimensional
                sv_instance_values = shap_values_instance.values
                if sv_instance_values.ndim > 1 and sv_instance_values.shape[-1] > 1: # e.g. (num_features, num_classes)
                    sv_instance_values = sv_instance_values[..., params.SHAP_CLASS_INDEX_FOR_SUMMARY]

                # Ensure base_values are scalar for the force plot
                bv_instance = shap_values_instance.base_values
                if isinstance(bv_instance, (list, np.ndarray)):
                     bv_instance = bv_instance[params.SHAP_CLASS_INDEX_FOR_SUMMARY] if len(bv_instance) > params.SHAP_CLASS_INDEX_FOR_SUMMARY else bv_instance[0]

                # Data for the instance
                data_instance = shap_values_instance.data
                if isinstance(data_instance, pd.DataFrame):
                    data_instance = data_instance.iloc[0] # Ensure it's a Series for display

                force_plot = shap.force_plot(
                    base_value=bv_instance,
                    shap_values=sv_instance_values, # Should be 1D array of SHAP values for the instance
                    features=data_instance, # Should be 1D array or Series of feature values
                    feature_names=X_instance_processed_df.columns.tolist(), # Ensure feature names are passed
                    matplotlib=matplotlib,
                    show=False # Important for capturing Matplotlib fig
                )

            elif isinstance(shap_values_instance, np.ndarray): # Raw SHAP values for the instance
                # Ensure shap_values_instance is 1D
                if shap_values_instance.ndim > 1:
                     # Assuming (1, num_features) or (num_features, num_classes)
                     if shap_values_instance.shape[0] == 1: # (1, num_features)
                         sv_plot = shap_values_instance.flatten()
                     elif shap_values_instance.ndim == 2 and shap_values_instance.shape[1] > 1: # (num_features, num_classes)
                         sv_plot = shap_values_instance[:, params.SHAP_CLASS_INDEX_FOR_SUMMARY]
                     else:
                         sv_plot = shap_values_instance.flatten() # Best guess
                else:
                    sv_plot = shap_values_instance

                force_plot = shap.force_plot(
                    base_value=base_value,
                    shap_values=sv_plot,
                    features=X_instance_processed_df.iloc[0], # Pass as Series
                    feature_names=X_instance_processed_df.columns.tolist(),
                    matplotlib=matplotlib,
                    show=False
                )
            else:
                st.error("Unsupported SHAP values format for individual force plot.")
                return None

            if matplotlib:
                import matplotlib.pyplot as plt
                plt.tight_layout() # Adjust layout
                return plt.gcf()  # Get current Matplotlib figure
            else:
                return force_plot # This will be the JS plot object
        except Exception as e:
            st.error(f"Error generating SHAP individual force plot: {e}")
            import traceback
            st.error(traceback.format_exc())
            if matplotlib and 'plt' in locals(): # Close figure if matplotlib was used and error occurred
                plt.close()
            return None

    def plot_shap_waterfall_plot(self, shap_values_instance):
        """
        Generates a SHAP waterfall plot for a single instance.
        Args:
            shap_values_instance (shap.Explanation): SHAP Explanation object for a single instance.
                                                    It should contain .values, .base_values, .data, .feature_names.
        Returns:
            A Matplotlib figure object or None if error.
        """
        if shap_values_instance is None or not (hasattr(shap_values_instance, 'values') and \
                                                hasattr(shap_values_instance, 'base_values') and \
                                                hasattr(shap_values_instance, 'data')):
            st.info("A complete SHAP Explanation object for the instance is required for waterfall plot.")
            return None

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

            # SHAP waterfall plot expects the shap_values for a single instance.
            # If shap_values_instance is an Explanation for multiple instances, select one.
            # Assuming shap_values_instance is already for a *single* instance here.
            # e.g., explainer(X_instance_processed_df)[0]

            # If the Explanation object's values are multi-dimensional (e.g. for multi-class)
            # select the relevant class.
            # Example: shap_values_instance.values could be (num_features, num_classes)
            # We need (num_features,) for the specific class.

            # Create a new Explanation object for the specific class if needed.
            # This is often handled by indexing the Explanation object like `shap_values_instance[:, class_index]`
            # but that might not always work as expected for waterfall.
            # Let's assume shap_values_instance is already correctly sliced for the class of interest.
            # Or, we construct a temporary one for the plot.

            current_sv = shap_values_instance
            if current_sv.values.ndim > 1 and current_sv.values.shape[-1] > 1:
                # Create a new Explanation object for the specific class
                class_idx = params.SHAP_CLASS_INDEX_FOR_SUMMARY

                # Ensure base_values and data are also correctly sliced/formatted if they were multi-output
                base_val_slice = current_sv.base_values
                if isinstance(base_val_slice, (np.ndarray, list)) and len(base_val_slice) > class_idx:
                    base_val_slice = base_val_slice[class_idx]

                # Data might not need slicing if it's just feature values
                # feature_names should be consistent.

                current_sv = shap.Explanation(
                    values=current_sv.values[..., class_idx], # Slice values for the class
                    base_values=base_val_slice,
                    data=current_sv.data, # Assumes data is (num_features,) or (1, num_features)
                    feature_names=current_sv.feature_names
                )

            shap.waterfall_plot(current_sv, show=False, ax=ax) # Pass the ax object
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error generating SHAP waterfall plot: {e}")
            import traceback
            st.error(traceback.format_exc())
            if 'fig' in locals() and fig is not None:
                plt.close(fig)
            return None

