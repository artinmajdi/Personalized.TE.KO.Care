"""
Data manager for the TE-KOA dashboard.

This module provides a central manager for loading, processing, and managing data
for the TE-KOA dashboard.
"""

import pandas as pd
import numpy as np
import streamlit as st
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Union

# Assuming these will be available or mocked from tekoa library
# from tekoa.clustering import find_optimal_clusters, perform_clustering, calculate_silhouette_scores
# from tekoa.validation import calculate_bootstrap_stability
# For now, using sklearn placeholders for some functionalities if tekoa.clustering is not ready
from sklearn.cluster import KMeans # Placeholder for tekoa.clustering.perform_clustering
from sklearn.metrics import silhouette_score, silhouette_samples # Placeholder for tekoa.clustering.calculate_silhouette_scores
import numpy as np # Ensure numpy is imported for array operations

from tekoa import logger
from tekoa.io import DataLoader
from tekoa.utils import (
    VariableScreener,
    DimensionalityReducer,
    DataQualityEnhancer,
)
from tekoa.configuration.params import DatasetNames



class DataManager:
    """Manager for TE-KOA data loading, processing, and state management."""

    def __init__(self):
        """Initialize the data manager."""
        self.data_loader: Optional[DataLoader] = None
        self.dataset_name = DatasetNames.TEKOA.value
        self.data = None
        self.dictionary = None
        self.imputed_data = None
        self.treatment_groups = None
        self.missing_data_report = None

        # Analysis components
        self.variable_screener = None
        self.dimensionality_reducer = None
        self.data_quality_enhancer = None

        # Initialize session state for data management
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables for data management."""
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}
        if 'phenotypes' not in st.session_state: # This might be used by phenotyping results
            st.session_state.phenotypes = None
        if 'phenotyping_results' not in st.session_state: # For storing results of perform_auto_phenotyping
            st.session_state.phenotyping_results = None
        if 'phenotype_names' not in st.session_state: # For user-defined phenotype names
            st.session_state.phenotype_names = {}
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Overview'

    def load_data(self, uploaded_file_obj: Optional[Any] = None) -> bool:
        """
        Load the TE-KOA dataset from a provided uploaded file object.

        Args:
            uploaded_file_obj: The uploaded file object (e.g., from Streamlit's file_uploader).
                                If None, loading will fail.

        Returns:
            bool: True if data loaded successfully, False otherwise.
        """
        self.data_loader = None # Reset data_loader at the start of an attempt

        if uploaded_file_obj is not None:
            file_name_for_log = getattr(uploaded_file_obj, 'name', 'uploaded_file') # Get name if available for logging
            logger.info(f"DataManager attempting to load data from provided uploaded file: {file_name_for_log}")
            # DataLoader is initialized with the uploaded file; data_dir will be None by default in DataLoader
            # ensuring it doesn't look for default paths.
            self.data_loader = DataLoader(uploaded_file=uploaded_file_obj)
        else:
            logger.warning("DataManager.load_data called without an 'uploaded_file_obj'. No data will be loaded.")
            return False # Cannot proceed if no file object is given

        if self.data_loader is None:
            # This case should ideally be caught by the 'else' above, but as a safeguard:
            logger.error("DataManager could not initialize DataLoader, though an uploaded_file_obj was expected.")
            return False

        try:
            # Load data using the initialized data loader
            self.data, self.dictionary = self.data_loader.load_data()
            self.missing_data_report = self.data_loader.get_missing_data_report()

            # Initialize data components if data loaded successfully
            if self.data is not None and self.dictionary is not None:
                st.session_state.processed_data = self.data.copy()
                st.session_state.data_loaded_time = pd.Timestamp.now()
                log_file_name = getattr(uploaded_file_obj, 'name', 'the_uploaded_file')
                logger.info(f"DataManager successfully loaded data: {len(self.data)} rows, {len(self.data.columns)} columns from {log_file_name}")

                # Check for required treatment column (optional, based on dataset specs)
                if 'tx.group' not in self.data.columns:
                    logger.warning("'tx.group' column not found in the loaded dataset.")
                return True
            else:
                log_file_name_fail = getattr(uploaded_file_obj, 'name', 'the_uploaded_file')
                logger.error(f"DataLoader failed to load data from {log_file_name_fail} (returned None for data/dictionary) within DataManager.")
                # Ensure data attributes are reset if loading fails partway
                self.data = None
                self.dictionary = None
                self.missing_data_report = None
                return False
        except Exception as e:
            log_file_name_exc = getattr(uploaded_file_obj, 'name', 'the_uploaded_file')
            logger.error(f"Exception occurred in DataManager.load_data while processing {log_file_name_exc}: {e}", exc_info=True)
            # Reset state on critical failure
            self.data = None
            self.dictionary = None
            self.missing_data_report = None
            # self.data_loader might be kept for debugging or reset: self.data_loader = None
            return False

    def get_data(self) -> pd.DataFrame:
        """Get the current processed data."""
        return st.session_state.processed_data if st.session_state.processed_data is not None else self.data

    def get_original_data(self) -> pd.DataFrame:
        """Get the original unprocessed data."""
        return self.data

    def get_data_dictionary(self) -> pd.DataFrame:
        """Get the data dictionary."""
        return self.dictionary

    def get_treatment_groups(self) -> Dict[str, pd.DataFrame]:
        """Get treatment groups from the data."""
        if self.treatment_groups is None:
            self.treatment_groups = self.data_loader.get_treatment_groups()
        return self.treatment_groups

    def get_missing_data_report(self) -> pd.DataFrame:
        """Get the missing data report."""
        if self.missing_data_report is None:
            self.missing_data_report = self.data_loader.get_missing_data_report()
        return self.missing_data_report

    def get_variable_categories(self) -> Dict[str, List[str]]:
        """Get categorized variables."""
        return self.data_loader.get_variable_categories() if self.data_loader else {}

    def get_variable_description(self, variable: str) -> str:
        """Get description for a variable."""
        return self.data_loader.get_variable_description(variable) if self.data_loader else None

    def initialize_variable_screener(self):
        """Initialize the variable screener if not already done."""
        if self.variable_screener is None and st.session_state.processed_data is not None:
            self.variable_screener = VariableScreener(st.session_state.processed_data)

    def initialize_dimensionality_reducer(self):
        """Initialize the dimensionality reducer if not already done."""
        if self.dimensionality_reducer is None and st.session_state.processed_data is not None:
            self.dimensionality_reducer = DimensionalityReducer(st.session_state.processed_data)

    def initialize_data_quality_enhancer(self):
        """Initialize the data quality enhancer if not already done."""
        if self.data_quality_enhancer is None and st.session_state.processed_data is not None:
            self.data_quality_enhancer = DataQualityEnhancer(st.session_state.processed_data)

    def impute_missing_values(self, method: str, knn_neighbors: int, cols_to_exclude: List[str]) -> pd.DataFrame:
        """
        Impute missing values in the dataset.

        Args:
            method: Imputation method ('mean', 'median', 'knn')
            knn_neighbors: Number of neighbors for KNN imputation
            cols_to_exclude: List of columns to exclude from imputation

        Returns:
            DataFrame with imputed values
        """
        # Impute missing values
        imputed_data = self.data_loader.impute_missing_values(
            method=method,
            knn_neighbors=knn_neighbors,
            cols_to_exclude=cols_to_exclude
        )

        # Store the imputed data
        self.imputed_data = imputed_data
        st.session_state.processed_data = imputed_data

        # Store imputation details in pipeline results
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}

        st.session_state.pipeline_results['imputation'] = {
            'method': method,
            'knn_neighbors': knn_neighbors,
            'cols_excluded': cols_to_exclude,
            'original_missing': self.data.isnull().sum().sum(),
            'remaining_missing': imputed_data.isnull().sum().sum()
        }

        return imputed_data

    def screen_variables(self, near_zero_threshold: float, collinearity_threshold: float, vif_threshold: float, force_include: List[str]) -> Dict[str, Any]:
        """
        Screen variables for near-zero variance, collinearity, and VIF.

        Args:
            near_zero_threshold: Threshold for near-zero variance
            collinearity_threshold: Threshold for collinearity
            vif_threshold: Threshold for VIF
            force_include: List of variables to force include

        Returns:
            Dictionary with screening results
        """
        self.initialize_variable_screener()

        # Get recommendations
        recommendations = self.variable_screener.recommend_variables(
            near_zero_threshold=near_zero_threshold,
            collinearity_threshold=collinearity_threshold,
            vif_threshold=vif_threshold,
            force_include=force_include
        )

        # Store in session state
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}

        st.session_state.pipeline_results['variable_screening'] = self.variable_screener.get_results()

        return recommendations

    def apply_variable_recommendations(self, recommended_vars: List[str]):
        """
        Apply variable recommendations to the dataset.

        Args:
            recommended_vars: List of recommended variables to keep
        """
        # Ensure all recommended variables exist in the dataset
        existing_vars = [var for var in recommended_vars if var in st.session_state.processed_data.columns]

        # Update processed data
        st.session_state.processed_data = st.session_state.processed_data[existing_vars]

        # Update variable screener to use new data
        self.variable_screener = None
        self.initialize_variable_screener()

    def perform_dimensionality_reduction(self, method: str, variables: List[str], n_components: int,
                                        standardize: bool = True) -> Dict[str, Any]:
        """
        Perform dimensionality reduction on the dataset.

        Args:
            method: Method for dimensionality reduction ('pca', 'famd')
            variables: List of variables to use
            n_components: Number of components to extract
            standardize: Whether to standardize data (for PCA)

        Returns:
            Dictionary with dimensionality reduction results
        """
        self.initialize_dimensionality_reducer()

        # Perform the reduction
        if method == 'pca':
            results = self.dimensionality_reducer.perform_pca(
                variables    = variables,
                n_components = n_components,
                standardize  = standardize
            )
        elif method == 'famd':
            results = self.dimensionality_reducer.perform_famd( n_components=n_components )
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")

        # Store in session state
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}

        # Safely handle numpy arrays for JSON serialization
        explained_variance = results.get('explained_variance_ratio', results.get('explained_variance', []))
        if isinstance(explained_variance, np.ndarray):
            explained_variance = explained_variance.tolist()

        cumulative_explained_variance = results.get('cumulative_explained_variance', [])
        if isinstance(cumulative_explained_variance, np.ndarray):
            cumulative_explained_variance = cumulative_explained_variance.tolist()

        st.session_state.pipeline_results['dimensionality_reduction'] = {
            'method': method,
            'n_components': n_components,
            'standardize': standardize if method == 'pca' else None,
            'variables': variables,
            'explained_variance': explained_variance,
            'cumulative_explained_variance': cumulative_explained_variance
        }

        return results

    def transform_data(self, method: str, n_components: int) -> pd.DataFrame:
        """
        Transform data using dimensionality reduction.

        Args:
            method: Method for dimensionality reduction ('pca', 'famd')
            n_components: Number of components to keep

        Returns:
            Transformed DataFrame
        """
        self.initialize_dimensionality_reducer()

        # Transform the data
        transformed_data = self.dimensionality_reducer.transform_data(
            method=method,
            n_components=n_components
        )

        # Store transformed data in session state
        st.session_state.processed_data = transformed_data

        # Update pipeline results
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}

        if 'dimensionality_reduction' not in st.session_state.pipeline_results:
            st.session_state.pipeline_results['dimensionality_reduction'] = {}

        st.session_state.pipeline_results['dimensionality_reduction']['transformed_data_shape'] = transformed_data.shape

        return transformed_data

    def detect_outliers(self, method: str, threshold: float, variables: List[str]) -> Dict[str, Any]:
        """
        Detect outliers in the dataset.

        Args:
            method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            variables: List of variables to check for outliers

        Returns:
            Dictionary with outlier detection results
        """
        self.initialize_data_quality_enhancer()

        # Detect outliers
        outliers = self.data_quality_enhancer.detect_outliers(
            method=method,
            threshold=threshold,
            variables=variables
        )

        # Store in pipeline results
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}

        if 'data_quality' not in st.session_state.pipeline_results:
            st.session_state.pipeline_results['data_quality'] = {}

        st.session_state.pipeline_results['data_quality']['outliers'] = {
            'method': method,
            'threshold': threshold,
            'variables': variables,
            'summary': outliers.get('summary', {})
        }

        return outliers

    def analyze_distributions(self, variables: List[str]) -> Dict[str, Any]:
        """
        Analyze distributions of variables.

        Args:
            variables: List of variables to analyze

        Returns:
            Dictionary with distribution analysis results
        """
        self.initialize_data_quality_enhancer()

        # Analyze distributions
        distributions = self.data_quality_enhancer.analyze_distributions(variables=variables)

        # Store in pipeline results
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}

        if 'data_quality' not in st.session_state.pipeline_results:
            st.session_state.pipeline_results['data_quality'] = {}

        st.session_state.pipeline_results['data_quality']['distributions'] = {
            'variables': variables,
            'summary': distributions.get('summary', {})
        }

        return distributions

    def recommend_transformations(self) -> Dict[str, Any]:
        """
        Recommend transformations for variables with skewed distributions.

        Returns:
            Dictionary with transformation recommendations
        """
        self.initialize_data_quality_enhancer()

        # Get recommendations
        transformations = self.data_quality_enhancer.recommend_transformations()

        # Store in pipeline results
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}

        if 'data_quality' not in st.session_state.pipeline_results:
            st.session_state.pipeline_results['data_quality'] = {}

        st.session_state.pipeline_results['data_quality']['transformations'] = {
            'summary': transformations.get('summary', {})
        }

        return transformations

    def apply_transformations(self, transformations_to_apply: Dict[str, str]) -> pd.DataFrame:
        """
        Apply transformations to variables.

        Args:
            transformations_to_apply: Dictionary mapping variable names to transformation types

        Returns:
            Transformed DataFrame
        """
        self.initialize_data_quality_enhancer()

        # Apply transformations
        transformed_data = self.data_quality_enhancer.apply_transformations(
            transformations=transformations_to_apply
        )

        # Update session state
        st.session_state.processed_data = transformed_data

        # Update pipeline results
        applied = self.data_quality_enhancer.results['transformations'].get('applied', {})

        if 'details' in applied:
            if 'pipeline_results' not in st.session_state:
                st.session_state.pipeline_results = {}

            if 'data_quality' not in st.session_state.pipeline_results:
                st.session_state.pipeline_results['data_quality'] = {}

            if 'transformations' not in st.session_state.pipeline_results['data_quality']:
                st.session_state.pipeline_results['data_quality']['transformations'] = {}

            st.session_state.pipeline_results['data_quality']['transformations']['applied'] = {
                'variables': list(applied['details'].keys()),
                'summary': applied.get('summary', {})
            }

        return transformed_data

    def standardize_variables(self, method: str, variables: List[str]) -> pd.DataFrame:
        """
        Standardize variables in the dataset.

        Args:
            method: Method for standardization ('zscore', 'robust', 'minmax')
            variables: List of variables to standardize

        Returns:
            Standardized DataFrame
        """
        self.initialize_data_quality_enhancer()

        # Standardize variables
        standardized_data = self.data_quality_enhancer.standardize_variables(
            variables=variables,
            method=method
        )

        # Store in pipeline results
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}

        if 'data_quality' not in st.session_state.pipeline_results:
            st.session_state.pipeline_results['data_quality'] = {}

        st.session_state.pipeline_results['data_quality']['standardization'] = {
            'method': method,
            'variables': variables
        }

        # Update session state
        st.session_state.processed_data = standardized_data

        return standardized_data

    def export_data(self, export_format: str, include_metadata: bool = True):
        """
        Export processed data and optionally metadata.

        Args:
            export_format: Format for export ('csv', 'excel')
            include_metadata: Whether to include pipeline metadata

        Returns:
            Tuple of (data, filename, format)
        """
        if export_format == "csv":
            # Export to CSV
            csv = st.session_state.processed_data.to_csv(index=False)
            filename = "te_koa_processed_data.csv"

            return csv, filename, "csv"

        elif export_format == "excel":
            # Export to Excel
            import io
            buffer = io.BytesIO()

            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.processed_data.to_excel(writer, sheet_name='Processed Data', index=False)

                # Add metadata if requested
                if include_metadata and 'pipeline_results' in st.session_state:
                    # Create metadata sheet
                    metadata = pd.DataFrame([
                        {'Step': 'Imputation', 'Details': str(st.session_state.pipeline_results.get('imputation', 'Not completed'))},
                        {'Step': 'Variable Screening', 'Details': str(st.session_state.pipeline_results.get('variable_screening', {}).get('summary', 'Not completed'))},
                        {'Step': 'Dimensionality Reduction', 'Details': str(st.session_state.pipeline_results.get('dimensionality_reduction', 'Not completed'))},
                        {'Step': 'Data Quality', 'Details': str(st.session_state.pipeline_results.get('data_quality', 'Not completed'))}
                    ])

                    metadata.to_excel(writer, sheet_name='Pipeline Metadata', index=False)

            filename = "te_koa_processed_data.xlsx"
            return buffer.getvalue(), filename, "excel"

    def export_metadata(self):
        """
        Export pipeline metadata as JSON.

        Returns:
            Tuple of (data, filename, format)
        """
        # Convert pipeline results to JSON
        def convert_for_json(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, (np.ndarray, list)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return float(obj) if np.issubdtype(type(obj), np.floating) else int(obj)
            else:
                return obj

        # Convert pipeline results
        json_results = json.dumps(convert_for_json(st.session_state.pipeline_results), indent=2)
        filename = "te_koa_pipeline_metadata.json"

        return json_results, filename, "json"

    def generate_report(self):
        """
        Generate a comprehensive report of the pipeline.

        Returns:
            Tuple of (report_markdown, filename, format)
        """
        pipeline_results = st.session_state.pipeline_results

        # Create a full report in Markdown format
        report = f"# TE-KOA Data Preparation Pipeline Report\n\n"
        report += f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

        # 1. Data Overview
        report += "## Data Overview\n\n"
        report += f"* Original dataset: {len(self.data)} participants, {len(self.data.columns)} variables\n"
        report += f"* Current dataset: {len(st.session_state.processed_data)} participants, {len(st.session_state.processed_data.columns)} variables\n"
        report += f"* Dimensionality reduction: {(1 - len(st.session_state.processed_data.columns) / len(self.data.columns)) * 100:.1f}%\n\n"

        # 2. Imputation
        if 'imputation' in pipeline_results:
            imputation = pipeline_results['imputation']

            report += "## Missing Data & Imputation\n\n"
            report += f"* Imputation method: {imputation.get('method', 'unknown')}\n"
            report += f"* KNN neighbors (if applicable): {imputation.get('knn_neighbors', 'N/A')}\n"
            report += f"* Original missing values: {imputation.get('original_missing', 'N/A')}\n"
            report += f"* Remaining missing values: {imputation.get('remaining_missing', 'N/A')}\n"
            report += f"* Excluded variables: {len(imputation.get('cols_excluded', []))}\n\n"

        # 3. Variable Screening
        if 'variable_screening' in pipeline_results:
            screening = pipeline_results['variable_screening']

            if 'summary' in screening:
                summary = screening['summary']

                report += "## Variable Screening\n\n"
                report += f"* Total variables analyzed: {summary.get('total_variables', 0)}\n"
                report += f"* Near-zero variance variables: {summary.get('near_zero_variables', 0)}\n"
                report += f"* Highly correlated pairs: {summary.get('highly_correlated_pairs', 0)}\n"
                report += f"* High VIF variables: {summary.get('high_vif_variables', 0)}\n"
                report += f"* Force-included variables: {summary.get('force_included_variables', 0)}\n"
                report += f"* Recommended variables: {summary.get('recommended_variables', 0)}\n\n"

        # 4. Dimensionality Reduction
        if 'dimensionality_reduction' in pipeline_results:
            dim_reduction = pipeline_results['dimensionality_reduction']

            report += "## Dimensionality Reduction\n\n"
            report += f"* Method: {dim_reduction.get('method', 'unknown')}\n"
            report += f"* Number of components: {dim_reduction.get('n_components', 0)}\n"
            report += f"* Variables analyzed: {len(dim_reduction.get('variables', []))}\n"

            if 'transformed_data_shape' in dim_reduction:
                report += f"* Transformed data shape: {dim_reduction.get('transformed_data_shape', 'N/A')}\n\n"

        # 5. Data Quality Enhancement
        if 'data_quality' in pipeline_results:
            data_quality = pipeline_results['data_quality']

            report += "## Data Quality Enhancement\n\n"

            if 'outliers' in data_quality:
                outliers = data_quality['outliers']

                report += "### Outlier Detection\n\n"
                report += f"* Method: {outliers.get('method', 'unknown')}\n"
                report += f"* Threshold: {outliers.get('threshold', 'N/A')}\n"
                report += f"* Variables analyzed: {len(outliers.get('variables', []))}\n"
                report += f"* Total outliers detected: {outliers.get('summary', {}).get('total_outliers_detected', 0)}\n\n"

            if 'distributions' in data_quality:
                distributions = data_quality['distributions']

                report += "### Distribution Analysis\n\n"
                report += f"* Variables analyzed: {distributions.get('summary', {}).get('variables_analyzed', 0)}\n"
                report += f"* Normal variables: {distributions.get('summary', {}).get('normal_variables', 0)}\n"
                report += f"* Skewed variables: {distributions.get('summary', {}).get('skewed_variables', 0)}\n"
                report += f"* Highly skewed variables: {distributions.get('summary', {}).get('highly_skewed_variables', 0)}\n\n"

            if 'transformations' in data_quality:
                transformations = data_quality['transformations']

                report += "### Variable Transformations\n\n"
                report += f"* Variables analyzed: {transformations.get('summary', {}).get('variables_analyzed', 0)}\n"
                report += f"* Variables needing transformation: {transformations.get('summary', {}).get('variables_needing_transformation', 0)}\n\n"

                if 'applied' in transformations:
                    applied = transformations['applied']

                    report += "### Applied Transformations\n\n"
                    report += f"* Variables transformed: {applied.get('summary', {}).get('variables_transformed', 0)}\n"

                    counts = applied.get('summary', {}).get('transformation_counts', {})
                    report += f"* Log transformations: {counts.get('log', 0)}\n"
                    report += f"* Square root transformations: {counts.get('sqrt', 0)}\n"
                    report += f"* Square transformations: {counts.get('square', 0)}\n"
                    report += f"* Yeo-Johnson transformations: {counts.get('yeo-johnson', 0)}\n\n"

            if 'standardization' in data_quality:
                standardization = data_quality['standardization']

                report += "### Variable Standardization\n\n"
                report += f"* Method: {standardization.get('method', 'unknown')}\n"
                report += f"* Variables standardized: {len(standardization.get('variables', []))}\n\n"

        # Add a summary table of processed data
        if not st.session_state.processed_data.empty:
            report += "## Processed Data Summary\n\n"

            # Convert DataFrame description to Markdown
            desc = st.session_state.processed_data.describe().to_markdown()
            report += desc + "\n\n"

        filename = "te_koa_pipeline_report.md"
        return report, filename, "markdown"

    # --- Phase-II: Auto-Phenotyping, Exploration, and Validation ---

    def perform_auto_phenotyping(self, processed_data: pd.DataFrame, clustering_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform auto-phenotyping using clustering on the processed data.

        Args:
            processed_data: DataFrame containing the data to be clustered.
            clustering_params: Optional dictionary of parameters for clustering.
                                Example: {'n_clusters': 3, 'method': 'kmeans', ...}

        Returns:
            Dictionary containing clustering results:
                - 'cluster_labels': Array of cluster labels for each sample.
                - 'n_clusters': Optimal number of clusters found.
                - 'silhouette_score': Average silhouette score for the clustering.
                - 'silhouette_values': Silhouette scores for each sample.
        """
        if processed_data is None or processed_data.empty:
            logger.warning("perform_auto_phenotyping: Processed data is not available.")
            return {}

        logger.info("Starting auto-phenotyping...")

        # Select only numeric columns for clustering
        numeric_data = processed_data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            logger.error("perform_auto_phenotyping: No numeric data available for clustering after filtering. Cannot proceed.")
            return {
                'error'              : "No numeric features available for phenotyping.",
                'cluster_labels'     : np.array([]),
                'clustered_data_index': pd.Index([]),
                'n_clusters'         : 0,
                'silhouette_score'   : -1,
                'silhouette_values'  : np.array([]),
                'params'             : clustering_params
            }

        excluded_cols = [col for col in processed_data.columns if col not in numeric_data.columns]
        if excluded_cols:
            logger.warning(f"perform_auto_phenotyping: Excluded non-numeric columns from clustering: {', '.join(excluded_cols)}")

        # Remove rows with NaN values from numeric_data
        if numeric_data.isnull().any().any():
            rows_before_dropna = len(numeric_data)
            numeric_data = numeric_data.dropna()
            rows_after_dropna = len(numeric_data)
            if rows_after_dropna < rows_before_dropna:
                logger.warning(f"perform_auto_phenotyping: Removed {rows_before_dropna - rows_after_dropna} rows containing NaN values from data before clustering.")

        if numeric_data.empty:
            logger.error("perform_auto_phenotyping: No data available for clustering after removing NaN values. Cannot proceed.")
            return {
                'error'              : "No data available for phenotyping after NaN removal.",
                'cluster_labels'     : np.array([]),
                'clustered_data_index': pd.Index([]),
                'n_clusters'         : 0,
                'silhouette_score'   : -1,
                'silhouette_values'  : np.array([]),
                'params'             : clustering_params
            }

        # Smart defaults for clustering_params if None
        if clustering_params is None:
            clustering_params = {
                'n_clusters'  : None,     # Placeholder to find optimal K
                'method'      : 'kmeans',
                'random_state': 42
            }

        # Placeholder for finding optimal number of clusters
        # optimal_k = tekoa.clustering.find_optimal_clusters(processed_data, k_range=(2, 10))
        # For now, let's assume a fixed number or one from params if provided
        n_clusters = clustering_params.get('n_clusters')
        if n_clusters is None:
            # Simple heuristic for now: trying a few k values and picking the best silhouette score
            best_score  = -1
            best_k      = 2
            best_labels = None
            for k_try in range(2, min(6, len(processed_data)-1)): # Try k from 2 to 5 or less if not enough samples

                try:
                    kmeans = KMeans(n_clusters=k_try, random_state=clustering_params.get('random_state', 42), n_init='auto')
                    labels_try = kmeans.fit_predict(numeric_data)
                    if len(np.unique(labels_try)) > 1: # Silhouette score requires at least 2 labels
                        score_try = silhouette_score(numeric_data, labels_try)
                        if score_try > best_score:
                            best_score  = score_try
                            best_k      = k_try
                            best_labels = labels_try

                except Exception as e:
                    logger.warning(f"Error during optimal k search for k={k_try}: {e}")
                    continue

            if best_labels is None: # Fallback if all k_try failed
                kmeans         = KMeans(n_clusters=2, random_state=clustering_params.get('random_state', 42), n_init='auto') # Default to 2 clusters
                cluster_labels = kmeans.fit_predict(numeric_data)
                n_clusters     = 2

            else:
                cluster_labels = best_labels
                n_clusters     = best_k

            logger.info(f"Determined optimal number of clusters: {n_clusters} with silhouette score: {best_score if best_score != -1 else 'N/A'}")

        else:
            # Perform clustering using specified parameters (e.g., from tekoa.clustering)
            # cluster_labels = tekoa.clustering.perform_clustering(
            #     processed_data,
            #     method=clustering_params.get('method', 'kmeans'),
            #     n_clusters=n_clusters,
            #     random_state=clustering_params.get('random_state', 42)
            # )
            # Using KMeans placeholder
            kmeans = KMeans(n_clusters=n_clusters, random_state=clustering_params.get('random_state', 42), n_init='auto')
            cluster_labels = kmeans.fit_predict(numeric_data)

        # Calculate silhouette scores (e.g., from tekoa.clustering or sklearn)
        # silhouette_avg = tekoa.clustering.calculate_silhouette_scores(processed_data, cluster_labels)['average']
        # sample_silhouette_values = tekoa.clustering.calculate_silhouette_scores(processed_data, cluster_labels)['samples']
        if len(np.unique(cluster_labels)) > 1:
            silhouette_avg           = silhouette_score(numeric_data, cluster_labels)
            sample_silhouette_values = silhouette_samples(numeric_data, cluster_labels)

        else: # Handle cases with only one cluster (silhouette score is undefined)
            silhouette_avg           = -1 # Or some other indicator
            sample_silhouette_values = np.zeros(len(numeric_data))
            logger.warning("Only one cluster found, silhouette scores cannot be meaningfully computed.")


        results = {
            'cluster_labels'      : cluster_labels,
            'clustered_data_index': numeric_data.index,
            'n_clusters'          : n_clusters,
            'silhouette_score'    : silhouette_avg,
            'silhouette_values'   : sample_silhouette_values,
            'params'              : clustering_params
        }

        # Update session state
        st.session_state.phenotyping_results = results
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}

        st.session_state.pipeline_results['phenotyping'] = {
            'method'          : clustering_params.get('method', 'kmeans'),
            'n_clusters'      : n_clusters,
            'silhouette_score': silhouette_avg,
            'data_shape'      : processed_data.shape
        }
        logger.info(f"Auto-phenotyping complete. Found {n_clusters} clusters with score {silhouette_avg:.3f}.")
        return results

    def get_silhouette_plot_data(self, clustering_results: Optional[Dict] = None) -> Optional[Dict]:
        """
        Prepare data suitable for rendering a silhouette plot.

        Args:
            clustering_results: The output from perform_auto_phenotyping.
                                If None, tries to use st.session_state.phenotyping_results.

        Returns:
            Dictionary containing data for silhouette plot:
                - 'silhouette_values': Silhouette scores for each sample.
                - 'cluster_labels': Cluster label for each sample.
                - 'n_clusters': Number of clusters.
                - 'silhouette_avg': Average silhouette score.
        """
        if clustering_results is None:
            clustering_results = st.session_state.get('phenotyping_results')

        if not clustering_results or 'silhouette_values' not in clustering_results or 'cluster_labels' not in clustering_results:
            logger.warning("get_silhouette_plot_data: Clustering results not available or incomplete.")
            return None

        return {
            'silhouette_values': clustering_results['silhouette_values'],
            'cluster_labels'   : clustering_results['cluster_labels'],
            'n_clusters'       : clustering_results['n_clusters'],
            'silhouette_avg'   : clustering_results['silhouette_score']
        }

    def get_phenotype_radar_chart_data(self, phenotype_id: int, processed_data_with_clusters: pd.DataFrame, features: Optional[List[str]] = None) -> Optional[Dict[str, float]]:
        """
        Calculate mean values of selected features for a given phenotype (cluster).

        Args:
            phenotype_id: The cluster label (phenotype identifier).
            processed_data_with_clusters: DataFrame containing processed data and 'cluster_labels' column.
            features: Optional list of features for the radar chart. If None, all numeric features are used.

        Returns:
            Dictionary of feature means for the phenotype, or None if data is invalid.
        """
        if processed_data_with_clusters is None or 'cluster_labels' not in processed_data_with_clusters.columns:
            logger.warning("get_phenotype_radar_chart_data: Data with cluster labels is not available.")
            return None

        phenotype_data = processed_data_with_clusters[processed_data_with_clusters['cluster_labels'] == phenotype_id]

        if phenotype_data.empty:
            logger.warning(f"get_phenotype_radar_chart_data: No data found for phenotype_id {phenotype_id}.")
            return None

        if features is None:
            features = phenotype_data.select_dtypes(include=np.number).columns.tolist()
            if 'cluster_labels' in features: # Exclude the label itself
                features.remove('cluster_labels')

        # Ensure features exist in the dataframe
        valid_features = [f for f in features if f in phenotype_data.columns]
        if not valid_features:
            logger.warning(f"get_phenotype_radar_chart_data: None of the specified features are available in the data for phenotype_id {phenotype_id}.")
            return None

        radar_data = phenotype_data[valid_features].mean().to_dict()
        logger.info(f"Generated radar chart data for phenotype {phenotype_id} with {len(valid_features)} features.")
        return radar_data

    def get_patients_for_phenotype(self, phenotype_id: int, processed_data_with_clusters: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Get patient identifiers belonging to a specific phenotype (cluster).
        Assumes patient identifiers are in the index of the DataFrame.

        Args:
            phenotype_id: The cluster label (phenotype identifier).
            processed_data_with_clusters: DataFrame containing processed data, cluster labels, and patient IDs in index.

        Returns:
            DataFrame containing patient identifiers for the phenotype, or None.
        """
        if processed_data_with_clusters is None or 'cluster_labels' not in processed_data_with_clusters.columns:
            logger.warning("get_patients_for_phenotype: Data with cluster labels is not available.")
            return None

        patients_data = processed_data_with_clusters[processed_data_with_clusters['cluster_labels'] == phenotype_id]

        if patients_data.empty:
            logger.warning(f"get_patients_for_phenotype: No patients found for phenotype_id {phenotype_id}.")
            return pd.DataFrame() # Return empty DataFrame

        logger.info(f"Retrieved {len(patients_data)} patients for phenotype {phenotype_id}.")
        return patients_data.index.to_frame() # Assuming patient IDs are in the index

    def save_phenotype_name(self, phenotype_id: int, phenotype_name: str):
        """
        Store a user-defined name for a given phenotype_id.

        Args:
            phenotype_id: The cluster label (phenotype identifier).
            phenotype_name: The user-defined name for the phenotype.
        """
        if 'phenotype_names' not in st.session_state:
            st.session_state.phenotype_names = {}
        st.session_state.phenotype_names[phenotype_id] = phenotype_name
        logger.info(f"Saved name '{phenotype_name}' for phenotype_id {phenotype_id}.")

    def get_phenotype_name(self, phenotype_id: int) -> str:
        """
        Retrieve the user-defined name for a phenotype_id.

        Args:
            phenotype_id: The cluster label (phenotype identifier).

        Returns:
            The user-defined name, or a default name if not set.
        """
        if 'phenotype_names' not in st.session_state:
            st.session_state.phenotype_names = {}
        return st.session_state.phenotype_names.get(phenotype_id, f"Phenotype {phenotype_id}")

    def get_phenotype_stability_data(self, clustering_results: Optional[Dict] = None) -> Optional[Dict]:
        """
        Get phenotype stability metrics using bootstrap validation.
        (Placeholder for tekoa.validation.calculate_bootstrap_stability)

        Args:
            clustering_results: The output from perform_auto_phenotyping.
                                If None, tries to use st.session_state.phenotyping_results.

        Returns:
            Dictionary containing stability data (e.g., Jaccard index, silhouette distributions).
        """
        if clustering_results is None:
            clustering_results = st.session_state.get('phenotyping_results')

        if not clustering_results or st.session_state.processed_data is None:
            logger.warning("get_phenotype_stability_data: Clustering results or processed data not available.")
            return None

        # Placeholder call to a tekoa validation function
        # stability_metrics = tekoa.validation.calculate_bootstrap_stability(
        #     st.session_state.processed_data,
        #     clustering_results['cluster_labels'],
        #     n_bootstraps=50, # Example parameter
        #     clustering_params=clustering_results.get('params', {})
        # )
        # Mocked stability data for now
        logger.info("Calculating phenotype stability (using mocked data)...")
        mock_jaccard_scores = np.random.rand(clustering_results['n_clusters']) * 0.2 + 0.75 # Scores between 0.75 and 0.95
        mock_silhouette_dist = [np.random.normal(loc=clustering_results['silhouette_score'], scale=0.05, size=50) for _ in range(clustering_results['n_clusters'])]

        stability_metrics = {
            'jaccard_scores_per_cluster': mock_jaccard_scores.tolist(),
            'avg_jaccard_score': np.mean(mock_jaccard_scores).item(),
            'silhouette_bootstrap_distributions': [s.tolist() for s in mock_silhouette_dist], # per cluster
            'n_bootstraps': 50 # Example
        }

        # Update pipeline results
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}
        if 'phenotyping' not in st.session_state.pipeline_results:
             st.session_state.pipeline_results['phenotyping'] = {}
        st.session_state.pipeline_results['phenotyping']['stability'] = {
            'avg_jaccard_score': stability_metrics['avg_jaccard_score'],
            'n_bootstraps': stability_metrics['n_bootstraps']
        }

        logger.info(f"Phenotype stability calculated (mocked): Avg Jaccard {stability_metrics['avg_jaccard_score']:.3f}")
        return stability_metrics

