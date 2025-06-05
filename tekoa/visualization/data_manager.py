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

from tekoa import logger
from tekoa.io import DataLoader
from tekoa.utils import (
    VariableScreener,
    DimensionalityReducer,
    DataQualityEnhancer,
)
from tekoa.utils.phenotype_discovery import PhenotypeDiscovery
from tekoa.configuration.params import DatasetNames



class DataManager:
    """Manager for TE-KOA data loading, processing, and state management."""

    def __init__(self):
        """Initialize the data manager."""
        self.data_loader: Optional[DataLoader] = None
        self.dataset_name = DatasetNames.TEKOA.value
        self._raw_data: Optional[pd.DataFrame] = None  # Stores the original loaded data
        self.dictionary: Optional[pd.DataFrame] = None
        self.imputed_data: Optional[pd.DataFrame] = None
        self.treatment_groups: Optional[Dict[str, pd.DataFrame]] = None
        self.missing_data_report: Optional[pd.DataFrame] = None

        # Analysis components
        self.variable_screener: Optional[VariableScreener] = None
        self.dimensionality_reducer: Optional[DimensionalityReducer] = None
        self.data_quality_enhancer: Optional[DataQualityEnhancer] = None
        self.phenotype_discovery: Optional[PhenotypeDiscovery] = None

        # Initialize session state for data management
        self._initialize_session_state()

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """
        Get the current working dataset, potentially filtered to exclude follow-up data.
        Returns None if no data is loaded (_raw_data is None).
        """
        if self._raw_data is None:
            return None

        if st.session_state.get('ignore_follow_up_data', False):
            if self.data_loader:
                variable_categories = self.data_loader.get_variable_categories()
                follow_up_cols = variable_categories.get('follow_up', [])
                # Ensure we only try to drop columns that actually exist in _raw_data
                cols_to_drop = [col for col in follow_up_cols if col in self._raw_data.columns]
                if cols_to_drop:
                    logger.info(f"DataManager: Ignoring follow-up columns: {cols_to_drop}")
                    return self._raw_data.drop(columns=cols_to_drop)
            else:
                logger.warning("DataManager: ignore_follow_up_data is True, but DataLoader is not available to get follow_up columns.")
        return self._raw_data

    def _initialize_session_state(self):
        """Initialize session state variables for data management."""
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}
        if 'phenotypes' not in st.session_state:
            st.session_state.phenotypes = None
        if 'phenotype_results' not in st.session_state:
            st.session_state.phenotype_results = {}
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
        # Reset relevant states on new load attempt
        self.data_loader = None
        self._raw_data = None
        self.dictionary = None
        self.missing_data_report = None
        self.imputed_data = None
        st.session_state.processed_data = None
        # Potentially reset other downstream analysis components if they hold state
        self.variable_screener = None
        self.dimensionality_reducer = None
        self.data_quality_enhancer = None
        self.phenotype_discovery = None

        if uploaded_file_obj is not None:
            file_name_for_log = getattr(uploaded_file_obj, 'name', 'uploaded_file')
            logger.info(f"DataManager attempting to load data from provided uploaded file: {file_name_for_log}")
            self.data_loader = DataLoader(uploaded_file=uploaded_file_obj)
        else:
            logger.warning("DataManager.load_data called without an 'uploaded_file_obj'. No data will be loaded.")
            return False

        if self.data_loader is None:
            logger.error("DataManager could not initialize DataLoader.")
            return False

        try:
            self._raw_data, self.dictionary = self.data_loader.load_data()
            self.missing_data_report = self.data_loader.get_missing_data_report()

            if self._raw_data is not None and self.dictionary is not None:
                # Access self.data via property to get potentially filtered data
                current_data_view = self.data 
                if current_data_view is not None:
                    st.session_state.processed_data = current_data_view.copy()
                    log_msg_data_part = f"Current view: {len(current_data_view)} rows, {len(current_data_view.columns)} columns."
                    if 'tx.group' not in current_data_view.columns:
                        logger.warning("'tx.group' column not found in the current data view.")
                else: # This case implies _raw_data might be empty or filtering resulted in None
                    st.session_state.processed_data = None # Ensure it's None if current_data_view is None
                    log_msg_data_part = "Current view is None (e.g. all columns filtered out or raw data empty)."
                
                st.session_state.data_loaded_time = pd.Timestamp.now()
                log_file_name = getattr(uploaded_file_obj, 'name', 'the_uploaded_file')
                logger.info(f"DataManager successfully loaded data. Raw: {len(self._raw_data)} rows, {len(self._raw_data.columns)} columns. {log_msg_data_part} From {log_file_name}")
                return True
            else:
                log_file_name_fail = getattr(uploaded_file_obj, 'name', 'the_uploaded_file')
                logger.error(f"DataLoader failed to load data from {log_file_name_fail} (returned None for _raw_data/dictionary) within DataManager.")
                self._raw_data = None
                self.dictionary = None
                self.missing_data_report = None
                return False
        except Exception as e:
            log_file_name_exc = getattr(uploaded_file_obj, 'name', 'the_uploaded_file')
            logger.error(f"Exception occurred in DataManager.load_data while processing {log_file_name_exc}: {e}", exc_info=True)
            self._raw_data = None
            self.dictionary = None
            self.missing_data_report = None
            return False

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the current processed data.
        If st.session_state.processed_data is None (e.g., after a reset or initial load),
        it attempts to re-initialize it from the base self.data property.
        """
        if st.session_state.get('processed_data') is None and self._raw_data is not None:
            # self.data is the property that applies filtering based on 'ignore_follow_up_data'
            current_data_view = self.data 
            if current_data_view is not None:
                st.session_state.processed_data = current_data_view.copy()
                logger.info("DataManager.get_data: Re-initialized st.session_state.processed_data from self.data property.")
            else:
                # This could happen if self.data (property) returns None, e.g., _raw_data was empty or all columns were filtered out.
                st.session_state.processed_data = None 
                logger.info("DataManager.get_data: self.data property returned None; st.session_state.processed_data remains None.")
        elif st.session_state.get('processed_data') is None and self._raw_data is None:
            logger.info("DataManager.get_data: No raw data loaded, processed_data is None.")
            
        return st.session_state.get('processed_data')

    def get_original_data(self) -> Optional[pd.DataFrame]:
        """Get the original, unfiltered, unprocessed data as loaded from the source."""
        return self._raw_data

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

    def impute_missing_values(self, method: str, knn_neighbors: int, cols_to_exclude: List[str]) -> Optional[pd.DataFrame]:
        """
        Impute missing values in the current working dataset (which respects 'ignore_follow_up_data').

        Args:
            method: Imputation method ('mean', 'median', 'knn')
            knn_neighbors: Number of neighbors for KNN imputation
            cols_to_exclude: List of columns to exclude from imputation

        Returns:
            DataFrame with imputed values, or None if imputation fails or no data.
        """
        current_data_to_impute = self.data # Access data via property to get the (potentially filtered) view
        
        if current_data_to_impute is None:
            logger.error("DataManager.impute_missing_values: No data available for imputation (current data view is None).")
            self.imputed_data = None
            st.session_state.processed_data = None # Ensure processed_data is also None
            return None
        
        if self.data_loader is None:
            logger.error("DataManager.impute_missing_values: DataLoader not initialized, cannot impute.")
            self.imputed_data = None
            st.session_state.processed_data = None
            return None

        # Pass a copy of the current data view to the imputation method in DataLoader
        # DataLoader's impute_missing_values is expected to handle the actual imputation logic.
        imputed_df = self.data_loader.impute_missing_values(
            data_to_impute=current_data_to_impute.copy(), # Important: pass a copy of the current view
            method=method,
            knn_neighbors=knn_neighbors,
            cols_to_exclude=cols_to_exclude
        )

        self.imputed_data = imputed_df # Store the result of imputation
        st.session_state.processed_data = imputed_df # Update session_state.processed_data with the imputed data

        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = {}
        
        st.session_state.pipeline_results['imputation'] = {
            'method': method,
            'knn_neighbors': knn_neighbors,
            'cols_excluded': cols_to_exclude,
            'timestamp': time.time(),
            'data_shape_before': current_data_to_impute.shape,
            'data_shape_after': imputed_df.shape if imputed_df is not None else None
        }
        
        if imputed_df is not None:
            logger.info(f"DataManager: Imputed missing values using {method} method on current data view. Excluded columns: {cols_to_exclude}. Shape before: {current_data_to_impute.shape}, after: {imputed_df.shape}")
        else:
            logger.warning(f"DataManager: Imputation with method {method} returned None.")
        
        return imputed_df

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

    def initialize_phenotype_discovery(self, use_transformed_data: bool = False):
        """Initialize the phenotype discovery module.

        Args:
            use_transformed_data: Whether to use dimensionality-reduced data
        """
        data_to_use = st.session_state.processed_data

        # Check if we should use transformed data (e.g., FAMD components)
        if use_transformed_data and 'dimensionality_reduction' in st.session_state.pipeline_results:
            dim_results = st.session_state.pipeline_results['dimensionality_reduction']
            if 'transformed_data_shape' in dim_results:
                # Data has been transformed, use it
                data_to_use = st.session_state.processed_data
                logger.info("Using transformed data for phenotype discovery")
            else:
                logger.warning("Transformed data not available, using processed data")

        if self.phenotype_discovery is None and data_to_use is not None:
            self.phenotype_discovery = PhenotypeDiscovery(data_to_use)

    def perform_clustering(self, method: str, n_clusters_range: range, columns_to_use: Optional[List[str]] = None) -> Dict:
        """Perform clustering using specified method.

        Args:
            method: Clustering method ('kmeans', 'agglomerative', 'gmm')
            n_clusters_range: Range of cluster numbers to try
            columns_to_use: Optional list of column names to use for clustering.
                            If None, uses all available features or transformed components.

        Returns:
            Dictionary with clustering results
        """
        # Ensure phenotype_discovery is initialized.
        # The use_transformed_data flag in initialize_phenotype_discovery is handled by the UI page before calling this.
        if self.phenotype_discovery is None:
            # This should ideally be called from the page UI logic based on the checkbox
            # For safety, we can call it here, but it might re-initialize with default if not set by UI
            # Consider if initialize_phenotype_discovery needs to be aware of columns_to_use if not using transformed data.
            # For now, assume initialize_phenotype_discovery has set up data_for_clustering correctly.
            st.warning("PhenotypeDiscovery not initialized prior to perform_clustering. Attempting default initialization.")
            self.initialize_phenotype_discovery(use_transformed_data=True) # Defaulting to true, UI should control this
            if self.phenotype_discovery is None:
                st.error("Failed to initialize PhenotypeDiscovery.")
                return {}

        if method == 'kmeans':
            results = self.phenotype_discovery.perform_kmeans(n_clusters_range, columns_to_use=columns_to_use)
        elif method == 'agglomerative':
            results = self.phenotype_discovery.perform_agglomerative(n_clusters_range, columns_to_use=columns_to_use)
        elif method == 'gmm':
            results = self.phenotype_discovery.perform_gmm(n_clusters_range, columns_to_use=columns_to_use)
        else:
            st.error(f"Unknown clustering method: {method}")
            raise ValueError(f"Unknown clustering method: {method}")

        # Store results in session state
        if 'phenotype_results' not in st.session_state:
            st.session_state.phenotype_results = {}
        if method not in st.session_state.phenotype_results:
            st.session_state.phenotype_results[method] = {}
        st.session_state.phenotype_results[method]['clustering'] = results

        return results

    def calculate_gap_statistic(self, method: str, n_clusters_range: range, n_references: int = 10) -> Dict:
        """Calculate gap statistic for optimal cluster determination.

        Args:
            method: Clustering method
            n_clusters_range: Range of cluster numbers
            n_references: Number of reference datasets

        Returns:
            Dictionary with gap statistics
        """
        self.initialize_phenotype_discovery()

        results = self.phenotype_discovery.calculate_gap_statistic(
            method=method,
            n_clusters_range=n_clusters_range,
            n_references=n_references
        )

        # Store in session state
        if method not in st.session_state.phenotype_results:
            st.session_state.phenotype_results[method] = {}
        st.session_state.phenotype_results[method]['gap_statistic'] = results

        return results

    def assess_bootstrap_stability(self, method: str, k: int, n_bootstrap: int = 100, subsample_size: float = 0.8) -> Dict:
        """Assess clustering stability using bootstrap.

        Args:
            method: Clustering method
            k: Number of clusters
            n_bootstrap: Number of bootstrap samples
            subsample_size: Proportion of data to subsample

        Returns:
            Dictionary with stability metrics
        """
        self.initialize_phenotype_discovery()

        results = self.phenotype_discovery.bootstrap_stability(
            method=method,
            k=k,
            n_bootstrap=n_bootstrap,
            subsample_size=subsample_size
        )

        # Store in session state
        if method not in st.session_state.phenotype_results:
            st.session_state.phenotype_results[method] = {}
        if 'stability' not in st.session_state.phenotype_results[method]:
            st.session_state.phenotype_results[method]['stability'] = {}
        st.session_state.phenotype_results[method]['stability'][k] = results

        return results

    def determine_optimal_clusters(self, min_cluster_size: int = 10) -> Dict:
        """Determine optimal number of clusters.

        Args:
            min_cluster_size: Minimum samples per cluster

        Returns:
            Dictionary with recommendations
        """
        self.initialize_phenotype_discovery()

        recommendations = self.phenotype_discovery.determine_optimal_clusters(min_cluster_size)

        # Store in session state
        st.session_state.phenotype_results['optimal_clusters'] = recommendations

        return recommendations

    def characterize_phenotypes(self, method: str, k: int = None) -> pd.DataFrame:
        """Create statistical characterization of phenotypes.

        Args:
            method: Clustering method
            k: Number of clusters

        Returns:
            DataFrame with phenotype characteristics
        """
        self.initialize_phenotype_discovery()

        char_df = self.phenotype_discovery.characterize_phenotypes(method, k)

        # Store in session state
        if method not in st.session_state.phenotype_results:
            st.session_state.phenotype_results[method] = {}
        st.session_state.phenotype_results[method]['characterization'] = char_df

        return char_df

    def compare_phenotypes(self, method: str, k: int = None, variables: List[str] = None) -> Dict:
        """Compare phenotypes statistically.

        Args:
            method: Clustering method
            k: Number of clusters
            variables: Variables to compare

        Returns:
            Dictionary with comparison results
        """
        self.initialize_phenotype_discovery()

        comparison = self.phenotype_discovery.compare_phenotypes(method, k, variables)

        # Store comparison results
        self.phenotype_discovery.comparison_results = comparison

        # Store in session state
        if method not in st.session_state.phenotype_results:
            st.session_state.phenotype_results[method] = {}
        st.session_state.phenotype_results[method]['comparison'] = comparison

        return comparison

    def export_phenotype_assignments(self, method: str, k: int = None) -> pd.DataFrame:
        """Export phenotype assignments.

        Args:
            method: Clustering method
            k: Number of clusters

        Returns:
            DataFrame with phenotype assignments
        """
        self.initialize_phenotype_discovery()

        phenotype_data = self.phenotype_discovery.export_phenotype_assignments(method, k)

        # Store as current phenotypes
        st.session_state.phenotypes = phenotype_data

        return phenotype_data
