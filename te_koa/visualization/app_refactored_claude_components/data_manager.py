"""
Data manager for the TE-KOA dashboard.

This module provides a central manager for loading, processing, and managing data
for the TE-KOA dashboard.
"""

import logging
import pandas as pd
import numpy as np
import streamlit as st
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from te_koa.io.data_loader import DataLoader
from te_koa.utils.variable_screener import VariableScreener
from te_koa.utils.dimensionality_reducer import DimensionalityReducer
from te_koa.utils.data_quality_enhancer import DataQualityEnhancer
from te_koa.configurations.params import DatasetNames

logger = logging.getLogger(__name__)

class DataManager:
    """Manager for TE-KOA data loading, processing, and state management."""

    def __init__(self):
        """Initialize the data manager."""
        self.data_loader: Optional[DataLoader] = None
        self.dataset_name = DatasetNames.TE_KOA.value
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
        if 'phenotypes' not in st.session_state:
            st.session_state.phenotypes = None
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Overview'

    def load_data(self) -> bool:
        """
        Load the TE-KOA dataset.

        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        # Check if file is uploaded in session state
        if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
            uploaded_file = st.session_state.uploaded_file
            logger.info(f"Loading data from uploaded file: {uploaded_file.name}")

            # Create DataLoader with uploaded file
            self.data_loader = DataLoader(uploaded_file=uploaded_file)

        if self.data_loader is None:
            return False

        # Load data using the data loader
        self.data, self.dictionary = self.data_loader.load_data()
        self.missing_data_report = self.data_loader.get_missing_data_report()

        # Initialize data components if data loaded successfully
        if self.data is not None:
            st.session_state.processed_data = self.data.copy()
            st.session_state.data_loaded_time = pd.Timestamp.now()

            logger.info(f"Successfully loaded data: {len(self.data)} rows, {len(self.data.columns)} columns")

            # Check for required treatment column
            if 'tx.group' not in self.data.columns:
                logger.warning("tx.group column not found in dataset")

            return True

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

    def screen_variables(self, near_zero_threshold: float, collinearity_threshold: float,
                         vif_threshold: float, force_include: List[str]) -> Dict[str, Any]:
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
                variables=variables,
                n_components=n_components,
                standardize=standardize
            )
        elif method == 'famd':
            results = self.dimensionality_reducer.perform_famd(
                n_components=n_components
            )
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
