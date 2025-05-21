"""
Enhanced Knee Osteoarthritis Data Loader.

This module provides comprehensive functionality for loading and preprocessing the Knee Osteoarthritis dataset.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


DATASET_NAME = 'te_koa_R01_only_RCT_data.xlsx'
class DataLoader:
    """An enhanced data loader for the Knee Osteoarthritis dataset."""

    def __init__(self, data_dir: Optional[str] = None, uploaded_file: Optional[str] = None):
        """
        Initialize the DataLoader.

        Args:
            data_dir: Path to the data directory or default Excel file.
            uploaded_file: An uploaded file object (e.g., from Streamlit), path to an Excel file,
                           or any object pandas.read_excel can handle. If provided, this will be used
                           for loading data, taking precedence over data_dir.
        """
        self.uploaded_file = uploaded_file  # Store the uploaded_file instance variable
        if data_dir is None:
            # Try to find a reasonable default path
            possible_paths = [
                f"/Users/artinmajdi/Documents/GitHubs/RAP/te_koa_c__lee/dataset/{DATASET_NAME}",
                f"./dataset/{DATASET_NAME}",
                f"./{DATASET_NAME}"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    self.data_dir = Path(path)
                    break
            else:
                # If no path is found, use current directory
                self.data_dir = Path.cwd() / DATASET_NAME
        else:
            self.data_dir = Path(data_dir)

        # Keep track of loaded data
        self.data = None
        self.data_dict = None
        self.missing_data_report = None
        self.variable_categories = None

    def load_data(self, sheet_name: str = "Sheet1", dictionary_sheet: str = "dictionary", **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the Knee Osteoarthritis dataset and data dictionary.
        Prioritizes `uploaded_file` if available, otherwise uses `data_dir`.

        Args:
            sheet_name: Name of the sheet containing the actual data
            dictionary_sheet: Name of the sheet containing the data dictionary
            **kwargs: Additional arguments to pass to pd.read_excel

        Returns:
            Tuple containing (data DataFrame, data dictionary DataFrame)

        Raises:
            FileNotFoundError: If the specified file path (from data_dir or uploaded_file if it's a path) doesn't exist.
            ValueError: If no data source (neither uploaded_file nor data_dir) is configured.
            Exception: For other pandas or I/O related loading errors.
        """
        source_to_load = None
        source_name_for_log = "unknown source" # Default for logging

        if self.uploaded_file:
            source_to_load = self.uploaded_file
            # Try to get a descriptive name for logging
            if hasattr(self.uploaded_file, 'name') and getattr(self.uploaded_file, 'name'): # Check if name exists and is not empty
                source_name_for_log = getattr(self.uploaded_file, 'name')
            elif isinstance(self.uploaded_file, str):
                source_name_for_log = Path(self.uploaded_file).name
            else: # It's some other object, or name attribute is empty/None
                source_name_for_log = "uploaded data"
            logger.info(f"Attempting to load data from uploaded source: '{source_name_for_log}'")
        elif self.data_dir:
            source_to_load = self.data_dir
            source_name_for_log = str(self.data_dir)
            logger.info(f"Attempting to load data from configured path: {source_name_for_log}")
        else:
            logger.error("DataLoader is not configured with a data source (uploaded_file or data_dir).")
            raise ValueError("No data source configured for DataLoader.")

        if source_to_load is None: # Safeguard, should be caught by the logic above
             logger.critical("Internal error: source_to_load is None before pd.read_excel. This should not happen.")
             raise ValueError("Internal configuration error: data source became None unexpectedly.")

        try:
            # Load both data and dictionary sheets
            data = pd.read_excel(source_to_load, sheet_name=sheet_name, **kwargs)
            data_dict = pd.read_excel(source_to_load, sheet_name=dictionary_sheet, **kwargs)

            logger.info(f"Successfully loaded {len(data)} rows from '{sheet_name}' sheet of '{source_name_for_log}'")
            logger.info(f"Successfully loaded {len(data_dict)} rows from '{dictionary_sheet}' sheet of '{source_name_for_log}'")

            # Store the loaded data
            self.data = data
            self.data_dict = data_dict

            # Ensure the data dictionary has appropriate column names
            self._normalize_data_dictionary()

            # Categorize variables by type and time point
            self._categorize_variables()

            # Generate missing data report
            self._analyze_missing_data()

            return data, data_dict

        except FileNotFoundError:
            logger.error(f"File not found at path: {str(source_to_load)}")
            raise
        except Exception as e:
            logger.error(f"Error loading Excel data from '{source_name_for_log}': {e}")
            raise

    def _normalize_data_dictionary(self):
        """Normalize the data dictionary columns to ensure consistent access."""
        # Assuming first column is variable name and second is description
        if len(self.data_dict.columns) >= 2:
            self.data_dict.columns = ['Variable', 'Description'] + list(self.data_dict.columns[2:])

        # Make sure column names are strings
        self.data_dict.columns = [str(col) for col in self.data_dict.columns]

    def _categorize_variables(self):
        """Categorize variables based on their names and purposes."""
        if self.data is None or self.data_dict is None:
            logger.warning("Data not loaded. Cannot categorize variables.")
            return

        # Initialize categories
        categories = {
            'demographic': [],
            'clinical': [],
            'treatment': [],
            'outcome': [],
            'baseline': [],
            'follow_up': [],
            'pain_assessment': [],
            'physical_assessment': [],
            'psychological': [],
            'qst': [],  # Quantitative Sensory Testing
            'biomarker': []
        }

        # Define patterns for categorization
        patterns = {
            'demographic': ['age', 'gender', 'race', 'education', 'height', 'weight', 'history', 'bmi'],
            'clinical': ['knee', 'pain', 'klscore', 'womac', 'oa', 'joint'],
            'treatment': ['tdcs', 'tx.group', 'meditation', 'treatment'],
            'outcome': ['differ', 'm1', 'm2', 'm3', 'change'],
            'baseline': ['_0', '.0', 'baseline'],
            'follow_up': ['_5', '_10', '.5', '.10', 'follow'],
            'pain_assessment': ['pain', 'womac', 'nrs'],
            'physical_assessment': ['functional', 'physical', 'step'],
            'psychological': ['pcs', 'fmi', 'css', 'cesd', 'catastrophizing', 'depression'],
            'qst': ['hpth', 'hpto', 'ppt', 'qst', 'temperature', 'punctate', 'cpm']
        }

        # Categorize each variable
        for col in self.data.columns:
            col_lower = col.lower()

            # Add to relevant categories based on patterns
            for category, pattern_list in patterns.items():
                if any(pattern in col_lower for pattern in pattern_list):
                    categories[category].append(col)

        # Store categories
        self.variable_categories = categories

    def _analyze_missing_data(self) -> pd.DataFrame:
        """
        Analyze missing data in the dataset.

        Returns:
            DataFrame containing missing data statistics
        """
        if self.data is None:
            logger.warning("No data loaded. Call load_data() first.")
            return None

        # Calculate missing values statistics
        missing = self.data.isnull().sum()
        missing_percent = (self.data.isnull().sum() / len(self.data)) * 100
        data_types = self.data.dtypes

        # Create a report
        missing_data_report = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_percent,
            'Data Type': data_types
        })

        # Sort by percentage of missing values, descending
        missing_data_report = missing_data_report.sort_values('Percentage', ascending=False)

        # Add variable descriptions
        if self.data_dict is not None:
            descriptions = {}

            for var in missing_data_report.index:
                description = self.get_variable_description(var)
                descriptions[var] = description or "No description available"

            missing_data_report['Description'] = pd.Series(descriptions)

        # Store the report
        self.missing_data_report = missing_data_report

        return missing_data_report

    def get_missing_data_report(self) -> pd.DataFrame:
        """
        Get the missing data report.

        Returns:
            DataFrame containing missing data statistics
        """
        if self.missing_data_report is None:
            logger.warning("No missing data report available. Call load_data() first.")
            return None

        return self.missing_data_report

    def impute_missing_values(self, method: str = 'knn', knn_neighbors: int = 5, cols_to_exclude: List[str] = None) -> pd.DataFrame:
        """
        Impute missing values in the dataset.

        Args:
            method: Imputation method ('mean', 'median', 'knn')
            knn_neighbors: Number of neighbors to use for KNN imputation
            cols_to_exclude: List of columns to exclude from imputation

        Returns:
            DataFrame with imputed values
        """
        if self.data is None:
            logger.warning("No data loaded. Call load_data() first.")
            return None

        # Make a copy of the data
        imputed_data = self.data.copy()

        # Determine columns to impute
        if cols_to_exclude is None:
            cols_to_exclude = []

        # Get numerical columns with missing values
        numeric_cols = imputed_data.select_dtypes(include=['float64', 'int64']).columns
        cols_to_impute = [col for col in numeric_cols if col not in cols_to_exclude]

        logger.info(f"Imputing missing values for columns: {cols_to_impute}")

        if method == 'knn':
            # KNN imputation
            imputer = KNNImputer(n_neighbors=knn_neighbors)

            # Scale the data before imputation
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(imputed_data[cols_to_impute])

            # Impute the scaled data
            imputed_scaled_data = imputer.fit_transform(scaled_data)

            # Create a DataFrame with the imputed values
            imputed_df = pd.DataFrame(
                scaler.inverse_transform(imputed_scaled_data),
                columns=cols_to_impute,
                index=imputed_data.index
            )

            # Update the imputed columns
            for col in cols_to_impute:
                imputed_data[col] = imputed_df[col]

        elif method == 'mean':
            # Mean imputation
            for col in cols_to_impute:
                imputed_data[col] = imputed_data[col].fillna(imputed_data[col].mean())

        elif method == 'median':
            # Median imputation
            for col in cols_to_impute:
                imputed_data[col] = imputed_data[col].fillna(imputed_data[col].median())

        else:
            logger.warning(f"Unknown imputation method: {method}. No imputation performed.")

        return imputed_data

    def get_treatment_groups(self) -> Dict[str, pd.DataFrame]:
        """
        Split the data into treatment groups based on tx.group variable.

        Based on the data dictionary, tx.group has values:
        0 = sham (control)
        1 = experimental
        2 = tDCS
        3 = meditation

        Returns:
            Dictionary of DataFrames for each treatment group
        """
        if self.data is None:
            logger.warning("No data loaded. Call load_data() first.")
            return None

        # Check if tx.group column exists
        tx_group_col = 'tx.group'
        if tx_group_col not in self.data.columns:
            logger.warning(f"Treatment group column '{tx_group_col}' not found. Looking for alternatives...")

            # Try to find an alternative treatment group column
            for col in self.data.columns:
                if 'tx' in col.lower() or 'treat' in col.lower() or 'group' in col.lower():
                    tx_group_col = col
                    logger.info(f"Using '{tx_group_col}' as the treatment group column.")
                    break
            else:
                logger.error("Could not find a treatment group column.")
                return None

        try:
            # Define groups based on tx.group values
            # From the data dictionary: 0=sham, 1=experimental, 2=tDCS, 3=meditation
            groups = {
                'Control (Sham)': self.data[self.data[tx_group_col] == 0],
                'Experimental': self.data[self.data[tx_group_col] == 1],
                'tDCS': self.data[self.data[tx_group_col] == 2],
                'Meditation': self.data[self.data[tx_group_col] == 3]
            }

            # Create 2x2 factorial design groups
            tdcs_groups = self.data[self.data[tx_group_col].isin([1, 2])]
            meditation_groups = self.data[self.data[tx_group_col].isin([1, 3])]

            groups.update({
                'Control (No tDCS, No Meditation)': self.data[self.data[tx_group_col] == 0],
                'tDCS Only': self.data[self.data[tx_group_col] == 2],
                'Meditation Only': self.data[self.data[tx_group_col] == 3],
                'tDCS + Meditation': self.data[self.data[tx_group_col] == 1]
            })

            # Log the group sizes
            for group, df in groups.items():
                logger.info(f"{group}: {len(df)} patients")

            return groups

        except Exception as e:
            logger.error(f"Error creating treatment groups: {e}")
            return None

    def get_variable_description(self, variable_name: str) -> str:
        """
        Get the description of a variable from the data dictionary.

        Args:
            variable_name: Name of the variable

        Returns:
            Description of the variable or None if not found
        """
        if self.data_dict is None:
            logger.warning("No data dictionary loaded. Call load_data() first.")
            return None

        # Try to find the variable in the data dictionary
        try:
            # Look for exact match
            matches = self.data_dict[self.data_dict['Variable'] == variable_name]

            if len(matches) == 0:
                # Try case-insensitive match
                matches = self.data_dict[self.data_dict['Variable'].str.lower() == variable_name.lower()]

            if len(matches) == 0:
                # Try substring match
                matches = self.data_dict[self.data_dict['Variable'].str.contains(variable_name, case=False, regex=False)]

            if len(matches) > 0:
                return matches.iloc[0]['Description']
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting variable description: {e}")
            return None

    def get_variable_categories(self, category: str = None) -> Union[Dict[str, List[str]], List[str]]:
        """
        Get variables categorized by type or purpose.

        Args:
            category: Specific category to retrieve. If None, returns all categories.

        Returns:
            Dictionary of categories or list of variables for a specific category
        """
        if self.variable_categories is None:
            logger.warning("Variables not categorized. Call load_data() first.")
            return {} if category is None else []

        if category is not None:
            return self.variable_categories.get(category, [])
        else:
            return self.variable_categories

    def save_processed_data(self, df: pd.DataFrame, filename: str, **kwargs) -> None:
        """
        Save processed data to a CSV file.

        Args:
            df: DataFrame to save
            filename: Name of the CSV file (relative to data_dir)
            **kwargs: Additional arguments to pass to df.to_csv
        """
        # If data_dir is an Excel file, use its parent directory
        if self.data_dir.is_file():
            save_dir = self.data_dir.parent
        else:
            save_dir = self.data_dir

        file_path = save_dir / filename
        logger.info(f"Saving {len(df)} rows to CSV file {file_path}")

        # Ensure directory exists
        os.makedirs(file_path.parent, exist_ok=True)

        try:
            df.to_csv(file_path, **kwargs)
            logger.info(f"Successfully saved data to {filename}")
        except Exception as e:
            logger.error(f"Error saving CSV file {filename}: {e}")
            raise

    def prepare_data_pipeline(self,
                             impute_method='knn',
                             screen_variables=True,
                             reduce_dimensions=True,
                             enhance_quality=True,
                             **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the complete data preparation pipeline.

        Args:
            impute_method: Method to use for imputation ('knn', 'mean', 'median')
            screen_variables: Whether to perform variable screening
            reduce_dimensions: Whether to perform dimensionality reduction
            enhance_quality: Whether to perform data quality enhancement
            **kwargs: Additional parameters for the pipeline steps

        Returns:
            Tuple of (processed DataFrame, pipeline results dictionary)
        """
        from tekoa.utils.variable_screener import VariableScreener
        from tekoa.utils.dimensionality_reducer import DimensionalityReducer
        from tekoa.utils.data_quality_enhancer import DataQualityEnhancer

        # Check if data is loaded
        if self.data is None:
            logger.warning("No data loaded. Call load_data() first.")
            return None, {}

        # Initialize results dictionary
        pipeline_results = {
            'imputation': {},
            'variable_screening': {},
            'dimensionality_reduction': {},
            'data_quality': {}
        }

        # Step 1: Impute missing values
        logger.info("Step 1: Imputing missing values...")
        processed_data = self.impute_missing_values(
            method=impute_method,
            knn_neighbors=kwargs.get('knn_neighbors', 5),
            cols_to_exclude=kwargs.get('cols_to_exclude', [])
        )

        pipeline_results['imputation'] = {
            'method': impute_method,
            'knn_neighbors': kwargs.get('knn_neighbors', 5),
            'cols_excluded': kwargs.get('cols_to_exclude', [])
        }

        # Step 2: Variable screening
        if screen_variables:
            logger.info("Step 2: Performing variable screening...")
            screener = VariableScreener(processed_data)

            # Detect near-zero variance
            screener.identify_near_zero_variance(
                threshold=kwargs.get('near_zero_threshold', 0.01)
            )

            # Analyze collinearity
            screener.analyze_collinearity(
                threshold=kwargs.get('collinearity_threshold', 0.85)
            )

            # Calculate VIF
            screener.calculate_vif(
                max_vif=kwargs.get('vif_threshold', 5.0)
            )

            # Get recommendations
            screening_results = screener.recommend_variables(
                near_zero_threshold=kwargs.get('near_zero_threshold', 0.01),
                collinearity_threshold=kwargs.get('collinearity_threshold', 0.85),
                vif_threshold=kwargs.get('vif_threshold', 5.0),
                force_include=kwargs.get('force_include', [])
            )

            pipeline_results['variable_screening'] = screener.get_results()

            # Filter data to keep only recommended variables if requested
            if kwargs.get('apply_screening_recommendations', True):
                recommended_vars = screening_results['recommended_variable_list']
                processed_data = processed_data[recommended_vars]
                logger.info(f"Reduced from {len(self.data.columns)} to {len(recommended_vars)} variables")

        # Step 3: Dimensionality reduction
        if reduce_dimensions:
            logger.info("Step 3: Performing dimensionality reduction...")
            reducer = DimensionalityReducer(processed_data)

            # Dimensionality reduction method
            dim_reduction_method = kwargs.get('dim_reduction_method', 'pca')

            if dim_reduction_method == 'pca':
                # Perform PCA
                reducer.perform_pca(
                    n_components=kwargs.get('n_components', None),
                    standardize=kwargs.get('standardize', True)
                )
            elif dim_reduction_method == 'famd':
                # Perform FAMD
                reducer.perform_famd(
                    n_components=kwargs.get('n_components', 10)
                )

            # Get optimal number of components
            optimal_components = reducer.get_optimal_components(
                method=dim_reduction_method,
                variance_threshold=kwargs.get('variance_threshold', 0.75)
            )

            pipeline_results['dimensionality_reduction'] = reducer.get_results()

            # Transform data if requested
            if kwargs.get('apply_dim_reduction', False):
                transformed_data = reducer.transform_data(
                    method=dim_reduction_method,
                    n_components=optimal_components
                )

                # Add original ID column if it exists
                if 'ID' in self.data.columns:
                    transformed_data['ID'] = self.data['ID'].values

                processed_data = transformed_data
                logger.info(f"Reduced to {optimal_components} principal components")

        # Step 4: Data quality enhancement
        if enhance_quality:
            logger.info("Step 4: Enhancing data quality...")
            enhancer = DataQualityEnhancer(processed_data)

            # Detect outliers
            enhancer.detect_outliers(
                method=kwargs.get('outlier_method', 'iqr'),
                threshold=kwargs.get('outlier_threshold', 1.5)
            )

            # Analyze distributions
            enhancer.analyze_distributions()

            # Get transformation recommendations
            enhancer.recommend_transformations()

            pipeline_results['data_quality'] = enhancer.get_results()

            # Apply transformations if requested
            if kwargs.get('apply_transformations', False):
                processed_data = enhancer.apply_transformations()
                logger.info("Applied recommended transformations")

            # Standardize variables if requested
            if kwargs.get('standardize_variables', False):
                processed_data = enhancer.standardize_variables(
                    method=kwargs.get('standardization_method', 'zscore')
                )
                logger.info(f"Standardized variables using {kwargs.get('standardization_method', 'zscore')} method")

        return processed_data, pipeline_results

    def get_pipeline_report(self) -> Dict:
        """
        Generate comprehensive report on pipeline steps and results.

        Returns:
            Dictionary containing pipeline results summary
        """
        # This would be populated after running the prepare_data_pipeline method
        # For now, return a placeholder
        return {
            "status": "Pipeline report not available. Run prepare_data_pipeline() first."
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create the data loader
    loader = DataLoader()

    # Load the data
    data, data_dict = loader.load_data()

    # Analyze missing data
    missing_report = loader.get_missing_data_report()
    print("\nMissing Data Report:")
    print(missing_report[missing_report['Missing Values'] > 0])

    # Impute missing values
    imputed_data = loader.impute_missing_values(method='knn')

    # Get treatment groups
    treatment_groups = loader.get_treatment_groups()

    # Print group statistics
    for group_name, group_data in treatment_groups.items():
        print(f"\n{group_name} (n={len(group_data)}):")

        # Check if WOMAC columns exist and calculate pain change
        if 'WOMAC.Pain.0' in group_data.columns and 'WOMAC.Pain.10' in group_data.columns:
            baseline = group_data['WOMAC.Pain.0'].mean()
            followup = group_data['WOMAC.Pain.10'].mean()
            change = followup - baseline
            print(f"  WOMAC Pain Change: {change:.2f} (Baseline: {baseline:.2f}, 10-day: {followup:.2f})")

    # Save the processed data
    loader.save_processed_data(imputed_data, "koa_processed_data.csv", index=False)
