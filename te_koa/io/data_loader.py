"""
Knee Osteoarthritis Data Loader.

This module provides functionality for loading and preprocessing the Knee Osteoarthritis dataset.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataLoader:
    """A specialized data loader for the Knee Osteoarthritis dataset."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the KOADataLoader with a data directory.

        Args:
            data_dir: Path to the data directory. If None, uses the current directory.
        """
        self.data_dir = Path( data_dir or "/Users/artinmajdi/Documents/GitHubs/RAP/te_koa/dataset/te_koa_R01_only_RCT_data.xlsx" )

        # Keep track of loaded data
        self.data = None
        self.data_dict = None
        self.missing_data_report = None

    def load_data( self, sheet_name: str = "Sheet1", dictionary_sheet: str = "dictionary", **kwargs ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the Knee Osteoarthritis dataset and data dictionary.

        Args:
            filename: Name of the Excel file (relative to data_dir)
            sheet_name: Name of the sheet containing the actual data
            dictionary_sheet: Name of the sheet containing the data dictionary
            **kwargs: Additional arguments to pass to pd.read_excel

        Returns:
            Tuple containing (data DataFrame, data dictionary DataFrame)

        Raises:
            FileNotFoundError: If the file doesn't exist
        """

        logger.info(f"Loading Knee Osteoarthritis data from {self.data_dir}")

        try:
            # Load both data and dictionary sheets
            data      = pd.read_excel(self.data_dir, sheet_name=sheet_name, **kwargs)
            data_dict = pd.read_excel(self.data_dir, sheet_name=dictionary_sheet, **kwargs)

            logger.info(f"Successfully loaded {len(data)} rows from {sheet_name} sheet")
            logger.info(f"Successfully loaded {len(data_dict)} rows from {dictionary_sheet} sheet")

            # Store the loaded data
            self.data = data
            self.data_dict = data_dict

            # Generate missing data report
            self._analyze_missing_data()

            return data, data_dict

        except FileNotFoundError:
            logger.error(f"File not found: {self.data_dir}")
            raise
        except Exception as e:
            logger.error(f"Error loading Excel file {self.data_dir.name}: {e}")
            raise

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

    def impute_missing_values( self, method: str = 'knn', knn_neighbors: int = 5, cols_to_exclude: List[str] = None ) -> pd.DataFrame:
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
            imputed_df = pd.DataFrame( scaler.inverse_transform(imputed_scaled_data), columns=cols_to_impute, index=imputed_data.index )

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
        Split the data into treatment groups based on tDCS and medication variables.

        Returns:
            Dictionary of DataFrames for each treatment group
        """
        if self.data is None:
            logger.warning("No data loaded. Call load_data() first.")
            return None

        # Identify treatment columns (assuming they're named 'tDCS' and 'Medication'
        # or can be identified from the data dictionary)
        tdcs_col = None
        med_col = None

        # Try to find the treatment columns
        for col in self.data.columns:
            if 'tdcs' in col.lower():
                tdcs_col = col
            elif 'med' in col.lower():
                med_col = col

        if tdcs_col is None or med_col is None:
            logger.warning("Could not identify treatment columns. Assuming 'tDCS' and 'Medication'.")
            tdcs_col = 'tDCS'
            med_col = 'Medication'

        # Split into treatment groups
        try:
            groups = {
                'Control': self.data[(self.data[tdcs_col] == 0) & (self.data[med_col] == 0)],
                'tDCS Only': self.data[(self.data[tdcs_col] == 1) & (self.data[med_col] == 0)],
                'Medication Only': self.data[(self.data[tdcs_col] == 0) & (self.data[med_col] == 1)],
                'tDCS + Medication': self.data[(self.data[tdcs_col] == 1) & (self.data[med_col] == 1)]
            }

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
        # The structure of the data dictionary may vary, so we need to be flexible
        try:
            # Identify which column contains variable names and which contains descriptions
            # This is a simplistic approach and may need to be adapted
            if 'Variable' in self.data_dict.columns and 'Description' in self.data_dict.columns:
                var_col = 'Variable'
                desc_col = 'Description'
            else:
                # Assume the first column is for variable names and the second for descriptions
                var_col = self.data_dict.columns[0]
                desc_col = self.data_dict.columns[1]

            # Look for exact match
            matches = self.data_dict[self.data_dict[var_col] == variable_name]

            if len(matches) == 0:
                # Try case-insensitive match
                matches = self.data_dict[self.data_dict[var_col].str.lower() == variable_name.lower()]

            if len(matches) == 0:
                # Try substring match
                matches = self.data_dict[self.data_dict[var_col].str.contains(variable_name, case=False)]

            if len(matches) > 0:
                return matches.iloc[0][desc_col]
            else:
                logger.warning(f"Variable '{variable_name}' not found in data dictionary.")
                return None

        except Exception as e:
            logger.error(f"Error getting variable description: {e}")
            return None

    def save_processed_data(self, df: pd.DataFrame, filename: str, **kwargs) -> None:
        """
        Save processed data to a CSV file.

        Args:
            df: DataFrame to save
            filename: Name of the CSV file (relative to data_dir)
            **kwargs: Additional arguments to pass to df.to_csv
        """
        file_path = self.data_dir / filename
        logger.info(f"Saving {len(df)} rows to CSV file {file_path}")

        # Ensure directory exists
        os.makedirs(file_path.parent, exist_ok=True)

        try:
            df.to_csv(file_path, **kwargs)
            logger.info(f"Successfully saved data to {filename}")
        except Exception as e:
            logger.error(f"Error saving CSV file {filename}: {e}")
            raise


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
        if 'WOMAC_pain_baseline' in group_data.columns and 'WOMAC_pain_6m' in group_data.columns:
            baseline = group_data['WOMAC_pain_baseline'].mean()
            followup = group_data['WOMAC_pain_6m'].mean()
            change = followup - baseline
            print(f"  WOMAC Pain Change: {change:.2f} (Baseline: {baseline:.2f}, 6-month: {followup:.2f})")

    # Save the processed data
    loader.save_processed_data(imputed_data, "koa_processed_data.csv", index=False)
