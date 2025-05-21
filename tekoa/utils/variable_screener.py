"""
Variable Screening Module for TE-KOA.

This module provides functionality for variable screening including:
- Near-zero variance detection
- Correlation/collinearity analysis
- Variance Inflation Factor (VIF) calculation
- Clinical vs. statistical variable importance evaluation
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class VariableScreener:
    """Class for screening variables in the TE-KOA dataset."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the VariableScreener.

        Args:
            data: DataFrame containing the dataset to screen
        """
        self.data = data.copy()
        self.near_zero_vars = None
        self.high_corr_pairs = None
        self.vif_values = None
        self.recommended_vars = None

        # Track numeric and categorical variables
        self.numeric_vars = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_vars = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Initialize results storage
        self.results = {
            'near_zero_variance': {},
            'correlation': {},
            'vif': {},
            'recommended_variables': []
        }

    def identify_near_zero_variance(self, threshold: float = 0.01, save_results: bool = True) -> List[str]:
        """
        Identify variables with near-zero variance.

        Args:
            threshold: Threshold for determining near-zero variance.
                       Variables with unique values < threshold * n_samples are considered near-zero.
            save_results: Whether to save results in the results dictionary

        Returns:
            List of variable names with near-zero variance
        """
        near_zero_vars = []
        n_samples = len(self.data)
        min_unique_values = max(2, int(n_samples * threshold))

        # Check each variable
        var_stats = {}
        for col in self.numeric_vars:
            n_unique = self.data[col].nunique()
            most_common_freq = self.data[col].value_counts().iloc[0] / n_samples if n_unique > 0 else 1.0

            var_stats[col] = {
                'n_unique': n_unique,
                'most_common_freq': most_common_freq,
                'is_near_zero': n_unique < min_unique_values
            }

            if n_unique < min_unique_values:
                near_zero_vars.append(col)

        # Save results if requested
        if save_results:
            self.results['near_zero_variance'] = var_stats
            self.near_zero_vars = near_zero_vars

        return near_zero_vars

    def analyze_collinearity(self, threshold: float = 0.85, save_results: bool = True) -> List[Tuple[str, str, float]]:
        """
        Analyze collinearity between numeric variables.

        Args:
            threshold: Correlation threshold for identifying highly collinear pairs
            save_results: Whether to save results in the results dictionary

        Returns:
            List of tuples containing (var1, var2, correlation) for highly correlated pairs
        """
        # Calculate correlation matrix for numeric variables
        corr_matrix = self.data[self.numeric_vars].corr().abs()

        # Find highly correlated pairs (upper triangle only to avoid duplicates)
        high_corr_pairs = []
        for i in range(len(self.numeric_vars)):
            for j in range(i+1, len(self.numeric_vars)):
                var1 = self.numeric_vars[i]
                var2 = self.numeric_vars[j]
                corr = corr_matrix.iloc[i, j]

                if corr >= threshold:
                    high_corr_pairs.append((var1, var2, corr))

        # Save results if requested
        if save_results:
            self.results['correlation'] = {
                'matrix': corr_matrix,
                'high_correlation_pairs': high_corr_pairs
            }
            self.high_corr_pairs = high_corr_pairs

        return high_corr_pairs

    def calculate_vif(self, max_vif: float = 5.0, save_results: bool = True) -> Dict[str, float]:
        """
        Calculate Variance Inflation Factor for numeric variables.

        Args:
            max_vif: Maximum VIF value for variables to keep
            save_results: Whether to save results in the results dictionary

        Returns:
            Dictionary with variable names as keys and VIF values as values
        """
        # Select numeric variables without missing values
        # VIF calculation requires complete cases
        complete_numeric_data = self.data[self.numeric_vars].dropna()

        if complete_numeric_data.empty:
            logger.warning("No complete cases found for VIF calculation.")
            return {}

        # Need at least 2 variables for VIF
        if len(complete_numeric_data.columns) < 2:
            logger.warning("At least 2 variables needed for VIF calculation.")
            return {}

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(complete_numeric_data)

        # Calculate VIF for each variable
        vif_values = {}

        try:
            # Add a constant column (intercept) for the linear regression model
            X = np.column_stack([np.ones(scaled_data.shape[0]), scaled_data])

            # Calculate VIF for each variable (skip the intercept)
            for i in range(1, X.shape[1]):
                vif = variance_inflation_factor(X, i)
                vif_values[complete_numeric_data.columns[i-1]] = vif
        except Exception as e:
            logger.error(f"Error calculating VIF: {e}")
            return {}

        # Save results if requested
        if save_results:
            self.results['vif'] = {
                'values': vif_values,
                'high_vif_variables': [var for var, vif in vif_values.items() if vif > max_vif]
            }
            self.vif_values = vif_values

        return vif_values

    def recommend_variables(self,
                           near_zero_threshold: float = 0.01,
                           collinearity_threshold: float = 0.85,
                           vif_threshold: float = 5.0,
                           force_include: List[str] = None) -> Dict:
        """
        Generate recommended variable list based on screening criteria.

        Args:
            near_zero_threshold: Threshold for near-zero variance detection
            collinearity_threshold: Threshold for collinearity detection
            vif_threshold: Threshold for VIF filtering
            force_include: List of variables to force include regardless of screening criteria

        Returns:
            Dictionary with screening results and recommended variables
        """
        # Ensure we have screening results
        if self.near_zero_vars is None:
            self.identify_near_zero_variance(threshold=near_zero_threshold)

        if self.high_corr_pairs is None:
            self.analyze_collinearity(threshold=collinearity_threshold)

        if self.vif_values is None:
            self.calculate_vif(max_vif=vif_threshold)

        # Initialize force_include if None
        if force_include is None:
            force_include = []

        # Start with all variables
        all_vars = set(self.data.columns)
        vars_to_exclude = set()

        # Exclude near-zero variance variables unless forced to include
        for var in self.near_zero_vars:
            if var not in force_include:
                vars_to_exclude.add(var)

        # Handle highly correlated pairs
        # For each pair, exclude the second variable unless it's in force_include
        for var1, var2, _ in self.high_corr_pairs:
            if var2 not in force_include:
                vars_to_exclude.add(var2)

        # Exclude high VIF variables unless forced to include
        if self.vif_values:
            for var, vif in self.vif_values.items():
                if vif > vif_threshold and var not in force_include:
                    vars_to_exclude.add(var)

        # Generate final recommended variables list
        recommended_vars = sorted(list(all_vars - vars_to_exclude))

        # Save results
        self.recommended_vars = recommended_vars
        self.results['recommended_variables'] = recommended_vars

        # Prepare summary information
        summary = {
            'total_variables': len(all_vars),
            'near_zero_variables': len(self.near_zero_vars),
            'highly_correlated_pairs': len(self.high_corr_pairs),
            'high_vif_variables': len(self.results['vif'].get('high_vif_variables', [])) if self.vif_values else 0,
            'force_included_variables': len(force_include),
            'recommended_variables': len(recommended_vars),
            'recommended_variable_list': recommended_vars
        }

        self.results['summary'] = summary
        return summary

    def get_correlation_matrix(self, variables: List[str] = None) -> pd.DataFrame:
        """
        Get correlation matrix for selected variables.

        Args:
            variables: List of variables to include in correlation matrix.
                      If None, uses all numeric variables.

        Returns:
            Correlation matrix as DataFrame
        """
        if variables is None:
            variables = self.numeric_vars
        else:
            # Keep only numeric variables from the provided list
            variables = [var for var in variables if var in self.numeric_vars]

        return self.data[variables].corr()

    def plot_correlation_heatmap(self, variables: List[str] = None, figsize: Tuple[int, int] = (12, 10)):
        """
        Plot correlation heatmap for selected variables.

        Args:
            variables: List of variables to include in heatmap.
                      If None, uses all numeric variables.
            figsize: Figure size as (width, height) tuple

        Returns:
            Matplotlib figure and axes
        """
        corr_matrix = self.get_correlation_matrix(variables)

        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create the heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax)

        # Customize the plot
        ax.set_title('Correlation Matrix')
        plt.tight_layout()

        return fig, ax

    def plot_vif_values(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot VIF values for all variables.

        Args:
            figsize: Figure size as (width, height) tuple

        Returns:
            Matplotlib figure and axes
        """
        if not self.vif_values:
            logger.warning("VIF values not calculated. Call calculate_vif() first.")
            return None, None

        # Create DataFrame for plotting
        vif_df = pd.DataFrame({
            'Variable': list(self.vif_values.keys()),
            'VIF': list(self.vif_values.values())
        }).sort_values('VIF', ascending=False)

        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create the bar plot
        sns.barplot(x='VIF', y='Variable', data=vif_df, ax=ax)

        # Add a vertical line at the threshold
        ax.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='VIF Threshold (5)')

        # Customize the plot
        ax.set_title('Variance Inflation Factor (VIF) Values')
        ax.set_xlabel('VIF')
        ax.set_ylabel('Variable')
        ax.legend()

        plt.tight_layout()

        return fig, ax

    def get_results(self) -> Dict:
        """
        Get all screening results.

        Returns:
            Dictionary containing all screening results
        """
        return self.results
