"""
Data Quality Enhancement Module for TE-KOA.

This module provides functionality for data quality assessment and enhancement including:
- Outlier detection using various methods
- Distribution analysis
- Variable transformation recommendations
- Data standardization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.impute import KNNImputer

from tekoa.configuration import logger


class DataQualityEnhancer:
    """Class for enhancing data quality in the TE-KOA dataset."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataQualityEnhancer.

        Args:
            data: DataFrame containing the dataset to analyze
        """
        self.data = data.copy()

        # Track numeric and categorical variables
        self.numeric_vars = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_vars = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Store results
        self.outliers = None
        self.distributions = None
        self.transformations = None
        self.standardized_data = None

        # Initialize results storage
        self.results = {
            'outliers': {},
            'distributions': {},
            'transformations': {},
            'standardization': {}
        }

    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5, variables: List[str] = None) -> Dict:
        """
        Detect outliers using various methods.

        Args:
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            variables: List of variables to check for outliers. If None, uses all numeric variables.

        Returns:
            Dictionary with outlier detection results
        """
        # Use all numeric variables if none specified
        if variables is None:
            variables = self.numeric_vars
        else:
            # Keep only numeric variables from the provided list
            variables = [var for var in variables if var in self.numeric_vars]

        if len(variables) == 0:
            logger.warning("No numeric variables available for outlier detection.")
            return {}

        # Initialize results dictionary
        outlier_results = {var: {} for var in variables}

        # Detect outliers for each variable
        for var in variables:
            # Skip variables with missing values
            if self.data[var].isnull().any():
                values = self.data[var].dropna().values
            else:
                values = self.data[var].values

            if len(values) == 0:
                continue

            if method == 'iqr':
                # IQR method
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1

                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

                outliers = self.data[
                    (self.data[var] < lower_bound) |
                    (self.data[var] > upper_bound)
                ].index.tolist()

                outlier_results[var] = {
                    'method': 'iqr',
                    'threshold': threshold,
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'n_outliers': len(outliers),
                    'outlier_indices': outliers,
                    'outlier_percentage': len(outliers) / len(self.data) * 100
                }

            elif method == 'zscore':
                # Z-score method
                mean = np.mean(values)
                std = np.std(values)

                zscores = np.abs((values - mean) / std)
                outlier_mask = zscores > threshold

                outliers = self.data.index[
                    self.data[var].notnull() &
                    (np.abs((self.data[var] - mean) / std) > threshold)
                ].tolist()

                outlier_results[var] = {
                    'method': 'zscore',
                    'threshold': threshold,
                    'mean': mean,
                    'std': std,
                    'n_outliers': len(outliers),
                    'outlier_indices': outliers,
                    'outlier_percentage': len(outliers) / len(self.data) * 100
                }

            elif method == 'modified_zscore':
                # Modified Z-score method (uses median instead of mean)
                median = np.median(values)
                mad = np.median(np.abs(values - median))  # Median Absolute Deviation

                # Factor 0.6745 to make MAD consistent with standard deviation for normal distributions
                modified_zscores = 0.6745 * np.abs(values - median) / mad if mad > 0 else np.zeros_like(values)

                outliers = self.data.index[
                    self.data[var].notnull() &
                    (0.6745 * np.abs(self.data[var] - median) / mad > threshold if mad > 0 else False)
                ].tolist()

                outlier_results[var] = {
                    'method': 'modified_zscore',
                    'threshold': threshold,
                    'median': median,
                    'mad': mad,
                    'n_outliers': len(outliers),
                    'outlier_indices': outliers,
                    'outlier_percentage': len(outliers) / len(self.data) * 100
                }

            else:
                logger.warning(f"Unknown outlier detection method: {method}")
                return {}

        # Create a summary of detected outliers
        outlier_summary = {
            'method': method,
            'threshold': threshold,
            'variables_checked': len(variables),
            'variables_with_outliers': sum(1 for var in outlier_results if outlier_results[var].get('n_outliers', 0) > 0),
            'total_outliers_detected': sum(outlier_results[var].get('n_outliers', 0) for var in outlier_results),
            'variable_outlier_counts': {var: outlier_results[var].get('n_outliers', 0) for var in outlier_results}
        }

        # Store results
        self.outliers = {
            'method': method,
            'threshold': threshold,
            'details': outlier_results,
            'summary': outlier_summary
        }

        self.results['outliers'] = self.outliers

        return self.outliers

    def analyze_distributions(self, variables: List[str] = None) -> Dict:
        """
        Analyze distributions of variables (normality, skewness, etc.)

        Args:
            variables: List of variables to analyze. If None, uses all numeric variables.

        Returns:
            Dictionary with distribution analysis results
        """
        # Use all numeric variables if none specified
        if variables is None:
            variables = self.numeric_vars
        else:
            # Keep only numeric variables from the provided list
            variables = [var for var in variables if var in self.numeric_vars]

        if len(variables) == 0:
            logger.warning("No numeric variables available for distribution analysis.")
            return {}

        # Initialize results dictionary
        distribution_results = {}

        # Analyze distribution for each variable
        for var in variables:
            # Skip variables with missing values
            if self.data[var].isnull().any():
                values = self.data[var].dropna().values
            else:
                values = self.data[var].values

            if len(values) == 0:
                continue

            # Basic statistics
            mean = np.mean(values)
            median = np.median(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)

            # Shape statistics
            skewness = stats.skew(values)
            kurtosis = stats.kurtosis(values)

            # Normality test (Shapiro-Wilk)
            # For larger samples, use a subset to avoid excessive sensitivity
            sample_size = min(len(values), 5000)
            sample = np.random.choice(values, size=sample_size, replace=False) if len(values) > sample_size else values

            try:
                shapiro_stat, shapiro_p = stats.shapiro(sample)
            except Exception:
                shapiro_stat, shapiro_p = np.nan, np.nan

            # Determine if distribution is approximately normal
            is_normal = shapiro_p > 0.05

            # Store results
            distribution_results[var] = {
                'mean': mean,
                'median': median,
                'std': std,
                'min': min_val,
                'max': max_val,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'is_normal': is_normal,
                'mean_median_difference': abs(mean - median),
                'n_samples': len(values)
            }

        # Create a summary of distribution analysis
        distribution_summary = {
            'variables_analyzed': len(distribution_results),
            'normal_variables': sum(1 for var in distribution_results if distribution_results[var]['is_normal']),
            'skewed_variables': sum(1 for var in distribution_results if abs(distribution_results[var]['skewness']) > 1),
            'highly_skewed_variables': sum(1 for var in distribution_results if abs(distribution_results[var]['skewness']) > 2),
            'variable_skewness': {var: distribution_results[var]['skewness'] for var in distribution_results}
        }

        # Store results
        self.distributions = {
            'details': distribution_results,
            'summary': distribution_summary
        }

        self.results['distributions'] = self.distributions

        return self.distributions

    def recommend_transformations(self) -> Dict:
        """
        Recommend transformations for skewed or non-normal variables.

        Returns:
            Dictionary with transformation recommendations
        """
        # Analyze distributions if not already done
        if self.distributions is None:
            self.analyze_distributions()

        if not self.distributions:
            logger.warning("No distribution analysis available.")
            return {}

        # Initialize results dictionary
        transformation_recommendations = {}

        # Get distribution details
        distribution_details = self.distributions.get('details', {})

        # Recommend transformations for each variable
        for var, stats in distribution_details.items():
            # Skip if distribution is already normal
            if stats.get('is_normal', False):
                transformation_recommendations[var] = {
                    'recommendation': 'none',
                    'reason': 'Distribution is already approximately normal'
                }
                continue

            # Get skewness
            skewness = stats.get('skewness', 0)

            # Recommend transformation based on skewness
            if abs(skewness) <= 0.5:
                recommendation = 'none'
                reason = 'Distribution is approximately symmetric'

            elif skewness > 2:
                # Highly positively skewed
                recommendation = 'log'
                reason = 'Distribution is highly positively skewed'

            elif skewness > 1:
                # Moderately positively skewed
                recommendation = 'sqrt'
                reason = 'Distribution is moderately positively skewed'

            elif skewness < -2:
                # Highly negatively skewed
                recommendation = 'square'
                reason = 'Distribution is highly negatively skewed'

            elif skewness < -1:
                # Moderately negatively skewed
                recommendation = 'square'
                reason = 'Distribution is moderately negatively skewed'

            else:
                # Slightly skewed
                recommendation = 'yeo-johnson'
                reason = 'Distribution is slightly skewed, power transformation recommended'

            # Add recommendation
            transformation_recommendations[var] = {
                'recommendation': recommendation,
                'reason': reason,
                'skewness': skewness,
                'is_normal': stats.get('is_normal', False)
            }

        # Create a summary of transformation recommendations
        recommendation_summary = {
            'variables_analyzed': len(transformation_recommendations),
            'variables_needing_transformation': sum(1 for var in transformation_recommendations
                                                  if transformation_recommendations[var]['recommendation'] != 'none'),
            'recommendation_counts': {
                'none': sum(1 for var in transformation_recommendations
                           if transformation_recommendations[var]['recommendation'] == 'none'),
                'log': sum(1 for var in transformation_recommendations
                          if transformation_recommendations[var]['recommendation'] == 'log'),
                'sqrt': sum(1 for var in transformation_recommendations
                           if transformation_recommendations[var]['recommendation'] == 'sqrt'),
                'square': sum(1 for var in transformation_recommendations
                             if transformation_recommendations[var]['recommendation'] == 'square'),
                'yeo-johnson': sum(1 for var in transformation_recommendations
                                  if transformation_recommendations[var]['recommendation'] == 'yeo-johnson')
            }
        }

        # Store results
        self.transformations = {
            'recommendations': transformation_recommendations,
            'summary': recommendation_summary
        }

        self.results['transformations'] = self.transformations

        return self.transformations

    def apply_transformations(self, transformations: Dict = None) -> pd.DataFrame:
        """
        Apply recommended or specified transformations.

        Args:
            transformations: Dictionary mapping variable names to transformation types.
                           If None, uses recommended transformations.

        Returns:
            DataFrame with transformed variables
        """
        # Get recommendations if not provided
        if transformations is None:
            if self.transformations is None:
                self.recommend_transformations()

            if not self.transformations:
                logger.warning("No transformation recommendations available.")
                return self.data.copy()

            # Extract recommendations
            recommendations = self.transformations.get('recommendations', {})
            transformations = {var: rec['recommendation'] for var, rec in recommendations.items()
                             if rec['recommendation'] != 'none'}

        # Initialize transformed dataframe
        transformed_data = self.data.copy()

        # Track applied transformations
        applied_transformations = {}

        # Apply transformations for each variable
        for var, transform_type in transformations.items():
            # Skip if variable not in DataFrame or not numeric
            if var not in self.data.columns or var not in self.numeric_vars:
                continue

            # Get variable values
            values = self.data[var].copy()

            # Skip if all values are missing
            if values.isnull().all():
                continue

            # Apply transformation
            try:
                if transform_type == 'log':
                    # Ensure all values are positive
                    min_val = values.min()
                    offset = abs(min_val) + 1 if min_val <= 0 else 0
                    transformed_data[var] = np.log(values + offset)

                    applied_transformations[var] = {
                        'type': 'log',
                        'offset': offset,
                        'formula': f'log({var} + {offset})'
                    }

                elif transform_type == 'sqrt':
                    # Ensure all values are positive
                    min_val = values.min()
                    offset = abs(min_val) + 0.01 if min_val < 0 else 0
                    transformed_data[var] = np.sqrt(values + offset)

                    applied_transformations[var] = {
                        'type': 'sqrt',
                        'offset': offset,
                        'formula': f'sqrt({var} + {offset})'
                    }

                elif transform_type == 'square':
                    transformed_data[var] = values ** 2

                    applied_transformations[var] = {
                        'type': 'square',
                        'formula': f'{var}^2'
                    }

                elif transform_type == 'yeo-johnson':
                    # Use Yeo-Johnson transformer which handles both positive and negative values
                    transformer = PowerTransformer(method='yeo-johnson')

                    # Transform non-missing values
                    non_missing_mask = ~values.isnull()
                    non_missing_values = values[non_missing_mask].values.reshape(-1, 1)

                    transformed_values = transformer.fit_transform(non_missing_values).flatten()

                    # Update only non-missing values
                    transformed_series = values.copy()
                    transformed_series[non_missing_mask] = transformed_values
                    transformed_data[var] = transformed_series

                    applied_transformations[var] = {
                        'type': 'yeo-johnson',
                        'lambda': transformer.lambdas_[0],
                        'formula': f'yeo-johnson({var}, lambda={transformer.lambdas_[0]:.4f})'
                    }

                else:
                    logger.warning(f"Unknown transformation type: {transform_type}")
                    continue

            except Exception as e:
                logger.error(f"Error applying {transform_type} transformation to {var}: {e}")
                continue

        # Create a summary of applied transformations
        transformation_summary = {
            'variables_transformed': len(applied_transformations),
            'transformation_counts': {
                'log': sum(1 for var in applied_transformations
                          if applied_transformations[var]['type'] == 'log'),
                'sqrt': sum(1 for var in applied_transformations
                           if applied_transformations[var]['type'] == 'sqrt'),
                'square': sum(1 for var in applied_transformations
                             if applied_transformations[var]['type'] == 'square'),
                'yeo-johnson': sum(1 for var in applied_transformations
                                  if applied_transformations[var]['type'] == 'yeo-johnson')
            }
        }

        # Store results
        self.results['transformations']['applied'] = {
            'details': applied_transformations,
            'summary': transformation_summary
        }

        return transformed_data

    def standardize_variables(self, variables: List[str] = None, method: str = 'zscore') -> pd.DataFrame:
        """
        Standardize variables (mean=0, std=1) or using other methods.

        Args:
            variables: List of variables to standardize. If None, uses all numeric variables.
            method: Standardization method ('zscore', 'robust', 'minmax')

        Returns:
            DataFrame with standardized variables
        """
        # Use all numeric variables if none specified
        if variables is None:
            variables = self.numeric_vars
        else:
            # Keep only numeric variables from the provided list
            variables = [var for var in variables if var in self.numeric_vars]

        if len(variables) == 0:
            logger.warning("No numeric variables available for standardization.")
            return self.data.copy()

        # Initialize standardized dataframe
        standardized_data = self.data.copy()

        # Track standardization details
        standardization_details = {}

        # Apply standardization based on method
        if method == 'zscore':
            # Z-score standardization (mean=0, std=1)
            scaler = StandardScaler()

            for var in variables:
                # Skip if all values are missing
                if standardized_data[var].isnull().all():
                    continue

                # Get non-missing values
                non_missing_mask = ~standardized_data[var].isnull()
                non_missing_values = standardized_data[var][non_missing_mask].values.reshape(-1, 1)

                # Fit scaler and transform non-missing values
                scaler.fit(non_missing_values)
                transformed_values = scaler.transform(non_missing_values).flatten()

                # Update only non-missing values
                standardized_series = standardized_data[var].copy()
                standardized_series[non_missing_mask] = transformed_values
                standardized_data[var] = standardized_series

                # Store standardization details
                standardization_details[var] = {
                    'method': 'zscore',
                    'mean': scaler.mean_[0],
                    'std': scaler.scale_[0],
                    'formula': f'({var} - {scaler.mean_[0]:.4f}) / {scaler.scale_[0]:.4f}'
                }

        elif method == 'robust':
            # Robust standardization (median=0, IQR=1)
            scaler = RobustScaler()

            for var in variables:
                # Skip if all values are missing
                if standardized_data[var].isnull().all():
                    continue

                # Get non-missing values
                non_missing_mask = ~standardized_data[var].isnull()
                non_missing_values = standardized_data[var][non_missing_mask].values.reshape(-1, 1)

                # Fit scaler and transform non-missing values
                scaler.fit(non_missing_values)
                transformed_values = scaler.transform(non_missing_values).flatten()

                # Update only non-missing values
                standardized_series = standardized_data[var].copy()
                standardized_series[non_missing_mask] = transformed_values
                standardized_data[var] = standardized_series

                # Store standardization details
                standardization_details[var] = {
                    'method': 'robust',
                    'center': scaler.center_[0],
                    'scale': scaler.scale_[0],
                    'formula': f'({var} - {scaler.center_[0]:.4f}) / {scaler.scale_[0]:.4f}'
                }

        elif method == 'minmax':
            # Min-max scaling to [0, 1]
            for var in variables:
                # Skip if all values are missing
                if standardized_data[var].isnull().all():
                    continue

                # Get non-missing values
                non_missing_mask = ~standardized_data[var].isnull()
                values = standardized_data[var][non_missing_mask]

                # Calculate min and max
                min_val = values.min()
                max_val = values.max()

                # Skip if min equals max
                if min_val == max_val:
                    continue

                # Apply min-max scaling
                standardized_series = standardized_data[var].copy()
                standardized_series[non_missing_mask] = (values - min_val) / (max_val - min_val)
                standardized_data[var] = standardized_series

                # Store standardization details
                standardization_details[var] = {
                    'method': 'minmax',
                    'min': min_val,
                    'max': max_val,
                    'formula': f'({var} - {min_val:.4f}) / ({max_val:.4f} - {min_val:.4f})'
                }

        else:
            logger.warning(f"Unknown standardization method: {method}")
            return self.data.copy()

        # Create a summary of standardization
        standardization_summary = {
            'method': method,
            'variables_standardized': len(standardization_details)
        }

        # Store results
        self.standardized_data = standardized_data
        self.results['standardization'] = {
            'method': method,
            'details': standardization_details,
            'summary': standardization_summary
        }

        return standardized_data

    def plot_distribution(self, variable: str, original: bool = True, transformed: bool = True,
                        transformation_type: str = None, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot distribution of a variable before and after transformation.

        Args:
            variable: Variable to plot
            original: Whether to plot original distribution
            transformed: Whether to plot transformed distribution
            transformation_type: Type of transformation to apply ('log', 'sqrt', 'square', 'yeo-johnson')
            figsize: Figure size as (width, height) tuple

        Returns:
            Matplotlib figure and axes
        """
        # Check if variable exists and is numeric
        if variable not in self.data.columns:
            logger.warning(f"Variable {variable} not found in dataset.")
            return None, None

        if variable not in self.numeric_vars:
            logger.warning(f"Variable {variable} is not numeric.")
            return None, None

        # Get original values (non-missing)
        original_values = self.data[variable].dropna()

        if len(original_values) == 0:
            logger.warning(f"Variable {variable} has no non-missing values.")
            return None, None

        # Create figure with one or two subplots
        if original and transformed:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
            ax2 = None

        # Plot original distribution
        if original:
            sns.histplot(original_values, kde=True, ax=ax1)
            ax1.set_title(f'Original Distribution - {variable}')
            ax1.set_xlabel(variable)
            ax1.set_ylabel('Frequency')

            # Add skewness
            skewness = stats.skew(original_values)
            ax1.text(0.05, 0.95, f'Skewness: {skewness:.2f}', transform=ax1.transAxes,
                    fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Plot transformed distribution if requested
        if transformed and ax2 is not None:
            # Get transformation type if not specified
            if transformation_type is None:
                # Use recommended transformation if available
                if self.transformations is not None:
                    recommendations = self.transformations.get('recommendations', {})
                    if variable in recommendations:
                        transformation_type = recommendations[variable]['recommendation']
                    else:
                        transformation_type = 'yeo-johnson'  # Default
                else:
                    transformation_type = 'yeo-johnson'  # Default

            # Apply transformation
            transformed_values = None

            if transformation_type == 'log':
                # Ensure all values are positive
                min_val = original_values.min()
                offset = abs(min_val) + 1 if min_val <= 0 else 0
                transformed_values = np.log(original_values + offset)
                transformation_name = f'Log({variable} + {offset:.2f})'

            elif transformation_type == 'sqrt':
                # Ensure all values are positive
                min_val = original_values.min()
                offset = abs(min_val) + 0.01 if min_val < 0 else 0
                transformed_values = np.sqrt(original_values + offset)
                transformation_name = f'Sqrt({variable} + {offset:.2f})'

            elif transformation_type == 'square':
                transformed_values = original_values ** 2
                transformation_name = f'{variable}²'

            elif transformation_type == 'yeo-johnson':
                # Use Yeo-Johnson transformer
                transformer = PowerTransformer(method='yeo-johnson')
                transformed_values = transformer.fit_transform(original_values.values.reshape(-1, 1)).flatten()
                transformation_name = f'Yeo-Johnson({variable})'

            else:
                logger.warning(f"Unknown transformation type: {transformation_type}")
                transformed_values = original_values
                transformation_name = variable

            # Plot transformed distribution
            if transformed_values is not None:
                sns.histplot(transformed_values, kde=True, ax=ax2)
                ax2.set_title(f'Transformed Distribution - {transformation_name}')
                ax2.set_xlabel(transformation_name)
                ax2.set_ylabel('Frequency')

                # Add skewness
                skewness = stats.skew(transformed_values)
                ax2.text(0.05, 0.95, f'Skewness: {skewness:.2f}', transform=ax2.transAxes,
                        fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.tight_layout()

        return fig, (ax1, ax2) if ax2 is not None else (ax1,)

    def plot_outliers(self, variable: str, method: str = 'iqr', threshold: float = 1.5, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot a variable with outliers highlighted.

        Args:
            variable: Variable to plot
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            figsize: Figure size as (width, height) tuple

        Returns:
            Matplotlib figure and axes
        """
        # Check if variable exists and is numeric
        if variable not in self.data.columns:
            logger.warning(f"Variable {variable} not found in dataset.")
            return None, None

        if variable not in self.numeric_vars:
            logger.warning(f"Variable {variable} is not numeric.")
            return None, None

        # Get values (non-missing)
        values = self.data[variable].dropna()

        if len(values) == 0:
            logger.warning(f"Variable {variable} has no non-missing values.")
            return None, None

        # Detect outliers if not already detected
        if self.outliers is None or self.outliers['method'] != method or self.outliers['threshold'] != threshold:
            self.detect_outliers(method=method, threshold=threshold, variables=[variable])

        # Get outlier information
        outlier_details = self.outliers.get('details', {}).get(variable, {})
        outlier_indices = outlier_details.get('outlier_indices', [])

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create a boxplot and jittered scatter plot
        sns.boxplot(x=values, ax=ax, color='lightblue', width=0.3)

        # Add jittered points
        y_jitter = np.random.normal(0, 0.05, size=len(values))

        # Separate outliers and non-outliers
        non_outlier_indices = [i for i in range(len(self.data)) if i not in outlier_indices and not pd.isna(self.data.iloc[i][variable])]

        outlier_values = [self.data.iloc[i][variable] for i in outlier_indices]
        non_outlier_values = [self.data.iloc[i][variable] for i in non_outlier_indices]

        # Plot non-outliers
        ax.scatter(non_outlier_values, y_jitter[:len(non_outlier_values)], alpha=0.5, color='blue')

        # Plot outliers
        if outlier_values:
            ax.scatter(outlier_values, y_jitter[-len(outlier_values):], alpha=0.7, color='red', label='Outliers')

        # Add bounds if using IQR method
        if method == 'iqr':
            lower_bound = outlier_details.get('lower_bound')
            upper_bound = outlier_details.get('upper_bound')

            if lower_bound is not None and upper_bound is not None:
                ax.axvline(x=lower_bound, color='red', linestyle='--', alpha=0.7)
                ax.axvline(x=upper_bound, color='red', linestyle='--', alpha=0.7)

                # Add text labels for bounds
                y_pos = -0.2
                ax.text(lower_bound, y_pos, f'Lower: {lower_bound:.2f}', ha='center', va='top', color='red')
                ax.text(upper_bound, y_pos, f'Upper: {upper_bound:.2f}', ha='center', va='top', color='red')

        # Customize plot
        ax.set_title(f'Outlier Detection - {variable} (Method: {method}, Threshold: {threshold})')
        ax.set_xlabel(variable)
        ax.set_ylabel('')
        ax.set_yticks([])

        # Add legend if there are outliers
        if outlier_values:
            ax.legend()

        # Add outlier count text
        ax.text(0.05, 0.95, f'Outliers: {len(outlier_indices)} ({len(outlier_indices)/len(values)*100:.1f}%)',
                transform=ax.transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.tight_layout()

        return fig, ax

    def get_results(self) -> Dict:
        """
        Get all data quality assessment results.

        Returns:
            Dictionary containing all results
        """
        return self.results
