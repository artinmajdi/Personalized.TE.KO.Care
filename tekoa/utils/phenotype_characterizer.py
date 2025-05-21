import pandas as pd
import numpy as np
from scipy import stats as scipy_stats # f_oneway, chi2_contingency
import logging

logger = logging.getLogger(__name__)

def _calculate_eta_squared(f_statistic: float, df_between: int, df_within: int) -> float:
    """
    Calculates Eta-squared, an effect size for ANOVA.

    Args:
        f_statistic: The F-statistic from the ANOVA test.
        df_between: Degrees of freedom between groups.
        df_within: Degrees of freedom within groups.

    Returns:
        Eta-squared value, or np.nan if calculation is not possible.
    """
    if df_within == 0 or np.isnan(f_statistic) or f_statistic * df_between + df_within == 0: # Added check for denominator
        return np.nan
    eta_sq = (f_statistic * df_between) / (f_statistic * df_between + df_within)
    return eta_sq

def _calculate_cramers_v(chi2_statistic: float, n: int, k: int, r: int) -> float:
    """
    Calculates Cramer's V, an effect size for Chi-Square test of association.

    Args:
        chi2_statistic: The Chi-Square statistic.
        n: Total number of observations.
        k: Number of columns in the contingency table (e.g., number of clusters).
        r: Number of rows in the contingency table (e.g., number of variable categories).

    Returns:
        Cramer's V value, or np.nan if calculation is not possible.
    """
    if n == 0 or np.isnan(chi2_statistic):
        return np.nan
    min_dim = min(k - 1, r - 1)
    if min_dim == 0:
        return np.nan
    cram_v = np.sqrt(chi2_statistic / (n * min_dim))
    return cram_v

def compare_variable_across_clusters(data_with_labels: pd.DataFrame, variable: str, cluster_col_name: str = 'Cluster') -> dict:
    """
    Compares a variable across different clusters using ANOVA for numeric variables
    or Chi-Square test for categorical variables.

    Args:
        data_with_labels: Pandas DataFrame containing the original variable data and a cluster label column.
        variable: The name of the variable column to compare.
        cluster_col_name: The name of the column containing cluster labels.

    Returns:
        A dictionary containing:
            - 'TestType': Type of test ('ANOVA', 'Chi-Square', or error type).
            - 'Statistic': F-statistic for ANOVA, Chi2-statistic for Chi-Square, or np.nan.
            - 'PValue': P-value from the test, or np.nan.
            - 'EffectSize': Eta-squared for ANOVA, Cramer's V for Chi-Square, or np.nan.
    """
    default_error_return = {'TestType': 'Error', 'Statistic': np.nan, 'PValue': np.nan, 'EffectSize': np.nan}

    if variable not in data_with_labels.columns or cluster_col_name not in data_with_labels.columns:
        logger.warning(f"Variable '{variable}' or cluster column '{cluster_col_name}' not found in DataFrame.")
        return {**default_error_return, 'TestType': 'Error_ColumnNotFound'}

    working_df = data_with_labels[[variable, cluster_col_name]].copy().dropna()
    unique_clusters = working_df[cluster_col_name].unique()

    if len(unique_clusters) < 2:
        logger.warning(f"Variable '{variable}': Insufficient clusters (<2) for comparison.")
        return {**default_error_return, 'TestType': 'InsufficientClusters'}

    if pd.api.types.is_numeric_dtype(working_df[variable].dtype):
        groups = [working_df[variable][working_df[cluster_col_name] == i] for i in unique_clusters]
        valid_groups = [g for g in groups if len(g) >= 1] # Each group must have at least one observation

        if len(valid_groups) < 2:
            logger.warning(f"Variable '{variable}': Not enough valid groups (<2) for ANOVA after dropping empty groups.")
            return {**default_error_return, 'TestType': 'ANOVA_NotEnoughGroups'}
        
        # Check if all group means are identical or groups have zero variance (scipy's f_oneway limitation)
        # Or if any group has zero variance and there are only two groups
        group_means = [g.mean() for g in valid_groups]
        group_vars = [g.var() for g in valid_groups]

        if len(set(group_means)) == 1 and all(v == 0 for v in group_vars):
             logger.warning(f"Variable '{variable}': All groups have identical means and zero variance. ANOVA not meaningful.")
             return {'TestType': 'ANOVA_NoVarianceOrSameMean', 'Statistic': 0.0, 'PValue': 1.0, 'EffectSize': 0.0}


        try:
            f_stat, p_value = scipy_stats.f_oneway(*valid_groups)
            df_between = len(valid_groups) - 1
            df_within = sum(len(g) for g in valid_groups) - len(valid_groups)
            eta_sq = _calculate_eta_squared(f_stat, df_between, df_within)
            return {'TestType': 'ANOVA', 'Statistic': f_stat, 'PValue': p_value, 'EffectSize': eta_sq}
        except ValueError as e:
            logger.error(f"Variable '{variable}': Error during ANOVA calculation: {e}")
            return {**default_error_return, 'TestType': 'ANOVA_CalculationError'}
    else: # Categorical variable
        try:
            contingency_table = pd.crosstab(working_df[variable], working_df[cluster_col_name])
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                logger.warning(f"Variable '{variable}': Contingency table too small for Chi-Square test ({contingency_table.shape}).")
                return {**default_error_return, 'TestType': 'ChiSquare_SmallTable'}
            
            chi2, p, dof, expected = scipy_stats.chi2_contingency(contingency_table)
            n_obs = contingency_table.sum().sum()
            cram_v = _calculate_cramers_v(chi2, n_obs, contingency_table.shape[1], contingency_table.shape[0])
            return {'TestType': 'Chi-Square', 'Statistic': chi2, 'PValue': p, 'EffectSize': cram_v}
        except ValueError as e:
            logger.error(f"Variable '{variable}': Error during Chi-Square calculation: {e}")
            return {**default_error_return, 'TestType': 'ChiSquare_CalculationError'}

def characterize_phenotypes(original_data: pd.DataFrame, labels: np.ndarray, variables_to_compare: list[str], cluster_col_name: str = 'Cluster') -> pd.DataFrame:
    """
    Characterizes phenotypes (clusters) by comparing specified variables across them.
    Performs ANOVA for numeric variables and Chi-Square for categorical variables.
    Applies FDR correction (Benjamini/Hochberg) to p-values.

    Args:
        original_data: Pandas DataFrame containing the original variables.
        labels: Numpy array of cluster labels for each row in original_data.
        variables_to_compare: A list of column names in original_data to compare across clusters.
        cluster_col_name: Name to be used for the cluster label column added to the data.

    Returns:
        A Pandas DataFrame with results, including 'Variable', 'TestType', 'Statistic',
        'PValue', 'CorrectedPValue', 'RejectNullFDR', and 'EffectSize' for each variable.
        Returns an empty DataFrame with these columns if inputs are invalid.
    """
    expected_cols = ['Variable', 'TestType', 'Statistic', 'PValue', 'CorrectedPValue', 'RejectNullFDR', 'EffectSize']

    if original_data.empty or len(labels) == 0 or not variables_to_compare or len(labels) != len(original_data):
        logger.warning("Invalid input for characterize_phenotypes. Returning empty DataFrame.")
        return pd.DataFrame(columns=expected_cols)

    data_with_labels = original_data.copy() # Uses deep=True by default
    data_with_labels[cluster_col_name] = labels

    results_list = []
    for variable in variables_to_compare:
        if variable not in original_data.columns:
            logger.warning(f"Variable '{variable}' not found in original_data. Skipping.")
            results_list.append({
                'Variable': variable, 
                'TestType': 'Error_VariableNotFoundInSource', 
                'Statistic': np.nan, 
                'PValue': np.nan, 
                'EffectSize': np.nan
            })
            continue
        comparison_result = compare_variable_across_clusters(data_with_labels, variable, cluster_col_name)
        results_list.append({'Variable': variable, **comparison_result})

    if not results_list:
        return pd.DataFrame(columns=expected_cols)
        
    results_df = pd.DataFrame(results_list)

    # Initialize FDR columns
    results_df['CorrectedPValue'] = np.nan
    results_df['RejectNullFDR'] = pd.NA # Use pandas NA for nullable boolean

    # Apply FDR correction
    valid_p_values_idx = results_df['PValue'].notna() & (results_df['PValue'] >= 0) & (results_df['PValue'] <= 1)
    
    if valid_p_values_idx.any():
        p_values_to_correct = results_df.loc[valid_p_values_idx, 'PValue']
        if not p_values_to_correct.empty:
            try:
                from statsmodels.stats.multitest import multipletests
                reject, pvals_corrected, _, _ = multipletests(p_values_to_correct, method='fdr_bh', alpha=0.05)
                results_df.loc[valid_p_values_idx, 'CorrectedPValue'] = pvals_corrected
                results_df.loc[valid_p_values_idx, 'RejectNullFDR'] = reject.astype(pd.BooleanDtype()) # Store as nullable boolean
            except ImportError:
                logger.warning(
                    "statsmodels.stats.multitest not found. FDR correction cannot be applied. "
                    "Please install statsmodels: pip install statsmodels"
                )
            except Exception as e: # Catch any other error during FDR
                logger.error(f"Error during FDR correction: {e}")


    # Ensure all expected columns exist and are in order
    for col in expected_cols:
        if col not in results_df.columns:
            if col == 'RejectNullFDR':
                results_df[col] = pd.NA 
            else:
                results_df[col] = np.nan
    
    return results_df[expected_cols]
