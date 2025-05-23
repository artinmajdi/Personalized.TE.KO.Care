import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from typing import Optional, List, Tuple # Corrected List and Tuple imports

from tekoa import logger

def generate_radar_chart_data(
    original_data: pd.DataFrame,
    labels: np.ndarray,
    numeric_vars: List[str],
    cluster_col_name: str = 'Cluster',
    scaler_type: str = 'zscore'
) -> pd.DataFrame:
    """
    Generates data suitable for a radar chart by calculating the mean of specified
    numeric variables for each cluster, after optional scaling.

    Args:
        original_data: Pandas DataFrame containing the original features.
        labels: Numpy array of cluster labels corresponding to rows in original_data.
        numeric_vars: List of numeric variable names to include in the radar chart.
        cluster_col_name: Name for the cluster label column that will be added to the data.
        scaler_type: Type of scaling to apply. Options:
                    'zscore': Standardizes data to have mean 0 and std 1.
                    'minmax': Scales data to a [0, 1] range.
                    'none'  : No scaling is applied.

    Returns:
        A Pandas DataFrame where rows are clusters and columns are the (potentially scaled)
        mean values of the numeric variables. Returns an empty DataFrame if inputs are invalid
        or processing fails.
    """
    if original_data.empty:
        logger.warning("generate_radar_chart_data: original_data is empty.")
        return pd.DataFrame()
    if labels.size == 0:
        logger.warning("generate_radar_chart_data: labels array is empty.")
        return pd.DataFrame()
    if not numeric_vars:
        logger.warning("generate_radar_chart_data: numeric_vars list is empty.")
        return pd.DataFrame()
    if len(original_data) != len(labels):
        logger.warning(f"generate_radar_chart_data: Mismatch between original_data length ({len(original_data)}) and labels length ({len(labels)}).")
        return pd.DataFrame()

    data_with_labels = original_data.copy()
    data_with_labels[cluster_col_name] = labels

    # Filter for valid numeric variables actually present in the dataframe
    valid_numeric_vars = []
    for var in numeric_vars:
        if var not in data_with_labels.columns:
            logger.warning(f"generate_radar_chart_data: Variable '{var}' not found in original_data. Skipping.")
            continue
        if not pd.api.types.is_numeric_dtype(data_with_labels[var]):
            logger.warning(f"generate_radar_chart_data: Variable '{var}' is not numeric. Skipping.")
            continue
        valid_numeric_vars.append(var)

    if not valid_numeric_vars:
        logger.warning("generate_radar_chart_data: No valid numeric variables found to process.")
        return pd.DataFrame()

    # Drop rows with NaNs *only in the selected valid numeric variables* before scaling
    # This prevents issues with scalers if NaNs are present.
    data_for_scaling = data_with_labels[valid_numeric_vars + [cluster_col_name]].dropna(subset=valid_numeric_vars)

    if data_for_scaling.empty:
        logger.warning("generate_radar_chart_data: DataFrame became empty after dropping NaNs from selected numeric variables.")
        return pd.DataFrame()

    scaled_vars_df = data_for_scaling[valid_numeric_vars].copy() # Work on a copy for scaling

    if scaler_type == 'zscore':
        try:
            # Apply zscore, handling columns that might be constant (std=0) which results in NaNs
            scaled_vars_df = scaled_vars_df.apply(lambda x: zscore(x, nan_policy='propagate') if x.notna().any() else x)
            # If zscore results in all NaNs for a column (e.g., constant value), fill with 0
            scaled_vars_df = scaled_vars_df.fillna(0)
        except Exception as e:
            logger.error(f"generate_radar_chart_data: Error during Z-score scaling: {e}")
            return pd.DataFrame() # Return empty on scaling error

    elif scaler_type == 'minmax':
        try:
            scaler = MinMaxScaler()
            # Scaler expects 2D array, so fit_transform on the DataFrame of numeric vars
            scaled_values = scaler.fit_transform(scaled_vars_df)
            scaled_vars_df = pd.DataFrame(scaled_values, columns=valid_numeric_vars, index=scaled_vars_df.index)
        except Exception as e:
            logger.error(f"generate_radar_chart_data: Error during MinMax scaling: {e}")
            return pd.DataFrame()

    elif scaler_type == 'none':
        logger.info("generate_radar_chart_data: No scaling applied to variables.")
    else:
        logger.warning(f"generate_radar_chart_data: Invalid scaler_type '{scaler_type}'. No scaling will be applied.")

    # Combine scaled numeric vars back with cluster labels from data_for_scaling
    # Ensure indices align after potential row drops by NaNs
    data_to_group = scaled_vars_df.join(data_for_scaling[cluster_col_name])

    if data_to_group.empty:
         logger.warning("generate_radar_chart_data: Data became empty before grouping. This might indicate an issue with scaling or joining.")
         return pd.DataFrame()

    try:
        radar_df = data_to_group.groupby(cluster_col_name)[valid_numeric_vars].mean()
    except Exception as e:
        logger.error(f"generate_radar_chart_data: Error during groupby mean calculation: {e}")
        return pd.DataFrame()

    logger.info(f"Radar chart data generated successfully with scaler: {scaler_type}. Shape: {radar_df.shape}")
    return radar_df


def plot_radar_chart(
    radar_data: pd.DataFrame,
    title: str,
    value_range: Optional[Tuple[float, float]] = None
) -> go.Figure:
    """
    Creates a radar chart (spider plot) from pre-calculated radar data.

    Args:
        radar_data: Pandas DataFrame where rows are clusters (or groups) and
                    columns are variables. Values are typically means of these variables.
                    This should be the output from `generate_radar_chart_data`.
        title: The title for the radar chart.
        value_range: Optional tuple (min_val, max_val) to set the range for the
                     radial axis. If None, Plotly auto-scales the axis.
                     Useful for scaled data (e.g., (0, 1) for minmax, or (-3, 3) for zscore).

    Returns:
        A Plotly graph_objects.Figure containing the radar chart. Returns an empty
        figure if radar_data is empty.
    """
    fig = go.Figure()

    if radar_data.empty:
        logger.warning("plot_radar_chart: radar_data is empty. Returning an empty figure.")
        fig.update_layout(title=f"{title} (No data to display)")
        return fig

    categories = radar_data.columns.tolist()
    if not categories:
        logger.warning("plot_radar_chart: radar_data has no columns (categories). Returning an empty figure.")
        fig.update_layout(title=f"{title} (No categories to display)")
        return fig

    for cluster_index in radar_data.index:
        try:
            values = radar_data.loc[cluster_index].tolist()
            fig.add_trace(go.Scatterpolar(
                r=values + values[:1],  # Close the loop
                theta=categories + categories[:1],  # Close the loop
                fill='toself',
                name=f'Cluster {cluster_index}'
            ))
        except Exception as e:
            logger.error(f"plot_radar_chart: Error adding trace for cluster {cluster_index}: {e}")
            # Continue to try and plot other clusters if possible

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=list(value_range) if value_range else None # Convert tuple to list for plotly
            )
        ),
        title=title,
        showlegend=True
    )
    logger.info(f"Radar chart '{title}' created successfully with {len(radar_data.index)} traces.")
    return fig
