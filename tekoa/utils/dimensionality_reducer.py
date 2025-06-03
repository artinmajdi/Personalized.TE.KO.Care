"""
Dimensionality Reduction Module for TE-KOA.

This module provides functionality for dimensionality reduction including:
- Principal Component Analysis (PCA) for numeric variables
- Factor Analysis of Mixed Data (FAMD) for mixed types
- Component visualization and interpretation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from prince import FAMD
from tekoa.configuration import logger


# TODO: currently, evertime user wants to change anything in the dashboard, it keeps reloading the dataset. i shoudl fix it so that it would only load the dataset once. ideally not reset teh whole dashboard until user has clicked on the "Perform PCA" or "Perform FAMD" button

class DimensionalityReducer:
    """Class for reducing dimensionality of TE-KOA dataset variables."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DimensionalityReducer.

        Args:
            data: DataFrame containing the dataset to analyze
        """
        self.data = data.copy()

        # Track numeric and categorical variables
        self.numeric_vars = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_vars = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Store results from dimensionality reduction methods
        self.pca_results = None
        self.famd_results = None

        # Track optimal number of components
        self.optimal_components = None

        # Initialize results storage
        self.results = {
            'pca': {},
            'famd': {},
            'variable_clusters': {},
            'optimal_components': None
        }

    def perform_pca(self,
                   variables: List[str] = None,
                   n_components: int = None,
                   standardize: bool = True,
                   random_state: int = 42) -> Dict:
        """
        Perform Principal Component Analysis on numeric variables.

        Args:
            variables: List of variables to include in PCA. If None, uses all numeric variables.
            n_components: Number of components to retain. If None, retains all components.
            standardize: Whether to standardize the data before PCA.
            random_state: Random state for reproducibility.

        Returns:
            Dictionary with PCA results
        """
        # Use all numeric variables if none specified
        if variables is None:
            variables = self.numeric_vars
        else:
            # Keep only numeric variables from the provided list
            variables = [var for var in variables if var in self.numeric_vars]

        if len(variables) == 0:
            logger.warning("No numeric variables available for PCA.")
            return {}

        # Extract the data
        X = self.data[variables].copy()

        # Handle missing values - drop rows with any missing values for now
        X_clean = X.dropna()

        if len(X_clean) < len(X):
            logger.warning(f"Dropped {len(X) - len(X_clean)} rows with missing values for PCA.")

        if len(X_clean) == 0:
            logger.error("No complete cases available for PCA.")
            return {}

        # Standardize the data if requested
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
        else:
            X_scaled = X_clean.values

        # Determine number of components
        if n_components is None:
            n_components = min(len(X_clean) - 1, len(variables))

        # Perform PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        components = pca.fit_transform(X_scaled)

        # Store results
        self.pca_results = {
            'pca_object': pca,
            'components': components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_explained_variance': np.cumsum(pca.explained_variance_ratio_),
            'loadings': pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=variables
            ),
            'scaler': scaler if standardize else None,
            'variables': variables,
            'samples_used': len(X_clean)
        }

        # Save results to the main results dict
        self.results['pca'] = self.pca_results

        return self.pca_results

    def perform_famd(self,
                    n_components: int = 10,
                    random_state: int = 42) -> Dict:
        """
        Perform Factor Analysis of Mixed Data for both numeric and categorical variables.

        Args:
            n_components: Number of components to retain.
            random_state: Random state for reproducibility.

        Returns:
            Dictionary with FAMD results containing:
            - 'famd_object'                  : The fitted FAMD object
            - 'transformed_data'             : Transformed data (scores)
            - 'explained_variance'           : Explained variance for each component
            - 'cumulative_explained_variance': Cumulative explained variance
            - 'variable_coordinates'         : Coordinates of variables in the component space
            - 'samples_used'                 : Number of samples used in the analysis
        """
        # Prepare the data - FAMD requires all categorical columns to be of type 'category'
        data_for_famd = self.data.copy()
        logger.info(f"Original data shape for FAMD: {data_for_famd.shape}")

        # Convert categorical columns to category type
        for col in self.categorical_vars:
            if col in data_for_famd.columns:  # Only convert if column exists
                data_for_famd[col] = data_for_famd[col].astype('category')


        # Handle missing values - drop rows with any missing values for now
        data_clean = data_for_famd.dropna()
        logger.info(f"Data shape after dropna() for FAMD: {data_clean.shape}")

        if len(data_clean) < len(data_for_famd):
            logger.warning(f"Dropped {len(data_for_famd) - len(data_clean)} rows with missing values for FAMD.")

        if data_clean.empty:
            logger.error("No data remaining after removing rows with missing values. FAMD cannot proceed.")
            raise ValueError("No data remaining for FAMD after removing rows with missing values.")

        try:
            # Perform FAMD
            famd = FAMD(n_components=n_components, n_iter=10, random_state=random_state)
            transformed_data = famd.fit_transform(data_clean)

            # Get variable coordinates (correlations between variables and components)
            # For FAMD, we'll use the column_correlations_ attribute if available
            # Otherwise, we'll compute correlations manually
            try:
                variable_coords = famd.column_correlations_
            except AttributeError:
                # Compute correlations manually if the attribute doesn't exist
                variable_coords = pd.DataFrame(
                    np.corrcoef(data_clean.select_dtypes(include=[np.number]).values.T,
                                transformed_data.values, rowvar=False)[:data_clean.shape[1], data_clean.shape[1]:],
                    index=data_clean.columns,
                    columns=[f'Dim{i+1}' for i in range(n_components)]
                )

            # Store results
            self.famd_results = {
                'famd_object': famd,
                'transformed_data': transformed_data,
                'explained_variance': famd.explained_inertia_,
                'cumulative_explained_variance': np.cumsum(famd.explained_inertia_),
                'variable_coordinates': variable_coords,
                'samples_used': len(data_clean),
                'feature_names': data_clean.columns.tolist()
            }

            # Save results to the main results dict
            self.results['famd'] = self.famd_results

            return self.famd_results

        except Exception as e:
            logger.error(f"Error performing FAMD: {e}", exc_info=True)
            return {}

    def get_component_loadings(self, method: str = 'pca') -> pd.DataFrame:
        """
        Extract and interpret component loadings.

        Args:
            method: Method used for dimensionality reduction ('pca' or 'famd')

        Returns:
            DataFrame with component loadings
        """
        if method == 'pca':
            if self.pca_results is None:
                logger.warning("PCA results not available. Call perform_pca() first.")
                return pd.DataFrame()

            return self.pca_results['loadings']

        elif method == 'famd':
            if self.famd_results is None:
                logger.warning("FAMD results not available. Call perform_famd() first.")
                return pd.DataFrame()

            return self.famd_results['variable_coordinates']

        else:
            logger.warning(f"Unknown method: {method}. Use 'pca' or 'famd'.")
            return pd.DataFrame()

    def get_variance_explained(self, method: str = 'pca') -> pd.DataFrame:
        """
        Calculate explained variance by components.

        Args:
            method: Method used for dimensionality reduction ('pca' or 'famd')

        Returns:
            DataFrame with explained variance information
        """
        if method == 'pca':
            if self.pca_results is None:
                logger.warning("PCA results not available. Call perform_pca() first.")
                return pd.DataFrame()

            return pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(self.pca_results['explained_variance_ratio']))],
                'Explained_Variance': self.pca_results['explained_variance_ratio'],
                'Cumulative_Variance': self.pca_results['cumulative_explained_variance']
            })

        elif method == 'famd':
            if self.famd_results is None:
                logger.warning("FAMD results not available. Call perform_famd() first.")
                return pd.DataFrame()

            return pd.DataFrame({
                'Component': [f'Dim{i+1}' for i in range(len(self.famd_results['explained_variance']))],
                'Explained_Variance': self.famd_results['explained_variance'],
                'Cumulative_Variance': self.famd_results['cumulative_explained_variance']
            })

        else:
            logger.warning(f"Unknown method: {method}. Use 'pca' or 'famd'.")
            return pd.DataFrame()

    def get_optimal_components(self, method: str = 'pca', variance_threshold: float = 0.75) -> int:
        """
        Determine optimal number of components based on variance threshold.

        Args:
            method: Method used for dimensionality reduction ('pca' or 'famd')
            variance_threshold: Minimum cumulative variance to retain

        Returns:
            Optimal number of components
        """
        variance_df = self.get_variance_explained(method)

        if variance_df.empty:
            return 0

        # Find the first component that exceeds the threshold
        for i, cum_var in enumerate(variance_df['Cumulative_Variance']):
            if cum_var >= variance_threshold:
                optimal = i + 1
                break
        else:
            # If threshold not reached, use all components
            optimal = len(variance_df)

        # Store the result
        self.optimal_components = optimal
        self.results['optimal_components'] = {
            'method': method,
            'variance_threshold': variance_threshold,
            'optimal_number': optimal
        }

        return optimal

    def transform_data(self, method: str = 'pca', n_components: int = None) -> pd.DataFrame:
        """
        Transform the data to the new component space.

        Args:
            method: Method used for dimensionality reduction ('pca' or 'famd')
            n_components: Number of components to use. If None, uses all available components.

        Returns:
            DataFrame with transformed data
        """
        if method == 'pca':
            if self.pca_results is None:
                logger.warning("PCA results not available. Call perform_pca() first.")
                return pd.DataFrame()

            # Limit to specified number of components
            if n_components is None:
                n_components = self.pca_results['components'].shape[1]
            else:
                n_components = min(n_components, self.pca_results['components'].shape[1])

            # Create DataFrame with transformed data
            return pd.DataFrame(
                self.pca_results['components'][:, :n_components],
                columns=[f'PC{i+1}' for i in range(n_components)]
            )

        elif method == 'famd':
            if self.famd_results is None:
                logger.warning("FAMD results not available. Call perform_famd() first.")
                return pd.DataFrame()

            # Limit to specified number of components
            if n_components is None:
                n_components = self.famd_results['transformed_data'].shape[1]
            else:
                n_components = min(n_components, self.famd_results['transformed_data'].shape[1])

            # Create DataFrame with transformed data
            return pd.DataFrame(
                self.famd_results['transformed_data'].iloc[:, :n_components],
                columns=[f'Dim{i+1}' for i in range(n_components)]
            )

        else:
            logger.warning(f"Unknown method: {method}. Use 'pca' or 'famd'.")
            return pd.DataFrame()

    def plot_scree(self, method: str = 'pca', figsize: Tuple[int, int] = (10, 6)):
        """
        Create a scree plot to visualize explained variance by component.

        Args:
            method: Method used for dimensionality reduction ('pca' or 'famd')
            figsize: Figure size as (width, height) tuple

        Returns:
            Matplotlib figure and axes
        """
        variance_df = self.get_variance_explained(method)

        if variance_df.empty:
            logger.warning(f"{method.upper()} results not available. Call perform_{method}() first.")
            # Create an empty figure with a message
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5,
                   f"{method.upper()} results not available.\nPlease perform {method.upper()} first.",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return fig, ax

        # Create the figure
        fig, ax1 = plt.subplots(figsize=figsize)


        # Plot individual explained variance
        ax1.bar(variance_df['Component'], variance_df['Explained_Variance'], alpha=0.7, label='Individual')
        ax1.set_ylabel('Explained Variance Ratio', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Rotate x-axis labels for better readability
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Create a second y-axis for cumulative variance
        ax2 = ax1.twinx()
        ax2.plot(variance_df['Component'], variance_df['Cumulative_Variance'], 'r-', marker='o', label='Cumulative')
        ax2.set_ylabel('Cumulative Explained Variance', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Add threshold line if optimal components determined
        if hasattr(self, 'optimal_components') and self.optimal_components and \
           'optimal_components' in self.results and self.results['optimal_components'] and \
           self.results['optimal_components'].get('method') == method:

            threshold = self.results['optimal_components'].get('variance_threshold', 0.75)
            ax2.axhline(y=threshold, color='g', linestyle='--', alpha=0.7,
                       label=f'{threshold*100:.0f}% Threshold')

            # Highlight optimal number of components
            optimal = self.optimal_components
            if optimal <= len(variance_df):
                ax1.axvline(x=optimal-1, color='g', linestyle='--', alpha=0.7)

        # Customize the plot
        ax1.set_title(f'Scree Plot - {method.upper()}')
        ax1.set_xlabel('Component')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                 loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        plt.tight_layout()

        return fig, (ax1, ax2)

    def plot_component_loadings(self, component: int = 1, method: str = 'pca', n_top: int = 10, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot component loadings to interpret component meaning.

        Args:
            component: Component index (1-based) to visualize
            method: Method used for dimensionality reduction ('pca' or 'famd')
            n_top: Number of top variables to show
            figsize: Figure size as (width, height) tuple

        Returns:
            Matplotlib figure and axes
        """
        loadings = self.get_component_loadings(method)

        if loadings.empty:
            logger.warning(f"{method.upper()} results not available.")
            return None, None

        # Determine component name based on method
        if method == 'pca':
            component_name = f'PC{component}'
        else:  # famd
            component_name = f'Dim{component}'

        if component_name not in loadings.columns:
            logger.warning(f"Component {component_name} not found in loadings.")
            return None, None

        # Get absolute loadings and sort
        abs_loadings = loadings[component_name].abs().sort_values(ascending=False)

        # Take top n variables
        top_vars = abs_loadings.head(n_top).index.tolist()

        # Create a new DataFrame with original loadings for top variables
        plot_df = pd.DataFrame({
            'Variable': top_vars,
            'Loading': [loadings.loc[var, component_name] for var in top_vars]
        }).sort_values('Loading')

        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create horizontal bar plot
        bars = ax.barh(plot_df['Variable'], plot_df['Loading'], color=plot_df['Loading'].map(lambda x: 'blue' if x > 0 else 'red'))

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else width - 0.05
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                    va='center', ha='left' if width > 0 else 'right', color='black')

        # Customize the plot
        ax.set_title(f'Top {n_top} Variable Loadings - {component_name}')
        ax.set_xlabel('Loading')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()

        return fig, ax

    def plot_biplot(self, pc1: int = 1, pc2: int = 2, method: str = 'pca', figsize: Tuple[int, int] = (12, 10)):
        """
        Create a biplot to visualize observations and variables in component space.

        Args:
            pc1: First principal component to plot (1-based index)
            pc2: Second principal component to plot (1-based index)
            method: Method used for dimensionality reduction ('pca' or 'famd')
            figsize: Figure size as (width, height) tuple

        Returns:
            Matplotlib figure and axes
        """
        if method == 'pca':
            if self.pca_results is None:
                logger.warning("PCA results not available. Call perform_pca() first.")
                return None, None

            # Get component indices (0-based)
            pc1_idx = pc1 - 1
            pc2_idx = pc2 - 1

            # Check if components are valid
            if pc1_idx >= self.pca_results['components'].shape[1] or pc2_idx >= self.pca_results['components'].shape[1]:
                logger.warning(f"Invalid component indices: PC{pc1} or PC{pc2} not available.")
                return None, None

            # Get principal components and loadings
            components = self.pca_results['components']
            loadings = self.pca_results['loadings'].values
            variables = self.pca_results['variables']

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Plot observations
            ax.scatter(components[:, pc1_idx], components[:, pc2_idx], alpha=0.7)

            # Plot loadings
            scaling = np.max(np.abs(components[:, [pc1_idx, pc2_idx]])) / np.max(np.abs(loadings[:, [pc1_idx, pc2_idx]])) * 0.8
            for i, var in enumerate(variables):
                ax.arrow(0, 0, loadings[i, pc1_idx] * scaling, loadings[i, pc2_idx] * scaling,
                         head_width=0.05, head_length=0.05, fc='red', ec='red')
                ax.text(loadings[i, pc1_idx] * scaling * 1.15, loadings[i, pc2_idx] * scaling * 1.15,
                        var, color='red', ha='center', va='center')

            # Customize plot
            ax.set_xlabel(f'PC{pc1} ({self.pca_results["explained_variance_ratio"][pc1_idx]:.2%})')
            ax.set_ylabel(f'PC{pc2} ({self.pca_results["explained_variance_ratio"][pc2_idx]:.2%})')
            ax.set_title(f'PCA Biplot - PC{pc1} vs PC{pc2}')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
            ax.grid(alpha=0.3)

            plt.tight_layout()

            return fig, ax

        elif method == 'famd':
            # Implement FAMD biplot if needed - similar to PCA but using FAMD results
            logger.warning("FAMD biplot not implemented yet.")
            return None, None

        else:
            logger.warning(f"Unknown method: {method}. Use 'pca' or 'famd'.")
            return None, None

    def get_results(self) -> Dict:
        """
        Get all dimensionality reduction results.

        Returns:
            Dictionary containing all results
        """
        return self.results
