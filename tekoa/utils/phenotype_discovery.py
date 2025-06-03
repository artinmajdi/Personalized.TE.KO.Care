"""
Phenotype Discovery Module for Knee Osteoarthritis Analysis.

This module implements Phase II of the KOA analysis pipeline, providing
clustering algorithms, validation metrics, and phenotype characterization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

from tekoa import logger


class PhenotypeDiscovery:
    """Class for discovering phenotypes through clustering analysis."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the PhenotypeDiscovery class.

        Args:
            data: Input DataFrame (can be original data or FAMD components)
        """
        self.data = data
        self.numeric_data = None
        self.clustering_results = {}
        self.validation_metrics = {}
        self.optimal_clusters = {}
        self.phenotype_profiles = {}

        # Prepare data for clustering
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for clustering by handling mixed types and missing values."""
        # Select only numeric columns for clustering
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.numeric_data = self.data[numeric_cols].copy()

        # Remove rows with any NaN values
        original_rows = len(self.numeric_data)
        self.numeric_data = self.numeric_data.dropna()
        rows_after_nan_removal = len(self.numeric_data)

        if original_rows != rows_after_nan_removal:
            logger.info(f"Removed {original_rows - rows_after_nan_removal} rows with missing values")

        # Standardize the data
        scaler = StandardScaler()
        self.scaled_data = pd.DataFrame(
            scaler.fit_transform(self.numeric_data),
            columns=self.numeric_data.columns,
            index=self.numeric_data.index
        )

        logger.info(f"Prepared {len(self.numeric_data.columns)} numeric features for clustering with {len(self.numeric_data)} complete cases")

    def perform_kmeans(self, n_clusters_range: range = range(2, 7)) -> Dict:
        """
        Perform K-means clustering with multiple k values.

        Args:
            n_clusters_range: Range of cluster numbers to try

        Returns:
            Dictionary containing clustering results
        """
        logger.info("Performing K-means clustering...")

        results = {}
        for k in n_clusters_range:
            # Fit K-means
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(self.scaled_data)

            # Calculate metrics
            silhouette = silhouette_score(self.scaled_data, labels)
            calinski = calinski_harabasz_score(self.scaled_data, labels)
            inertia = kmeans.inertia_

            results[k] = {
                'model': kmeans,
                'labels': labels,
                'silhouette': silhouette,
                'calinski': calinski,
                'inertia': inertia,
                'centers': kmeans.cluster_centers_
            }

            logger.info(f"K={k}: Silhouette={silhouette:.3f}, Calinski={calinski:.1f}")

        self.clustering_results['kmeans'] = results
        return results

    def perform_agglomerative(self, n_clusters_range: range = range(2, 7)) -> Dict:
        """
        Perform Agglomerative Hierarchical clustering.

        Args:
            n_clusters_range: Range of cluster numbers to try

        Returns:
            Dictionary containing clustering results
        """
        logger.info("Performing Agglomerative clustering...")

        results = {}
        for k in n_clusters_range:
            # Fit Agglomerative clustering
            agglomerative = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = agglomerative.fit_predict(self.scaled_data)

            # Calculate metrics
            silhouette = silhouette_score(self.scaled_data, labels)
            calinski = calinski_harabasz_score(self.scaled_data, labels)

            results[k] = {
                'model': agglomerative,
                'labels': labels,
                'silhouette': silhouette,
                'calinski': calinski,
                'n_clusters': agglomerative.n_clusters_
            }

            logger.info(f"K={k}: Silhouette={silhouette:.3f}, Calinski={calinski:.1f}")

        self.clustering_results['agglomerative'] = results
        return results

    def perform_gmm(self, n_components_range: range = range(2, 7)) -> Dict:
        """
        Perform Gaussian Mixture Model clustering.

        Args:
            n_components_range: Range of component numbers to try

        Returns:
            Dictionary containing clustering results
        """
        logger.info("Performing Gaussian Mixture Model clustering...")

        results = {}
        for n in n_components_range:
            # Fit GMM
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(self.scaled_data)
            labels = gmm.predict(self.scaled_data)
            proba = gmm.predict_proba(self.scaled_data)

            # Calculate metrics
            silhouette = silhouette_score(self.scaled_data, labels)
            calinski = calinski_harabasz_score(self.scaled_data, labels)
            bic = gmm.bic(self.scaled_data)
            aic = gmm.aic(self.scaled_data)

            results[n] = {
                'model': gmm,
                'labels': labels,
                'probabilities': proba,
                'silhouette': silhouette,
                'calinski': calinski,
                'bic': bic,
                'aic': aic,
                'means': gmm.means_,
                'covariances': gmm.covariances_
            }

            logger.info(f"N={n}: Silhouette={silhouette:.3f}, BIC={bic:.1f}, AIC={aic:.1f}")

        self.clustering_results['gmm'] = results
        return results

    def calculate_gap_statistic(self, method: str = 'kmeans',
                              n_clusters_range: range = range(2, 7),
                              n_references: int = 10) -> Dict:
        """
        Calculate gap statistic for determining optimal number of clusters.

        Args:
            method: Clustering method to use
            n_clusters_range: Range of cluster numbers to try
            n_references: Number of reference datasets to generate

        Returns:
            Dictionary containing gap statistics
        """
        logger.info(f"Calculating gap statistic for {method}...")

        gaps = []
        sk_values = []

        for k in n_clusters_range:
            # Get within-cluster sum of squares for actual data
            if method == 'kmeans':
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                kmeans.fit(self.scaled_data)
                W_k = self._calculate_W_k(self.scaled_data, kmeans.labels_, kmeans.cluster_centers_)
            else:
                raise ValueError(f"Gap statistic not implemented for method: {method}")

            # Generate reference datasets and calculate W_k for each
            W_kb_values = []
            for _ in range(n_references):
                # Generate random data with same shape
                random_data = np.random.uniform(
                    self.scaled_data.min().min(),
                    self.scaled_data.max().max(),
                    size=self.scaled_data.shape
                )
                random_df = pd.DataFrame(random_data, columns=self.scaled_data.columns)

                # Cluster random data
                kmeans_random = KMeans(n_clusters=k, n_init=10, random_state=42)
                kmeans_random.fit(random_df)
                W_kb = self._calculate_W_k(random_df, kmeans_random.labels_, kmeans_random.cluster_centers_)
                W_kb_values.append(np.log(W_kb))

            # Calculate gap statistic
            gap = np.mean(W_kb_values) - np.log(W_k)
            sk = np.std(W_kb_values) * np.sqrt(1 + 1/n_references)

            gaps.append(gap)
            sk_values.append(sk)

        # Determine optimal k using gap statistic criterion
        optimal_k = n_clusters_range[0]
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - sk_values[i + 1]:
                optimal_k = n_clusters_range[i]
                break

        gap_results = {
            'gaps': gaps,
            'sk_values': sk_values,
            'k_values': list(n_clusters_range),
            'optimal_k': optimal_k
        }

        self.validation_metrics['gap_statistic'] = gap_results
        return gap_results

    def _calculate_W_k(self, data: pd.DataFrame, labels: np.ndarray, centers: np.ndarray) -> float:
        """Calculate within-cluster sum of squares."""
        W_k = 0
        for i in range(len(centers)):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                W_k += np.sum(cdist(cluster_points, [centers[i]], 'euclidean')**2)
        return W_k

    def bootstrap_stability(self, method: str = 'kmeans', k: int = 3,
                          n_bootstrap: int = 100, subsample_size: float = 0.8) -> Dict:
        """
        Assess clustering stability using bootstrap resampling.

        Args:
            method: Clustering method to use
            k: Number of clusters
            n_bootstrap: Number of bootstrap samples
            subsample_size: Proportion of data to use in each bootstrap

        Returns:
            Dictionary containing stability metrics
        """
        logger.info(f"Assessing bootstrap stability for {method} with k={k}...")

        n_samples = len(self.scaled_data)
        n_subsample = int(n_samples * subsample_size)

        # Store cluster assignments for each bootstrap
        bootstrap_labels = []

        for i in range(n_bootstrap):
            # Create bootstrap sample
            indices = np.random.choice(n_samples, n_subsample, replace=True)
            bootstrap_data = self.scaled_data.iloc[indices]

            # Perform clustering
            if method == 'kmeans':
                model = KMeans(n_clusters=k, n_init=10, random_state=i)
            elif method == 'agglomerative':
                model = AgglomerativeClustering(n_clusters=k, linkage='ward')
            elif method == 'gmm':
                model = GaussianMixture(n_components=k, random_state=i)
            else:
                raise ValueError(f"Unknown method: {method}")

            labels = model.fit_predict(bootstrap_data)

            # Store labels with original indices
            full_labels = np.full(n_samples, -1)
            full_labels[indices] = labels
            bootstrap_labels.append(full_labels)

        # Calculate Jaccard similarity between bootstrap results
        jaccard_scores = []
        for i in range(n_bootstrap):
            for j in range(i + 1, n_bootstrap):
                # Get indices where both have valid labels
                valid_indices = (bootstrap_labels[i] != -1) & (bootstrap_labels[j] != -1)

                if np.sum(valid_indices) > 0:
                    labels1 = bootstrap_labels[i][valid_indices]
                    labels2 = bootstrap_labels[j][valid_indices]

                    # Calculate Jaccard similarity
                    jaccard = self._calculate_jaccard(labels1, labels2)
                    jaccard_scores.append(jaccard)

        stability_results = {
            'mean_jaccard': np.mean(jaccard_scores),
            'std_jaccard': np.std(jaccard_scores),
            'min_jaccard': np.min(jaccard_scores),
            'is_stable': np.mean(jaccard_scores) >= 0.75  # Threshold from the document
        }

        logger.info(f"Stability assessment - Mean Jaccard: {stability_results['mean_jaccard']:.3f}")

        return stability_results

    def _calculate_jaccard(self, labels1: np.ndarray, labels2: np.ndarray) -> float:
        """Calculate Jaccard similarity between two clusterings."""
        n = len(labels1)
        pairs_in_same_cluster = 0
        pairs_in_either_cluster = 0

        for i in range(n):
            for j in range(i + 1, n):
                same_cluster1 = labels1[i] == labels1[j]
                same_cluster2 = labels2[i] == labels2[j]

                if same_cluster1 and same_cluster2:
                    pairs_in_same_cluster += 1
                if same_cluster1 or same_cluster2:
                    pairs_in_either_cluster += 1

        if pairs_in_either_cluster == 0:
            return 0.0

        return pairs_in_same_cluster / pairs_in_either_cluster

    def determine_optimal_clusters(self, min_cluster_size: int = 10) -> Dict:
        """
        Determine optimal number of clusters based on multiple criteria.

        Args:
            min_cluster_size: Minimum number of samples per cluster

        Returns:
            Dictionary with optimal cluster recommendations
        """
        logger.info("Determining optimal number of clusters...")

        recommendations = {}

        for method in self.clustering_results:
            method_results = self.clustering_results[method]

            # Find k with highest silhouette score
            silhouette_scores = {k: v['silhouette'] for k, v in method_results.items()}
            optimal_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)

            # Check minimum cluster size constraint
            valid_k_values = []
            for k, results in method_results.items():
                labels = results['labels']
                min_size = min(np.bincount(labels))
                if min_size >= min_cluster_size:
                    valid_k_values.append(k)

            # Get optimal k considering constraints
            if optimal_k_silhouette in valid_k_values:
                optimal_k = optimal_k_silhouette
            else:
                # Find next best k that satisfies constraints
                valid_silhouettes = {k: silhouette_scores[k] for k in valid_k_values}
                optimal_k = max(valid_silhouettes, key=valid_silhouettes.get) if valid_silhouettes else None

            recommendations[method] = {
                'optimal_k': optimal_k,
                'silhouette_score': silhouette_scores.get(optimal_k, None),
                'valid_k_values': valid_k_values
            }

            logger.info(f"{method}: Optimal k={optimal_k}, Silhouette={silhouette_scores.get(optimal_k, 0):.3f}")

        self.optimal_clusters = recommendations
        return recommendations

    def characterize_phenotypes(self, method: str = 'kmeans', k: int = None) -> pd.DataFrame:
        """
        Create statistical characterization of phenotypes.

        Args:
            method: Clustering method to use
            k: Number of clusters (if None, uses optimal)

        Returns:
            DataFrame with phenotype characteristics
        """
        if k is None:
            k = self.optimal_clusters.get(method, {}).get('optimal_k', 3)

        # Try to get labels from instance results first, then from session state
        labels = None
        if method in self.clustering_results and k in self.clustering_results[method]:
            labels = self.clustering_results[method][k]['labels']
        else:
            # Check session state for results
            import streamlit as st
            if (hasattr(st, 'session_state') and
                'phenotype_results' in st.session_state and
                method in st.session_state.phenotype_results and
                'clustering' in st.session_state.phenotype_results[method] and
                k in st.session_state.phenotype_results[method]['clustering']):
                labels = st.session_state.phenotype_results[method]['clustering'][k]['labels']

        if labels is None:
            raise ValueError(f"No clustering results found for method '{method}' with k={k}. Please run clustering first.")

        # Add cluster labels to cleaned data that was actually used for clustering
        data_with_clusters = self.numeric_data.copy()
        data_with_clusters['Phenotype'] = labels

        # Calculate statistics for each phenotype
        phenotype_stats = []

        for phenotype in range(k):
            phenotype_data = data_with_clusters[data_with_clusters['Phenotype'] == phenotype]
            n_samples = len(phenotype_data)

            stats = {'Phenotype': phenotype, 'N_Samples': n_samples}

            # For numeric variables
            numeric_cols = phenotype_data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'Phenotype']

            for col in numeric_cols:
                stats[f'{col}_mean'] = phenotype_data[col].mean()
                stats[f'{col}_std'] = phenotype_data[col].std()

            phenotype_stats.append(stats)

        phenotype_df = pd.DataFrame(phenotype_stats)
        self.phenotype_profiles[f'{method}_k{k}'] = phenotype_df

        return phenotype_df

    def compare_phenotypes(self, method: str = 'kmeans', k: int = None, variables: List[str] = None) -> Dict:
        """
        Statistical comparison between phenotypes.

        Args:
            method: Clustering method
            k: Number of clusters
            variables: List of variables to compare (if None, uses all numeric)

        Returns:
            Dictionary with statistical test results
        """
        if k is None:
            k = self.optimal_clusters.get(method, {}).get('optimal_k', 3)

        # Try to get labels from instance results first, then from session state
        labels = None
        if method in self.clustering_results and k in self.clustering_results[method]:
            labels = self.clustering_results[method][k]['labels']
        else:
            # Check session state for results
            import streamlit as st
            if (hasattr(st, 'session_state') and
                'phenotype_results' in st.session_state and
                method in st.session_state.phenotype_results and
                'clustering' in st.session_state.phenotype_results[method] and
                k in st.session_state.phenotype_results[method]['clustering']):
                labels = st.session_state.phenotype_results[method]['clustering'][k]['labels']

        if labels is None:
            raise ValueError(f"No clustering results found for method '{method}' with k={k}. Please run clustering first.")

        data_with_clusters = self.numeric_data.copy()
        data_with_clusters['Phenotype'] = labels

        if variables is None:
            variables = self.numeric_data.columns.tolist()

        comparison_results = {}

        for var in variables:
            if var not in data_with_clusters.columns:
                continue

            # Group data by phenotype
            groups = [data_with_clusters[data_with_clusters['Phenotype'] == i][var].dropna()
                    for i in range(k)]

            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)

            # Calculate effect size (eta squared)
            ss_between = sum(len(g) * (g.mean() - data_with_clusters[var].mean())**2 for g in groups)
            ss_total = sum((data_with_clusters[var] - data_with_clusters[var].mean())**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            comparison_results[var] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'significant': p_value < 0.05
            }

        return comparison_results

    def plot_phenotype_visualization(self, method: str = 'kmeans', k: int = None, plot_type: str = 'radar') -> plt.Figure:
        """
        Create visualization of phenotype characteristics.

        Args:
            method: Clustering method
            k: Number of clusters
            plot_type: Type of plot ('radar', 'heatmap', 'pca')

        Returns:
            Matplotlib figure
        """
        if k is None:
            k = self.optimal_clusters.get(method, {}).get('optimal_k', 3)

        # Try to get labels from instance results first, then from session state
        labels = None
        if method in self.clustering_results and k in self.clustering_results[method]:
            labels = self.clustering_results[method][k]['labels']

        else:
            # Check session state for results
            import streamlit as st
            if (hasattr(st, 'session_state') and
                'phenotype_results' in st.session_state and
                method in st.session_state.phenotype_results and
                'clustering' in st.session_state.phenotype_results[method] and
                k in st.session_state.phenotype_results[method]['clustering']):

                labels = st.session_state.phenotype_results[method]['clustering'][k]['labels']

        if labels is None:
            raise ValueError(f"No clustering results found for method '{method}' with k={k}. Please run clustering first.")

        if plot_type == 'radar':
            return self._plot_radar_chart(labels, k)
        elif plot_type == 'heatmap':
            return self._plot_heatmap(labels, k)
        elif plot_type == 'pca':
            return self._plot_pca_scatter(labels, k)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def _plot_radar_chart(self, labels: np.ndarray, k: int) -> plt.Figure:
        """Create radar chart of phenotype characteristics."""
        # Select top variables based on ANOVA results
        if hasattr(self, 'comparison_results'):
            # Sort variables by effect size
            sorted_vars = sorted(self.comparison_results.items(),
                               key=lambda x: x[1]['eta_squared'], reverse=True)
            top_vars = [var for var, _ in sorted_vars[:8]]  # Top 8 variables
        else:
            # Use first 8 numeric variables
            top_vars = self.numeric_data.columns[:8].tolist()

        # Calculate mean values for each phenotype
        phenotype_means = []
        for i in range(k):
            mask = labels == i
            means = self.scaled_data[mask][top_vars].mean()
            phenotype_means.append(means)

        # Create radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')

        # Number of variables
        num_vars = len(top_vars)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Plot each phenotype
        colors = plt.cm.Set3(np.linspace(0, 1, k))
        for i, means in enumerate(phenotype_means):
            values = means.tolist() + [means.iloc[0]]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i],
                   label=f'Phenotype {i+1}')
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_vars, size=10)
        ax.set_ylim(-2, 2)  # Standardized scale

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Phenotype Characteristics (Standardized Values)', size=16, y=1.08)

        return fig

    def _plot_heatmap(self, labels: np.ndarray, k: int) -> plt.Figure:
        """Create heatmap of phenotype characteristics."""
        # Calculate z-scores for each phenotype
        phenotype_zscores = []

        for i in range(k):
            mask = labels == i
            phenotype_data = self.numeric_data[mask]

            # Calculate z-scores relative to overall population
            zscores = (phenotype_data.mean() - self.numeric_data.mean()) / self.numeric_data.std()
            phenotype_zscores.append(zscores)

        # Create DataFrame for heatmap
        zscore_df = pd.DataFrame(phenotype_zscores,
                               index=[f'Phenotype {i+1}' for i in range(k)])

        # Select top variables with highest variance across phenotypes
        var_across_phenotypes = zscore_df.var()
        top_vars = var_across_phenotypes.nlargest(20).index

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(zscore_df[top_vars].T, cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Z-score'},
                   xticklabels=True, yticklabels=True, ax=ax)

        plt.title('Phenotype Characteristics (Z-scores)', fontsize=16)
        plt.xlabel('Phenotype', fontsize=12)
        plt.ylabel('Variable', fontsize=12)
        plt.tight_layout()

        return fig

    def _plot_pca_scatter(self, labels: np.ndarray, k: int) -> plt.Figure:
        """Create PCA scatter plot with phenotype coloring."""
        from sklearn.decomposition import PCA

        # Perform PCA for visualization
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(self.scaled_data)

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Set3(np.linspace(0, 1, k))
        for i in range(k):
            mask = labels == i
            ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                      c=[colors[i]], label=f'Phenotype {i+1}',
                      alpha=0.6, s=50)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('Phenotype Distribution in PCA Space', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def export_phenotype_assignments(self, method: str = 'kmeans', k: int = None) -> pd.DataFrame:
        """
        Export phenotype assignments for further analysis.

        Args:
            method: Clustering method
            k: Number of clusters

        Returns:
            DataFrame with original data and phenotype assignments
        """
        if k is None:
            k = self.optimal_clusters.get(method, {}).get('optimal_k', 3)

        # Try to get labels from instance results first, then from session state
        labels = None
        if method in self.clustering_results and k in self.clustering_results[method]:
            labels = self.clustering_results[method][k]['labels']
        else:
            # Check session state for results
            import streamlit as st
            if (hasattr(st, 'session_state') and
                'phenotype_results' in st.session_state and
                method in st.session_state.phenotype_results and
                'clustering' in st.session_state.phenotype_results[method] and
                k in st.session_state.phenotype_results[method]['clustering']):
                labels = st.session_state.phenotype_results[method]['clustering'][k]['labels']

        if labels is None:
            raise ValueError(f"No clustering results found for method '{method}' with k={k}. Please run clustering first.")

        # Create output DataFrame using the numeric data that was actually used for clustering
        # This ensures the labels length matches the data length (after NaN removal)
        output_df = self.numeric_data.copy()
        output_df['Phenotype'] = labels
        output_df['Phenotype_Name'] = output_df['Phenotype'].map(lambda x: f'Phenotype_{x+1}')

        # Add cluster membership probabilities for GMM
        proba = None
        if method == 'gmm':
            # Try to get probabilities from instance results first, then from session state
            if method in self.clustering_results and k in self.clustering_results[method]:
                if 'probabilities' in self.clustering_results[method][k]:
                    proba = self.clustering_results[method][k]['probabilities']
            else:
                # Check session state for probabilities
                import streamlit as st
                if (hasattr(st, 'session_state') and
                    'phenotype_results' in st.session_state and
                    method in st.session_state.phenotype_results and
                    'clustering' in st.session_state.phenotype_results[method] and
                    k in st.session_state.phenotype_results[method]['clustering'] and
                    'probabilities' in st.session_state.phenotype_results[method]['clustering'][k]):
                    proba = st.session_state.phenotype_results[method]['clustering'][k]['probabilities']

            if proba is not None:
                for i in range(k):
                    output_df[f'Phenotype_{i+1}_Probability'] = proba[:, i]

        return output_df
