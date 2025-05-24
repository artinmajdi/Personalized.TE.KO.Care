"""
Page components for the TE-KOA dashboard.

This module provides page rendering components for each section of the TE-KOA dashboard.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from tekoa import logger
from tekoa.visualization.data_manager import DataManager
from tekoa.visualization.ui_utils import create_download_link


class ClusteringPage:
    """Clustering analysis page component for Phase II of the dashboard."""

    @staticmethod
    def render(data_manager: DataManager):
        """Render the clustering analysis page."""
        st.header("Clustering Analysis")

        st.markdown("""
        This page implements clustering algorithms to discover phenotypes in the data.
        You can use K-means, K-medoids (PAM), or Gaussian Mixture Models (GMM) to identify
        natural groupings of patients that may respond differently to treatments.
        """)

        # Check if data is ready for clustering
        if st.session_state.processed_data is None:
            st.warning("Please complete Phase I data preparation before proceeding to clustering.")
            return

        # Initialize phenotype discovery if needed
        use_transformed = st.checkbox(
            "Use dimensionality-reduced data (FAMD/PCA components)",
            value=True,
            help="If checked, uses the transformed data from dimensionality reduction. Otherwise uses processed variables."
        )

        data_manager.initialize_phenotype_discovery(use_transformed_data=use_transformed)

        if data_manager.phenotype_discovery is None:
            st.error("Unable to initialize phenotype discovery. Please check data preparation.")
            return

        # Tab navigation
        tabs = st.tabs(["K-means", "Agglomerative", "Gaussian Mixture Model", "Compare Methods"])

        with tabs[0]:  # K-means
            st.subheader("K-means Clustering")

            st.markdown("""
            K-means is a fast, simple clustering algorithm that partitions data into k clusters
            by minimizing within-cluster variance. It works best with spherical clusters of similar size.
            """)

            col1, col2 = st.columns(2)

            with col1:
                k_range_kmeans = st.slider(
                    "Range of clusters to try",
                    min_value=2,
                    max_value=10,
                    value=(2, 6),
                    key="k_range_kmeans"
                )

            with col2:
                if st.button("Run K-means Clustering", key="run_kmeans"):
                    with st.spinner("Performing K-means clustering..."):
                        results = data_manager.perform_clustering(
                            method='kmeans',
                            n_clusters_range=range(k_range_kmeans[0], k_range_kmeans[1] + 1)
                        )

                        st.success("K-means clustering completed!")

                        # Display results
                        st.subheader("Clustering Results")

                        # Create metrics table
                        metrics_data = []
                        for k, res in results.items():
                            metrics_data.append({
                                'K': k,
                                'Silhouette Score': res['silhouette'],
                                'Calinski-Harabasz': res['calinski'],
                                'Inertia': res['inertia']
                            })

                        metrics_df = pd.DataFrame(metrics_data)

                        # Plot metrics
                        fig = px.line(
                            metrics_df,
                            x='K',
                            y='Silhouette Score',
                            title='Silhouette Score by Number of Clusters',
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Display table
                        st.dataframe(metrics_df.style.format({
                            'Silhouette Score': '{:.3f}',
                            'Calinski-Harabasz': '{:.1f}',
                            'Inertia': '{:.1f}'
                        }))

        with tabs[1]:  # Agglomerative
            st.subheader("Agglomerative Clustering")

            st.markdown("""
            Agglomerative clustering is a bottom-up approach that merges clusters in a bottom-up manner.
            """)

            col1, col2 = st.columns(2)

            with col1:
                k_range_agglomerative = st.slider(
                    "Range of clusters to try",
                    min_value=2,
                    max_value=10,
                    value=(2, 6),
                    key="k_range_agglomerative"
                )

            with col2:
                if st.button("Run Agglomerative Clustering", key="run_agglomerative"):
                    with st.spinner("Performing Agglomerative clustering..."):
                        results = data_manager.perform_clustering(
                            method='agglomerative',
                            n_clusters_range=range(k_range_agglomerative[0], k_range_agglomerative[1] + 1)
                        )

                        st.success("Agglomerative clustering completed!")

                        # Display results
                        st.subheader("Clustering Results")

                        # Create metrics table
                        metrics_data = []
                        for k, res in results.items():
                            metrics_data.append({
                                'K': k,
                                'Silhouette Score': res['silhouette'],
                                'Calinski-Harabasz': res['calinski']
                            })

                        metrics_df = pd.DataFrame(metrics_data)

                        # Plot metrics
                        fig = px.line(
                            metrics_df,
                            x='K',
                            y='Silhouette Score',
                            title='Silhouette Score by Number of Clusters',
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Display table
                        st.dataframe(metrics_df.style.format({
                            'Silhouette Score': '{:.3f}',
                            'Calinski-Harabasz': '{:.1f}'
                        }))

        with tabs[2]:  # GMM
            st.subheader("Gaussian Mixture Model")

            st.markdown("""
            Gaussian Mixture Models provide soft clustering assignments, where each patient has a
            probability of belonging to each cluster. This is useful when clusters overlap.
            """)

            col1, col2 = st.columns(2)

            with col1:
                n_components_range = st.slider(
                    "Range of components to try",
                    min_value=2,
                    max_value=10,
                    value=(2, 6),
                    key="n_components_gmm"
                )

            with col2:
                if st.button("Run GMM Clustering", key="run_gmm"):
                    with st.spinner("Performing GMM clustering..."):
                        results = data_manager.perform_clustering(
                            method='gmm',
                            n_clusters_range=range(n_components_range[0], n_components_range[1] + 1)
                        )

                        st.success("GMM clustering completed!")

                        # Display results
                        st.subheader("Clustering Results")

                        # Create metrics table
                        metrics_data = []
                        for k, res in results.items():
                            metrics_data.append({
                                'Components': k,
                                'Silhouette Score': res['silhouette'],
                                'BIC': res['bic'],
                                'AIC': res['aic']
                            })

                        metrics_df = pd.DataFrame(metrics_data)

                        # Plot BIC/AIC
                        fig = px.line(
                            metrics_df,
                            x='Components',
                            y=['BIC', 'AIC'],
                            title='Model Selection Criteria',
                            labels={'value': 'Score', 'variable': 'Criterion'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Display table
                        st.dataframe(metrics_df.style.format({
                            'Silhouette Score': '{:.3f}',
                            'BIC': '{:.1f}',
                            'AIC': '{:.1f}'
                        }))

        with tabs[3]:  # Compare Methods
            st.subheader("Compare Clustering Methods")

            # Check if results exist
            if 'phenotype_results' not in st.session_state or not st.session_state.phenotype_results:
                st.info("Please run at least one clustering method to see comparisons.")
                return

            # Collect all results
            all_results = []
            for method, method_data in st.session_state.phenotype_results.items():
                if 'clustering' in method_data:
                    for k, res in method_data['clustering'].items():
                        all_results.append({
                            'Method': method.upper(),
                            'K': k,
                            'Silhouette Score': res['silhouette']
                        })

            if all_results:
                comparison_df = pd.DataFrame(all_results)

                # Create comparison plot
                fig = px.line(
                    comparison_df,
                    x='K',
                    y='Silhouette Score',
                    color='Method',
                    title='Silhouette Score Comparison Across Methods',
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)

                # Find best configuration
                best_config = comparison_df.loc[comparison_df['Silhouette Score'].idxmax()]
                st.success(f"Best configuration: {best_config['Method']} with K={best_config['K']} (Silhouette Score: {best_config['Silhouette Score']:.3f})")

                # Display comparison table
                pivot_df = comparison_df.pivot(index='K', columns='Method', values='Silhouette Score')
                st.dataframe(pivot_df.style.format('{:.3f}').highlight_max(axis=1))


class ValidationPage:
    """Cluster validation page component for Phase II of the dashboard."""

    @staticmethod
    def render(data_manager: DataManager):
        """Render the cluster validation page."""
        st.header("Cluster Validation")

        st.markdown("""
        This page provides tools to validate clustering results and determine the optimal number of clusters
        using gap statistic, bootstrap stability assessment, and other validation metrics.
        """)

        # Check if clustering has been performed
        if 'phenotype_results' not in st.session_state or not st.session_state.phenotype_results:
            st.warning("Please perform clustering analysis first.")
            return

        # Initialize phenotype discovery if needed
        data_manager.initialize_phenotype_discovery()

        # Tab navigation
        tabs = st.tabs(["Gap Statistic", "Bootstrap Stability", "Optimal Clusters", "Cluster Quality"])

        with tabs[0]:  # Gap Statistic
            st.subheader("Gap Statistic Analysis")

            st.markdown("""
            The gap statistic compares the within-cluster dispersion with that expected under a
            null reference distribution. The optimal k is where the gap statistic is maximized.
            """)

            col1, col2, col3 = st.columns(3)

            with col1:
                method = st.selectbox(
                    "Select clustering method",
                    options=['kmeans'],  # Currently only implemented for kmeans
                    format_func=lambda x: x.upper()
                )

            with col2:
                k_range = st.slider(
                    "Range of clusters",
                    min_value=2,
                    max_value=10,
                    value=(2, 6)
                )

            with col3:
                n_references = st.slider(
                    "Number of reference datasets",
                    min_value=5,
                    max_value=50,
                    value=10,
                    help="More references give more stable results but take longer"
                )

            if st.button("Calculate Gap Statistic"):
                with st.spinner("Calculating gap statistic..."):
                    gap_results = data_manager.calculate_gap_statistic(
                        method=method,
                        n_clusters_range=range(k_range[0], k_range[1] + 1),
                        n_references=n_references
                    )

                    st.success("Gap statistic calculation completed!")

                    # Plot gap statistic
                    fig = plt.figure(figsize=(10, 6))

                    k_values = gap_results['k_values']
                    gaps = gap_results['gaps']
                    sk_values = gap_results['sk_values']

                    plt.errorbar(k_values, gaps, yerr=sk_values, marker='o', capsize=5)
                    plt.xlabel('Number of Clusters (k)')
                    plt.ylabel('Gap Statistic')
                    plt.title('Gap Statistic by Number of Clusters')
                    plt.grid(True, alpha=0.3)

                    # Mark optimal k
                    optimal_k = gap_results['optimal_k']
                    optimal_idx = k_values.index(optimal_k)
                    plt.scatter(optimal_k, gaps[optimal_idx], color='red', s=200, zorder=5)
                    plt.annotate(f'Optimal k={optimal_k}',
                               xy=(optimal_k, gaps[optimal_idx]),
                               xytext=(optimal_k + 0.5, gaps[optimal_idx]),
                               fontsize=12,
                               color='red')

                    st.pyplot(fig)

                    st.info(f"Optimal number of clusters according to gap statistic: **{optimal_k}**")

        with tabs[1]:  # Bootstrap Stability
            st.subheader("Bootstrap Stability Assessment")

            st.markdown("""
            Bootstrap stability analysis resamples the data many times to assess how stable the
            clustering assignments are. A mean Jaccard index ≥ 0.75 indicates stable clusters.
            """)

            col1, col2 = st.columns(2)

            with col1:
                method = st.selectbox(
                    "Select method",
                    options=['kmeans', 'agglomerative', 'gmm'],
                    format_func=lambda x: x.upper(),
                    key="stability_method"
                )

                # Get available k values for this method
                available_k = []
                if method in st.session_state.phenotype_results:
                    if 'clustering' in st.session_state.phenotype_results[method]:
                        available_k = list(st.session_state.phenotype_results[method]['clustering'].keys())

                if available_k:
                    k = st.selectbox(
                        "Number of clusters",
                        options=sorted(available_k)
                    )
                else:
                    k = 3
                    st.warning(f"No clustering results found for {method.upper()}")

            with col2:
                n_bootstrap = st.slider(
                    "Number of bootstrap samples",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=50
                )

                subsample_size = st.slider(
                    "Subsample size (%)",
                    min_value=50,
                    max_value=90,
                    value=80,
                    step=5
                ) / 100

            if st.button("Assess Stability") and available_k:
                with st.spinner(f"Running {n_bootstrap} bootstrap samples..."):
                    stability_results = data_manager.assess_bootstrap_stability(
                        method=method,
                        k=k,
                        n_bootstrap=n_bootstrap,
                        subsample_size=subsample_size
                    )

                    st.success("Bootstrap stability assessment completed!")

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Mean Jaccard Index", f"{stability_results['mean_jaccard']:.3f}")

                    with col2:
                        st.metric("Std Dev", f"{stability_results['std_jaccard']:.3f}")

                    with col3:
                        stability_status = "✅ Stable" if stability_results['is_stable'] else "❌ Unstable"
                        st.metric("Stability", stability_status)

                    # Interpretation
                    if stability_results['is_stable']:
                        st.success(f"The {method.upper()} clustering with k={k} is stable (Jaccard ≥ 0.75)")
                    else:
                        st.warning(f"The {method.upper()} clustering with k={k} may be unstable (Jaccard < 0.75)")

                    st.info(f"Minimum Jaccard similarity: {stability_results['min_jaccard']:.3f}")

        with tabs[2]:  # Optimal Clusters
            st.subheader("Determine Optimal Number of Clusters")

            st.markdown("""
            This analysis combines multiple criteria to recommend the optimal number of clusters,
            considering silhouette scores, minimum cluster size constraints, and stability.
            """)

            min_cluster_size = st.slider(
                "Minimum cluster size",
                min_value=5,
                max_value=20,
                value=10,
                help="Clusters smaller than this will be considered invalid"
            )

            if st.button("Determine Optimal Clusters"):
                with st.spinner("Analyzing optimal cluster configurations..."):
                    recommendations = data_manager.determine_optimal_clusters(min_cluster_size)

                    st.success("Analysis completed!")

                    # Display recommendations for each method
                    for method, rec in recommendations.items():
                        if rec['optimal_k'] is not None:
                            st.subheader(f"{method.upper()} Recommendations")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Optimal K", rec['optimal_k'])

                            with col2:
                                st.metric("Silhouette Score", f"{rec['silhouette_score']:.3f}")

                            with col3:
                                st.metric("Valid K Values", ", ".join(map(str, rec['valid_k_values'])))
                        else:
                            st.warning(f"No valid clustering found for {method.upper()} with minimum cluster size {min_cluster_size}")

        with tabs[3]:  # Cluster Quality
            st.subheader("Cluster Quality Metrics")

            # Select method and k
            method = st.selectbox(
                "Select method",
                options=['kmeans', 'agglomerative', 'gmm'],
                format_func=lambda x: x.upper(),
                key="quality_method"
            )

            # Get available k values
            available_k = []
            if method in st.session_state.phenotype_results:
                if 'clustering' in st.session_state.phenotype_results[method]:
                    available_k = list(st.session_state.phenotype_results[method]['clustering'].keys())

            if available_k:
                k = st.selectbox(
                    "Number of clusters",
                    options=sorted(available_k),
                    key="quality_k"
                )

                if method in st.session_state.phenotype_results and 'clustering' in st.session_state.phenotype_results[method]:
                    results = st.session_state.phenotype_results[method]['clustering'][k]

                    # Display cluster sizes
                    st.subheader("Cluster Sizes")
                    labels = results['labels']
                    cluster_sizes = pd.Series(labels).value_counts().sort_index()

                    # Create bar chart
                    fig = px.bar(
                        x=cluster_sizes.index,
                        y=cluster_sizes.values,
                        labels={'x': 'Cluster', 'y': 'Number of Patients'},
                        title='Patients per Cluster'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Quality metrics
                    st.subheader("Quality Metrics")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Silhouette Score", f"{results['silhouette']:.3f}")
                        st.markdown("""
                        **Interpretation:**
                        - > 0.7: Strong structure
                        - 0.5-0.7: Reasonable structure
                        - 0.25-0.5: Weak structure
                        - < 0.25: No structure
                        """)

                    with col2:
                        if 'calinski' in results:
                            st.metric("Calinski-Harabasz Index", f"{results['calinski']:.1f}")
                            st.markdown("""
                            **Interpretation:**
                            Higher values indicate better-defined clusters
                            """)

                    # Additional metrics for GMM
                    if method == 'gmm' and 'bic' in results:
                        st.subheader("Model Selection Criteria")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("BIC", f"{results['bic']:.1f}")

                        with col2:
                            st.metric("AIC", f"{results['aic']:.1f}")

                        st.info("Lower BIC/AIC values indicate better model fit")
            else:
                st.warning(f"No clustering results found for {method.upper()}")


class CharacterizationPage:
    """Phenotype characterization page component for Phase II of the dashboard."""

    @staticmethod
    def render(data_manager: DataManager):
        """Render the phenotype characterization page."""
        st.header("Phenotype Characterization")

        st.markdown("""
        This page provides tools to characterize and interpret the discovered phenotypes,
        including statistical comparisons, visualizations, and clinical interpretation.
        """)

        # Check if clustering has been performed
        if 'phenotype_results' not in st.session_state or not st.session_state.phenotype_results:
            st.warning("Please perform clustering analysis first.")
            return

        # Initialize phenotype discovery if needed
        data_manager.initialize_phenotype_discovery()

        # Select method and k
        col1, col2 = st.columns(2)

        with col1:
            # Get methods with results
            available_methods = [m for m in ['kmeans', 'agglomerative', 'gmm']
                               if m in st.session_state.phenotype_results
                               and 'clustering' in st.session_state.phenotype_results[m]]

            if not available_methods:
                st.warning("No clustering results available.")
                return

            method = st.selectbox(
                "Select clustering method",
                options=available_methods,
                format_func=lambda x: x.upper()
            )

        with col2:
            # Get available k values
            available_k = list(st.session_state.phenotype_results[method]['clustering'].keys())

            # Use optimal k if available
            default_k = available_k[0]
            if 'optimal_clusters' in st.session_state.phenotype_results:
                if method in st.session_state.phenotype_results['optimal_clusters']:
                    optimal_k = st.session_state.phenotype_results['optimal_clusters'][method].get('optimal_k')
                    if optimal_k in available_k:
                        default_k = optimal_k

            k = st.selectbox(
                "Number of clusters",
                options=sorted(available_k),
                index=sorted(available_k).index(default_k) if default_k in available_k else 0
            )

        # Tab navigation
        tabs = st.tabs(["Statistical Profile", "Visualizations", "Phenotype Comparison", "Clinical Interpretation"])

        with tabs[0]:  # Statistical Profile
            st.subheader("Statistical Characterization")

            if st.button("Generate Phenotype Profiles"):
                with st.spinner("Characterizing phenotypes..."):
                    char_df = data_manager.characterize_phenotypes(method, k)

                    st.success("Phenotype characterization completed!")

                    # Display basic statistics
                    st.subheader("Phenotype Summary")

                    # Show cluster sizes
                    for idx, row in char_df.iterrows():
                        st.metric(f"Phenotype {row['Phenotype'] + 1}", f"{row['N_Samples']} patients")

                    # Get numeric columns for detailed stats
                    numeric_cols = [col for col in char_df.columns
                                  if col.endswith('_mean') or col.endswith('_std')]

                    if numeric_cols:
                        # Create expandable sections for each phenotype
                        for idx, row in char_df.iterrows():
                            with st.expander(f"Phenotype {row['Phenotype'] + 1} Details"):
                                # Extract variable names and values
                                var_stats = []
                                for col in numeric_cols:
                                    if col.endswith('_mean'):
                                        var_name = col[:-5]  # Remove '_mean'
                                        if f'{var_name}_std' in char_df.columns:
                                            var_stats.append({
                                                'Variable': var_name,
                                                'Mean': row[col],
                                                'Std Dev': row[f'{var_name}_std']
                                            })

                                if var_stats:
                                    stats_df = pd.DataFrame(var_stats)
                                    st.dataframe(stats_df.style.format({
                                        'Mean': '{:.3f}',
                                        'Std Dev': '{:.3f}'
                                    }))

        with tabs[1]:  # Visualizations
            st.subheader("Phenotype Visualizations")

            # Check for clustering results in instance or session state
            has_results = False
            if method in data_manager.phenotype_discovery.clustering_results:
                if k in data_manager.phenotype_discovery.clustering_results[method]:
                    has_results = True
            elif ('phenotype_results' in st.session_state and
                  method in st.session_state.phenotype_results and
                  'clustering' in st.session_state.phenotype_results[method] and
                  k in st.session_state.phenotype_results[method]['clustering']):
                has_results = True

            if has_results:

                    plot_type = st.selectbox(
                        "Select visualization type",
                        options=['radar', 'heatmap', 'pca'],
                        format_func={
                            'radar': 'Radar Chart',
                            'heatmap': 'Z-score Heatmap',
                            'pca': 'PCA Scatter Plot'
                        }.get
                    )

                    if st.button("Generate Visualization"):
                        with st.spinner("Creating visualization..."):
                            try:
                                fig = data_manager.phenotype_discovery.plot_phenotype_visualization(
                                    method=method,
                                    k=k,
                                    plot_type=plot_type
                                )

                                if fig:
                                    st.pyplot(fig)
                                else:
                                    st.error("Unable to generate visualization")
                            except Exception as e:
                                st.error(f"Error creating visualization: {str(e)}")
            else:
                st.info("No clustering results available. Please run clustering first in the 'Clustering' section.")

        with tabs[2]:  # Phenotype Comparison
            st.subheader("Statistical Comparison Between Phenotypes")

            # Variable selection for comparison
            numeric_vars = data_manager.phenotype_discovery.numeric_data.columns.tolist()

            # Group variables by category if available
            variable_categories = data_manager.get_variable_categories()

            if variable_categories:
                selected_vars = []
                for category in ['outcome', 'clinical', 'qst', 'psychological']:
                    if category in variable_categories:
                        category_vars = [v for v in variable_categories[category] if v in numeric_vars]
                        if category_vars:
                            with st.expander(f"{category.capitalize()} Variables"):
                                selected = st.multiselect(
                                    f"Select {category} variables",
                                    options=category_vars,
                                    default=category_vars[:3] if len(category_vars) > 3 else category_vars,
                                    key=f"compare_{category}"
                                )
                                selected_vars.extend(selected)
            else:
                selected_vars = st.multiselect(
                    "Select variables to compare",
                    options=numeric_vars,
                    default=numeric_vars[:5] if len(numeric_vars) > 5 else numeric_vars
                )

            if selected_vars and st.button("Compare Phenotypes"):
                with st.spinner("Performing statistical comparisons..."):
                    comparison_results = data_manager.compare_phenotypes(method, k, selected_vars)

                    # Create results table
                    comparison_data = []
                    for var, stats in comparison_results.items():
                        comparison_data.append({
                            'Variable': var,
                            'F-statistic': stats['f_statistic'],
                            'p-value': stats['p_value'],
                            'Effect Size (η²)': stats['eta_squared'],
                            'Significant': '✓' if stats['significant'] else ''
                        })

                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df = comparison_df.sort_values('p-value')

                    # Display results
                    st.dataframe(comparison_df.style.format({
                        'F-statistic': '{:.2f}',
                        'p-value': '{:.4f}',
                        'Effect Size (η²)': '{:.3f}'
                    }).apply(lambda x: ['background-color: #90EE90' if v else ''
                                       for v in x == '✓'], axis=1))

                    # Summary
                    n_significant = sum(1 for r in comparison_results.values() if r['significant'])
                    st.info(f"Found {n_significant} out of {len(selected_vars)} variables with significant differences between phenotypes (p < 0.05)")

                    # Effect size interpretation
                    st.markdown("""
                    **Effect Size Interpretation (η²):**
                    - Small: 0.01 - 0.06
                    - Medium: 0.06 - 0.14
                    - Large: > 0.14
                    """)

        with tabs[3]:  # Clinical Interpretation
            st.subheader("Clinical Interpretation & Naming")

            # Get phenotype assignments
            phenotype_data = data_manager.export_phenotype_assignments(method, k)

            if phenotype_data is not None:
                st.markdown("### Suggested Phenotype Names")
                st.markdown("""
                Based on the literature, we expect to find phenotypes similar to:
                - **Inflammatory**: High inflammatory markers (CRP, IL-6)
                - **Mechanical/Structural**: High radiographic grade, activity-related pain
                - **Pain Sensitivity**: High pain despite minimal structural damage
                - **Metabolic**: High BMI, systemic inflammation
                - **Minimal Joint Disease**: Low pain, minimal radiographic changes
                """)

                # Allow naming of phenotypes
                st.markdown("### Name Your Phenotypes")

                phenotype_names = {}
                for i in range(k):
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.metric(f"Phenotype {i+1}", f"{sum(phenotype_data['Phenotype'] == i)} patients")

                    with col2:
                        name = st.text_input(
                            f"Name for Phenotype {i+1}",
                            key=f"name_phenotype_{i}",
                            placeholder="e.g., Inflammatory, Mechanical, etc."
                        )
                        if name:
                            phenotype_names[i] = name

                # Save phenotype names
                if phenotype_names and st.button("Save Phenotype Names"):
                    # Update phenotype data with names
                    phenotype_data['Phenotype_Name'] = phenotype_data['Phenotype'].map(
                        lambda x: phenotype_names.get(x, f'Phenotype_{x+1}')
                    )

                    # Update session state
                    st.session_state.phenotypes = phenotype_data
                    st.success("Phenotype names saved!")

                # Export phenotype assignments
                st.markdown("### Export Phenotype Assignments")

                if st.button("Export Phenotype Data"):
                    # Create export data
                    export_df = phenotype_data.copy()

                    # Add patient IDs if available
                    if 'ID' in data_manager.get_original_data().columns:
                        export_df['Patient_ID'] = data_manager.get_original_data()['ID']

                    # Download button
                    csv = export_df.to_csv(index=False)
                    href = create_download_link(
                        csv,
                        f"phenotype_assignments_{method}_k{k}.csv",
                        "Download Phenotype Assignments",
                        "csv"
                    )
                    st.markdown(href, unsafe_allow_html=True)

                    st.success("Phenotype assignments ready for download!")
