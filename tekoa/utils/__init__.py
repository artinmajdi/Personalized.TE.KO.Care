"""
Utility functions and classes for the TE-KOA project.

This package contains common utilities used throughout the project for data processing,
statistical analysis, and visualization of knee osteoarthritis data.
"""
from .variable_screener import VariableScreener
from .dimensionality_reducer import DimensionalityReducer
from .data_quality_enhancer import DataQualityEnhancer
from .clustering_manager import ClusteringManager
from .phenotype_characterizer import characterize_phenotypes, compare_variable_across_clusters
from .clustering_algorithms import perform_kmeans, perform_pam, perform_gmm
from .clustering_validation import calculate_silhouette_score, calculate_davies_bouldin_score, calculate_model_native_score
from .visualization_utils import plot_radar_chart, generate_radar_chart_data

__all__ = [
    # Variable Screening
    'VariableScreener',

    # Dimensionality Reduction
    'DimensionalityReducer',

    # Data Quality
    'DataQualityEnhancer',

    # Clustering
    'ClusteringManager',

    # Clustering Algorithms
    'perform_kmeans',
    'perform_pam',
    'perform_gmm',

    # Phenotype Characterization
    'characterize_phenotypes',
    'compare_variable_across_clusters',

    # Clustering Validation
    'calculate_silhouette_score',
    'calculate_davies_bouldin_score',
    'calculate_model_native_score',

    # Visualization
    'plot_radar_chart',
    'generate_radar_chart_data'
]
