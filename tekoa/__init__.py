"""
TE-KOA-C Dataset Analysis and Visualization

A data science and machine learning framework for nursing research.
"""

# Configure logging
from .configuration.settings import setup_logging, logger

# Set up default logging when the package is imported
setup_logging()


from .io import DataLoader, analyze_dictionary, analyze_excel_file
from .utils import (
    VariableScreener,
    DimensionalityReducer,
    DataQualityEnhancer,
    ClusteringManager,
    characterize_phenotypes,
    compare_variable_across_clusters,
    perform_kmeans,
    perform_pam,
    perform_gmm,
    calculate_silhouette_score,
    calculate_davies_bouldin_score,
    calculate_model_native_score,
    plot_radar_chart,
    generate_radar_chart_data
)


from .visualization import Dashboard, DataManager, pages


# __version__ = "0.1.0"
# __author__ = "Artin Majdi"


__all__ = [
    # IO
    'DataLoader',
    'analyze_dictionary',
    'analyze_excel_file',

    # Utils
    'VariableScreener',
    'DimensionalityReducer',
    'DataQualityEnhancer',
    'ClusteringManager',
    'characterize_phenotypes',
    'compare_variable_across_clusters',
    'perform_kmeans',
    'perform_pam',
    'perform_gmm',
    'calculate_silhouette_score',
    'calculate_davies_bouldin_score',
    'calculate_model_native_score',
    'plot_radar_chart',
    'generate_radar_chart_data',

    # Visualization
    'Dashboard',
    'DataManager',
    'pages',
]
