"""
TE-KOA-C Dataset Analysis and Visualization

A data science and machine learning framework for nursing research.
"""

# Configure logging
from .configuration.settings import setup_logging, logger

# Set up default logging when the package is imported
setup_logging()


from .io import DataLoader, analyze_dictionary, analyze_excel_file
from .utils import VariableScreener, DimensionalityReducer, DataQualityEnhancer
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

    # Visualization
    'Dashboard',
    'DataManager',
    'pages',
]
