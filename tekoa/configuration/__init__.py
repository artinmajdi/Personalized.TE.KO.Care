"""
Configuration module for the TE-KOA project.

This module provides configuration classes, parameter definitions, and
settings handling for the TE-KOA dataset analysis.
"""

# Import all classes from params module
from .params import DatasetNames, EnumWithHelpers

# Import settings classes
from .settings import ConfigManager, Settings, setup_logging, logger

# Set up default logging when the package is imported
setup_logging()

__all__ = [
    # Enums and parameter classes
    'EnumWithHelpers',
    'DatasetNames',

    # Settings classes
    'Settings',
    'ConfigManager',
    # Logging
    'logger',
    'setup_logging',
]
