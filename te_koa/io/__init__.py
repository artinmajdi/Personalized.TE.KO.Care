"""
Data loading and processing modules for the TE-KOA project.

This module provides functionality for loading and preprocessing the TE-KOA-C dataset,
including dictionary parsing, data cleaning, and feature extraction.
"""

from .data_loader import DataLoader
from .analyze_dictionary import analyze_dictionary
from .analyze_excel_file import analyze_excel_file

__all__ = [
    'DataLoader',
    'analyze_dictionary',
    'analyze_excel_file'
]
