"""
Visualization module for the TE-KOA project.

This package provides visualization tools and dashboard components for the TE-KOA-C dataset,
including treatment comparison, longitudinal outcomes, and pain assessment visualizations.
"""

# from .app2 import Dashboard
from .app import Dashboard

__all__ = [
    'Dashboard',
    # Add other visualization components as they're implemented
    # 'TreatmentComparisonPlot',
    # 'LongitudinalOutcomesPlot',
    # 'PainAssessmentVisualization',
]
