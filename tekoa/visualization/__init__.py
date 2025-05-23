"""
Visualization module for the TE-KOA project.

This package provides visualization tools and dashboard components for the TE-KOA-C dataset,
including treatment comparison, longitudinal outcomes, and pain assessment visualizations.
"""

from .app import Dashboard
from .data_manager import DataManager

from .pages import ( HeaderComponent,
                    SidebarComponent,
                    OverviewPage,
                    DataExplorerPage,
                    DictionaryPage,
                    MissingDataPage,
                    ScreeningPage,
                    DimensionalityPage,
                    QualityPage,
                    TreatmentGroupsPage,
                    PipelinePage )

from .ui_utils import apply_custom_css, COLOR_PALETTE, TREATMENT_COLORS, create_download_link, plot_correlation_network

__all__ = [
        'Dashboard',

        # Page components
        'HeaderComponent',
        'SidebarComponent',
        'OverviewPage',
        'DataExplorerPage',
        'DictionaryPage',
        'MissingDataPage',
        'ScreeningPage',
        'DimensionalityPage',
        'QualityPage',
        'TreatmentGroupsPage',
        'PipelinePage',

        # Data manager
        'DataManager',

        # UI utils
        'apply_custom_css',
        'COLOR_PALETTE',
        'TREATMENT_COLORS',
        'create_download_link',
        'plot_correlation_network'
    ]
