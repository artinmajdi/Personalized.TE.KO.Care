"""TE-KOA Dataset Dashboard Component.

This module provides a specialized dashboard for visualizing and exploring the TE-KOA dataset.
It includes functionality for data exploration, visualization, preprocessing, and analysis.
"""

import logging
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from te_koa.io.data_loader import DataLoader
from te_koa.utils.variable_screener import VariableScreener
from te_koa.utils.dimensionality_reducer import DimensionalityReducer
from te_koa.utils.data_quality_enhancer import DataQualityEnhancer

# Set up logging
logger = logging.getLogger(__name__)


class Dashboard:
    """Dashboard component for visualizing the TE-KOA clinical research dataset."""

    def __init__(self):
        """Initialize the TE-KOA dashboard component."""
        self.data_loader = DataLoader()
        self.data = None
        self.dictionary = None
        self.missing_data_report = None

        # Analysis components
        self.variable_screener = None
        self.dimensionality_reducer = None
        self.data_quality_enhancer = None

        # Data states
        self.imputed_data = None
        self.screened_data = None
        self.reduced_data = None
        self.enhanced_data = None
        self.treatment_groups = None

        # Session state initialization
        if 'pipeline_stage' not in st.session_state:
            st.session_state.pipeline_stage = {
                'data_loaded': False,
                'data_imputed': False,
                'variables_screened': False,
                'dimensions_reduced': False,
                'data_quality_enhanced': False,
                'treatment_groups_analyzed': False
            }

    def load_data(self):
        """Load the TE-KOA dataset."""
        try:
            self.data, self.dictionary = self.data_loader.load_data()
            self.missing_data_report = self.data_loader.get_missing_data_report()
            st.session_state.pipeline_stage['data_loaded'] = True
            return True
        except Exception as e:
            logger.error(f"Error loading TE-KOA dataset: {e}")
            st.error(f"Error loading dataset: {e}")
            return False

    def render(self):
        """Render the TE-KOA dashboard."""
        st.set_page_config(page_title="TE-KOA Clinical Research Dashboard",
                           layout="wide",
                           page_icon="🧠")

        # Custom CSS for improved aesthetics
        st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        h1 {
            color: #2c3e50;
        }
        h2 {
            color: #34495e;
        }
        h3 {
            color: #7b8a8b;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f1f3f4;
            border-radius: 4px 4px 0 0;
            padding: 8px 16px;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4e73df !important;
            color: white !important;
        }
        .stButton>button {
            background-color: #4e73df;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 10px 24px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #3a58c5;
        }
        </style>
        """, unsafe_allow_html=True)

        # Header with title and description
        st.title("🧠 TE-KOA Clinical Research Dashboard")
        st.markdown("""
        This dashboard facilitates the analysis of clinical research data related to
        Transcranial Electrical Stimulation (tDCS) for Knee Osteoarthritis (KOA).
        """)

        # Data loading section
        with st.expander("📂 Load Dataset", expanded=not st.session_state.pipeline_stage['data_loaded']):
            if st.button("Load TE-KOA Dataset") or st.session_state.pipeline_stage['data_loaded']:
                with st.spinner("Loading dataset..."):
                    if not st.session_state.pipeline_stage['data_loaded']:
                        if self.load_data():
                            st.success("Dataset loaded successfully!")
                            st.session_state.pipeline_stage['data_loaded'] = True

                    if st.session_state.pipeline_stage['data_loaded']:
                        st.write(f"Dataset loaded with {len(self.data)} rows and {len(self.data.columns)} columns.")

                        # Display basic info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Participants", len(self.data))
                        with col2:
                            st.metric("Variables", len(self.data.columns))
                        with col3:
                            missing = self.data.isnull().sum().sum()
                            st.metric("Missing Values", missing, f"{missing/(len(self.data)*len(self.data.columns))*100:.1f}%")

        # Create tabs for different views
        if st.session_state.pipeline_stage['data_loaded']:
            tabs = st.tabs([
                "Overview",
                "Data Explorer",
                "Variable Screening",
                "Dimensionality Reduction",
                "Data Quality",
                "Treatment Analysis",
                "Export Data"
            ])

            with tabs[0]:
                self._render_overview()

            with tabs[1]:
                self._render_data_explorer()

            with tabs[2]:
                self._render_variable_screening()

            with tabs[3]:
                self._render_dimensionality_reduction()

            with tabs[4]:
                self._render_data_quality()

            with tabs[5]:
                self._render_treatment_analysis()

            with tabs[6]:
                self._render_export_data()

    def _render_overview(self):
        """Render overview information about the dataset."""
        st.header("📊 Dataset Overview")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Dataset Summary")

            # Display data types distribution
            dtypes = self.data.dtypes.value_counts().reset_index()
            dtypes.columns = ['Data Type', 'Count']

            fig = px.bar(dtypes, x='Data Type', y='Count',
                        color='Count', color_continuous_scale='Blues',
                        title='Variable Types Distribution')
            fig.update_layout(xaxis_title="Data Type", yaxis_title="Number of Variables")
            st.plotly_chart(fig, use_container_width=True)

            # Display missing data summary
            if self.missing_data_report is not None:
                missing_data = self.missing_data_report.copy()
                missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values('Percentage', ascending=False).head(15)

                if not missing_data.empty:
                    fig = px.bar(missing_data, x=missing_data.index, y='Percentage',
                                color='Percentage', color_continuous_scale='Reds',
                                title='Top 15 Variables with Missing Data')
                    fig.update_layout(xaxis_title="Variable", yaxis_title="Missing (%)", xaxis={'categoryorder':'total descending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No missing values found in the dataset.")

        with col2:
            st.subheader("Data Dictionary")
            if self.dictionary is not None:
                st.dataframe(self.dictionary.head(10), use_container_width=True)
                st.write(f"Showing 10 of {len(self.dictionary)} dictionary entries")

            st.subheader("Data Pipeline Status")

            # Create a pipeline status indicator
            stages = [
                {"name": "Data Loaded", "status": st.session_state.pipeline_stage['data_loaded']},
                {"name": "Missing Data Imputed", "status": st.session_state.pipeline_stage['data_imputed']},
                {"name": "Variables Screened", "status": st.session_state.pipeline_stage['variables_screened']},
                {"name": "Dimensions Reduced", "status": st.session_state.pipeline_stage['dimensions_reduced']},
                {"name": "Data Quality Enhanced", "status": st.session_state.pipeline_stage['data_quality_enhanced']},
                {"name": "Treatment Analysis", "status": st.session_state.pipeline_stage['treatment_groups_analyzed']}
            ]

            for stage in stages:
                if stage["status"]:
                    st.markdown(f"✅ **{stage['name']}**")
                else:
                    st.markdown(f"⬜ {stage['name']}")

            # Phase I Progress
            completed_stages = sum(1 for stage in stages if stage["status"])
            progress = completed_stages / len(stages)
            st.progress(progress, text=f"Phase I Progress: {progress*100:.0f}%")

    def _render_data_explorer(self):
        """Render data exploration tools."""
        st.header("🔍 Data Explorer")

        col1, col2 = st.columns([1, 3])

        with col1:
            # Data filtering options
            st.subheader("Data Filtering")

            # Column selector
            all_columns = list(self.data.columns)
            default_columns = all_columns[:5] if len(all_columns) > 5 else all_columns

            selected_columns = st.multiselect(
                "Select columns to display",
                options=all_columns,
                default=default_columns
            )

            # Data state selector
            data_state = st.radio(
                "Select data state",
                options=["Original", "Imputed", "Screened", "Reduced", "Enhanced"],
                disabled=[
                    False,
                    not st.session_state.pipeline_stage['data_imputed'],
                    not st.session_state.pipeline_stage['variables_screened'],
                    not st.session_state.pipeline_stage['dimensions_reduced'],
                    not st.session_state.pipeline_stage['data_quality_enhanced']
                ]
            )

            # Get the appropriate data based on selection
            if data_state == "Imputed" and self.imputed_data is not None:
                display_data = self.imputed_data
            elif data_state == "Screened" and self.screened_data is not None:
                display_data = self.screened_data
            elif data_state == "Reduced" and self.reduced_data is not None:
                display_data = self.reduced_data
            elif data_state == "Enhanced" and self.enhanced_data is not None:
                display_data = self.enhanced_data
            else:
                display_data = self.data

            # Search filter
            search_term = st.text_input("Search by value", "")

            # Row limiter
            num_rows = st.slider("Number of rows to display", 5, 100, 20)

        with col2:
            st.subheader("Data Preview")

            # Filter the data based on selections
            if not selected_columns:
                st.warning("Please select at least one column to display.")
                filtered_data = pd.DataFrame()
            else:
                # Filter by selected columns
                available_columns = [col for col in selected_columns if col in display_data.columns]
                filtered_data = display_data[available_columns].copy()

                # Apply search filter if provided
                if search_term:
                    mask = filtered_data.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
                    filtered_data = filtered_data[mask]

                # Limit the number of rows
                filtered_data = filtered_data.head(num_rows)

            # Display the filtered data
            if not filtered_data.empty:
                st.dataframe(filtered_data, use_container_width=True)
                st.write(f"Showing {len(filtered_data)} of {len(display_data)} rows")
            else:
                st.info("No data to display with current filters.")

        # Variable statistics and visualization
        if selected_columns:
            st.subheader("Variable Statistics & Visualization")

            col1, col2 = st.columns([1, 2])

            with col1:
                # Select a column to visualize
                numeric_cols = display_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                categorical_cols = display_data.select_dtypes(include=['object', 'category']).columns.tolist()

                viz_col = st.selectbox(
                    "Select column to visualize",
                    options=selected_columns,
                    index=0 if selected_columns else None
                )

                # Display column statistics
                if viz_col in display_data.columns:
                    st.write("Column Statistics:")

                    if viz_col in numeric_cols:
                        stats = display_data[viz_col].describe()
                        stats_df = pd.DataFrame({
                            'Statistic': stats.index,
                            'Value': stats.values
                        })
                        st.dataframe(stats_df, use_container_width=True)
                    else:
                        # For categorical columns, show value counts
                        value_counts = display_data[viz_col].value_counts()
                        value_counts_df = pd.DataFrame({
                            'Value': value_counts.index,
                            'Count': value_counts.values,
                            'Percentage': (value_counts.values / len(display_data) * 100)
                        })
                        st.dataframe(value_counts_df, use_container_width=True)

                    # Show missing values
                    missing = display_data[viz_col].isnull().sum()
                    st.metric("Missing Values", missing, f"{missing/len(display_data)*100:.1f}%")

            with col2:
                # Visualize the selected column
                if viz_col in display_data.columns:
                    # For numeric columns
                    if viz_col in numeric_cols:
                        viz_type = st.radio(
                            "Select visualization type",
                            options=["Histogram", "Box Plot", "Violin Plot"],
                            horizontal=True
                        )

                        if viz_type == "Histogram":
                            fig = px.histogram(
                                display_data, x=viz_col,
                                marginal="box",
                                title=f"Distribution of {viz_col}",
                                color_discrete_sequence=['#4e73df']
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        elif viz_type == "Box Plot":
                            fig = px.box(
                                display_data, y=viz_col,
                                title=f"Box Plot of {viz_col}",
                                color_discrete_sequence=['#4e73df']
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        elif viz_type == "Violin Plot":
                            fig = px.violin(
                                display_data, y=viz_col,
                                box=True, points="all",
                                title=f"Violin Plot of {viz_col}",
                                color_discrete_sequence=['#4e73df']
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # For categorical columns
                    else:
                        fig = px.bar(
                            display_data[viz_col].value_counts().reset_index(),
                            x='index', y=viz_col,
                            title=f"Value Counts for {viz_col}",
                            color=viz_col,
                            labels={'index': viz_col, viz_col: 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # Variable relationships
        if len(selected_columns) > 1:
            st.subheader("Variable Relationships")

            col1, col2 = st.columns(2)

            with col1:
                # Scatter plot
                numeric_cols = display_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col in selected_columns]

                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("X-axis", options=numeric_cols, index=0)
                    y_col = st.selectbox("Y-axis", options=numeric_cols, index=min(1, len(numeric_cols)-1))

                    color_col = st.selectbox("Color by", options=['None'] + selected_columns, index=0)

                    if x_col and y_col:
                        if color_col != 'None':
                            fig = px.scatter(
                                display_data, x=x_col, y=y_col, color=color_col,
                                title=f"{y_col} vs {x_col} by {color_col}",
                                trendline="ols"
                            )
                        else:
                            fig = px.scatter(
                                display_data, x=x_col, y=y_col,
                                title=f"{y_col} vs {x_col}",
                                trendline="ols"
                            )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least two numeric columns to create a scatter plot.")

            with col2:
                # Correlation heatmap
                numeric_cols = display_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col in selected_columns]

                if len(numeric_cols) >= 2:
                    corr = display_data[numeric_cols].corr()

                    fig = px.imshow(
                        corr.values,
                        x=corr.columns,
                        y=corr.index,
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix"
                    )

                    # Add correlation values as text
                    annotations = []
                    for i, row in enumerate(corr.values):
                        for j, value in enumerate(row):
                            annotations.append(
                                dict(
                                    x=j, y=i,
                                    text=f"{value:.2f}",
                                    showarrow=False,
                                    font=dict(color="white" if abs(value) > 0.5 else "black")
                                )
                            )

                    fig.update_layout(annotations=annotations)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least two numeric columns to create a correlation heatmap.")

    def _render_variable_screening(self):
        """Render variable screening tools."""
        st.header("🧹 Variable Screening")
        st.markdown("""
        This section helps identify variables that may not be useful for analysis,
        including near-zero variance variables and highly collinear variables.
        """)

        # Initialize the variable screener
        data_to_use = self.imputed_data if self.imputed_data is not None else self.data

        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Screening Settings")

            # Near-zero variance threshold
            nzv_threshold = st.slider(
                "Near-zero variance threshold",
                0.01, 0.2, 0.01,
                help="Variables with unique values / total observations < threshold will be flagged"
            )

            # Collinearity threshold
            collinearity_threshold = st.slider(
                "Collinearity threshold",
                0.5, 1.0, 0.85,
                help="Variables with correlation > threshold will be flagged as highly collinear"
            )

            # VIF threshold
            vif_threshold = st.slider(
                "VIF threshold",
                2.0, 10.0, 5.0,
                help="Variables with VIF > threshold may have multicollinearity issues"
            )

            # Force include variables
            force_include = st.multiselect(
                "Force include variables",
                options=data_to_use.columns,
                default=[col for col in data_to_use.columns if 'tdcs' in col.lower() or 'medication' in col.lower()],
                help="These variables will be included regardless of screening results"
            )

            # Screening button
            if st.button("Run Variable Screening"):
                with st.spinner("Screening variables..."):
                    self.variable_screener = VariableScreener(data_to_use)

                    # Run screening
                    self.variable_screener.identify_near_zero_variance(threshold=nzv_threshold)
                    self.variable_screener.analyze_collinearity(threshold=collinearity_threshold)
                    self.variable_screener.calculate_vif(max_vif=vif_threshold)

                    # Get recommendations
                    recommendations = self.variable_screener.recommend_variables(
                        near_zero_threshold=nzv_threshold,
                        collinearity_threshold=collinearity_threshold,
                        vif_threshold=vif_threshold,
                        force_include=force_include
                    )

                    # Create screened dataset
                    self.screened_data = data_to_use[recommendations['variables']].copy()

                    # Update session state
                    st.session_state.pipeline_stage['variables_screened'] = True

                    # Show success message
                    st.success(f"Variable screening complete! Reduced from {len(data_to_use.columns)} to {len(recommendations['variables'])} variables.")

        with col2:
            # Display screening results
            if self.variable_screener is not None and st.session_state.pipeline_stage['variables_screened']:
                # Create tabs for different screening results
                screening_tabs = st.tabs(["Recommendations", "Near-Zero Variance", "Collinearity", "VIF"])

                with screening_tabs[0]:
                    st.subheader("Recommended Variables")

                    # Display variable reduction summary
                    recommendations = self.variable_screener.recommended_vars

                    if recommendations:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Variables", recommendations['explanation']['total_variables'])
                        with col2:
                            st.metric("Recommended Variables", recommendations['explanation']['recommended_variables'])
                        with col3:
                            reduction = recommendations['explanation']['total_variables'] - recommendations['explanation']['recommended_variables']
                            st.metric("Variables Removed", reduction, f"-{reduction/recommendations['explanation']['total_variables']*100:.1f}%")

                        # Display recommended variables
                        st.write("Recommended Variables:")
                        st.write(recommendations['variables'])

                        # Display removed variables by category
                        expander = st.expander("View Removed Variables")
                        with expander:
                            st.write("Near-Zero Variance Variables Removed:")
                            st.write(recommendations['explanation']['near_zero_removed'])

                            st.write("Collinear Variables Removed:")
                            st.write(recommendations['explanation']['collinear_removed'])

                            st.write("High VIF Variables Removed:")
                            st.write(recommendations['explanation']['high_vif_removed'])

                            st.write("Force Included Variables:")
                            st.write(recommendations['explanation']['force_included'])
                    else:
                        st.info("No variable screening results available. Run screening first.")

                with screening_tabs[1]:
                    st.subheader("Near-Zero Variance Variables")

                    if hasattr(self.variable_screener, 'near_zero_vars') and self.variable_screener.near_zero_vars is not None:
                        if not self.variable_screener.near_zero_vars.empty:
                            # Display near-zero variance variables in a table
                            st.dataframe(self.variable_screener.near_zero_vars, use_container_width=True)

                            # Create a bar chart of unique ratio
                            if len(self.variable_screener.near_zero_vars) > 0:
                                fig = px.bar(
                                    self.variable_screener.near_zero_vars,
                                    x='Variable',
                                    y='Unique Ratio',
                                    color='Unique Ratio',
                                    color_continuous_scale='Reds_r',
                                    title="Unique Value Ratio (lower is worse)"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.success("No near-zero variance variables detected!")
                    else:
                        st.info("No near-zero variance analysis available. Run screening first.")

                with screening_tabs[2]:
                    st.subheader("Collinearity Analysis")

                    if hasattr(self.variable_screener, 'high_corr_pairs') and self.variable_screener.high_corr_pairs is not None:
                        if not self.variable_screener.high_corr_pairs.empty:
                            # Display high correlation pairs
                            st.dataframe(self.variable_screener.high_corr_pairs, use_container_width=True)

                            # Display correlation heatmap
                            fig = self.variable_screener.plot_correlation_heatmap(threshold=collinearity_threshold)
                            st.pyplot(fig)
                        else:
                            st.success("No highly collinear variables detected!")
                    else:
                        st.info("No collinearity analysis available. Run screening first.")

                with screening_tabs[3]:
                    st.subheader("Variance Inflation Factor (VIF)")

                    if hasattr(self.variable_screener, 'vif_factors') and self.variable_screener.vif_factors is not None:
                        # Display VIF values
                        st.dataframe(self.variable_screener.vif_factors, use_container_width=True)

                        # Display VIF plot
                        fig = self.variable_screener.plot_vif_factors()
                        st.pyplot(fig)
                    else:
                        st.info("No VIF analysis available. Run screening first.")

    def _render_dimensionality_reduction(self):
        """Render dimensionality reduction tools."""
        st.header("📉 Dimensionality Reduction")
        st.markdown("""
        This section helps reduce the dimensionality of the dataset using methods
        like Principal Component Analysis (PCA) and Factor Analysis of Mixed Data (FAMD).
        """)

        # Get appropriate data to use
        if self.screened_data is not None and st.session_state.pipeline_stage['variables_screened']:
            data_to_use = self.screened_data
        elif self.imputed_data is not None and st.session_state.pipeline_stage['data_imputed']:
            data_to_use = self.imputed_data
        else:
            data_to_use = self.data

        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Reduction Settings")

            # Method selection
            method = st.radio(
                "Reduction Method",
                options=["PCA", "FAMD"],
                help="PCA for numeric data only, FAMD for mixed data types"
            )

            # Number of components
            max_components = min(20, min(data_to_use.shape))
            n_components = st.slider(
                "Maximum Number of Components",
                2, max_components, min(10, max_components),
                help="Maximum number of components to extract"
            )

            # Variance threshold
            variance_threshold = st.slider(
                "Variance Threshold",
                0.5, 1.0, 0.75,
                help="Minimum cumulative explained variance to retain"
            )

            # Categorical columns (for FAMD)
            if method == "FAMD":
                categorical_cols = st.multiselect(
                    "Categorical Columns",
                    options=data_to_use.columns,
                    default=data_to_use.select_dtypes(include=['object', 'category']).columns.tolist(),
                    help="Select categorical columns for FAMD"
                )
            else:
                categorical_cols = None

            # Run dimensionality reduction
            if st.button("Run Dimensionality Reduction"):
                with st.spinner(f"Running {method}..."):
                    # Initialize dimensionality reducer
                    self.dimensionality_reducer = DimensionalityReducer(data_to_use, categorical_cols)

                    # Run reduction
                    if method == "PCA":
                        results = self.dimensionality_reducer.perform_pca(
                            n_components=n_components,
                            variance_threshold=variance_threshold
                        )
                    else:  # FAMD
                        results = self.dimensionality_reducer.perform_famd(
                            n_components=n_components,
                            variance_threshold=variance_threshold
                        )

                    # Get transformed data
                    self.reduced_data = self.dimensionality_reducer.transform_data(method=method.lower())

                    # Update session state
                    st.session_state.pipeline_stage['dimensions_reduced'] = True

                    # Show success message
                    optimal_n = self.dimensionality_reducer.optimal_components
                    st.success(f"{method} completed! Reduced to {optimal_n} components.")

        with col2:
            # Display dimensionality reduction results
            if self.dimensionality_reducer is not None and st.session_state.pipeline_stage['dimensions_reduced']:
                # Create tabs for different reduction results
                reduction_tabs = st.tabs(["Overview", "Scree Plot", "Component Loadings", "Biplot", "Transformed Data"])

                with reduction_tabs[0]:
                    st.subheader("Dimensionality Reduction Overview")

                    # Display variance explained
                    variance_df = self.dimensionality_reducer.get_variance_explained()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Variables", len(data_to_use.columns))
                    with col2:
                        st.metric("Optimal Components", self.dimensionality_reducer.optimal_components)
                    with col3:
                        explained_var = variance_df.iloc[self.dimensionality_reducer.optimal_components-1]['Cumulative Variance']
                        st.metric("Explained Variance", f"{explained_var:.2%}")

                    # Display variance table
                    st.dataframe(variance_df.head(10), use_container_width=True)

                with reduction_tabs[1]:
                    st.subheader("Scree Plot")

                    # Display scree plot
                    fig = self.dimensionality_reducer.plot_scree()
                    st.pyplot(fig)

                with reduction_tabs[2]:
                    st.subheader("Component Loadings")

                    # Display component loadings
                    loadings = self.dimensionality_reducer.get_component_loadings()
                    st.dataframe(loadings, use_container_width=True)

                    # Display component interpretation
                    interpretation = self.dimensionality_reducer.get_component_interpretation()
                    st.subheader("Component Interpretation")
                    st.dataframe(interpretation, use_container_width=True)

                with reduction_tabs[3]:
                    st.subheader("Biplot")

                    # Component selectors
                    col1, col2 = st.columns(2)
                    with col1:
                        pc1 = st.selectbox(
                            "X-axis Component",
                            options=range(1, self.dimensionality_reducer.optimal_components + 1),
                            index=0
                        )
                    with col2:
                        pc2 = st.selectbox(
                            "Y-axis Component",
                            options=range(1, self.dimensionality_reducer.optimal_components + 1),
                            index=1 if self.dimensionality_reducer.optimal_components > 1 else 0
                        )

                    # Display biplot
                    fig = self.dimensionality_reducer.plot_biplot(pc1=pc1, pc2=pc2)
                    st.pyplot(fig)

                with reduction_tabs[4]:
                    st.subheader("Transformed Data")

                    # Display transformed data preview
                    if self.reduced_data is not None:
                        st.dataframe(self.reduced_data.head(10), use_container_width=True)
                        st.write(f"Showing 10 of {len(self.reduced_data)} rows")

                        # Option to add treatment columns
                        if st.checkbox("Add Treatment Columns to Transformed Data"):
                            # Find treatment columns
                            treatment_cols = [col for col in data_to_use.columns
                                            if 'tdcs' in col.lower() or 'medication' in col.lower()]

                            if treatment_cols:
                                # Add treatment columns to transformed data
                                enhanced_reduced_data = self.reduced_data.copy()
                                for col in treatment_cols:
                                    if col in data_to_use.columns:
                                        enhanced_reduced_data[col] = data_to_use[col].values

                                self.reduced_data = enhanced_reduced_data
                                st.success(f"Added treatment columns: {treatment_cols}")
                                st.dataframe(self.reduced_data.head(10), use_container_width=True)
                            else:
                                st.warning("No treatment columns found in the dataset.")

    def _render_data_quality(self):
        """Render data quality enhancement tools."""
        st.header("🔄 Data Quality Enhancement")
        st.markdown("""
        This section helps improve data quality by detecting outliers, analyzing distributions,
        and applying appropriate transformations to variables.
        """)

        # Get appropriate data to use
        if self.reduced_data is not None and st.session_state.pipeline_stage['dimensions_reduced']:
            # If using reduced data, we need to add back any relevant columns
            data_to_use = self.reduced_data
        elif self.screened_data is not None and st.session_state.pipeline_stage['variables_screened']:
            data_to_use = self.screened_data
        elif self.imputed_data is not None and st.session_state.pipeline_stage['data_imputed']:
            data_to_use = self.imputed_data
        else:
            data_to_use = self.data

        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Enhancement Settings")

            # Outlier detection settings
            outlier_method = st.radio(
                "Outlier Detection Method",
                options=["IQR", "Z-Score", "Both"],
                help="Method to detect outliers"
            )

            outlier_threshold = st.slider(
                "Outlier Threshold",
                1.0, 3.0, 1.5,
                help="Threshold for outlier detection (IQR multiplier or Z-score)"
            )

            # Handle transformations
            auto_transform = st.checkbox(
                "Auto-recommend Transformations",
                value=True,
                help="Automatically recommend transformations for non-normal variables"
            )

            # Standardization
            standardize = st.checkbox(
                "Standardize Variables",
                value=True,
                help="Standardize numeric variables (mean=0, std=1)"
            )

            # Process data quality
            if st.button("Enhance Data Quality"):
                with st.spinner("Enhancing data quality..."):
                    # Initialize data quality enhancer
                    self.data_quality_enhancer = DataQualityEnhancer(data_to_use)

                    # Detect outliers
                    self.data_quality_enhancer.detect_outliers(
                        method=outlier_method.lower(),
                        threshold=outlier_threshold
                    )

                    # Analyze distributions
                    self.data_quality_enhancer.analyze_distributions()

                    # Apply transformations if requested
                    if auto_transform:
                        self.data_quality_enhancer.recommend_transformations()
                        self.enhanced_data = self.data_quality_enhancer.apply_transformations()
                    else:
                        self.enhanced_data = data_to_use.copy()

                    # Standardize if requested
                    if standardize:
                        numeric_cols = data_to_use.select_dtypes(include=['float64', 'int64']).columns.tolist()
                        self.enhanced_data = self.data_quality_enhancer.standardize_variables(numeric_cols)

                    # Update session state
                    st.session_state.pipeline_stage['data_quality_enhanced'] = True

                    # Show success message
                    st.success("Data quality enhancement complete!")

        with col2:
            # Display data quality results
            if self.data_quality_enhancer is not None and st.session_state.pipeline_stage['data_quality_enhanced']:
                # Create tabs for different quality results
                quality_tabs = st.tabs(["Outliers", "Distributions", "Transformations", "Enhanced Data"])

                with quality_tabs[0]:
                    st.subheader("Outlier Detection")

                    if hasattr(self.data_quality_enhancer, 'outliers') and self.data_quality_enhancer.outliers:
                        # Select a column to view outliers
                        outlier_cols = list(self.data_quality_enhancer.outliers.keys())

                        if outlier_cols:
                            selected_col = st.selectbox(
                                "Select Column to View Outliers",
                                options=outlier_cols
                            )

                            # Display outlier plot
                            fig = self.data_quality_enhancer.plot_outliers(selected_col, method=outlier_method.lower())
                            if fig:
                                st.pyplot(fig)

                            # Display outlier details
                            outlier_info = self.data_quality_enhancer.outliers[selected_col]

                            for method_key, method_info in outlier_info.items():
                                st.write(f"**{method_key.upper()} Method:**")
                                st.write(f"Number of outliers: {method_info['num_outliers']} ({method_info['percent_outliers']:.2f}%)")

                                if method_info['num_outliers'] > 0:
                                    # Show outlier values
                                    outlier_df = pd.DataFrame({
                                        'Index': method_info['outliers'],
                                        'Value': method_info['outlier_values']
                                    })
                                    st.dataframe(outlier_df, use_container_width=True)
                        else:
                            st.info("No outliers detected in any columns.")
                    else:
                        st.info("No outlier analysis available. Run data quality enhancement first.")

                with quality_tabs[1]:
                    st.subheader("Distribution Analysis")

                    if hasattr(self.data_quality_enhancer, 'distribution_stats') and self.data_quality_enhancer.distribution_stats:
                        # Create summary of distribution statistics
                        dist_summary = []
                        for col, stats in self.data_quality_enhancer.distribution_stats.items():
                            dist_summary.append({
                                'Column': col,
                                'Mean': stats['mean'],
                                'Median': stats['median'],
                                'Skewness': stats['skewness'],
                                'Kurtosis': stats['kurtosis'],
                                'Is Normal': 'Yes' if stats['is_normal'] else 'No' if stats['is_normal'] is not None else 'Unknown'
                            })

                        dist_summary_df = pd.DataFrame(dist_summary)
                        st.dataframe(dist_summary_df, use_container_width=True)

                        # Select a column to view distribution
                        selected_col = st.selectbox(
                            "Select Column to View Distribution",
                            options=list(self.data_quality_enhancer.distribution_stats.keys())
                        )

                        # Display distribution plot
                        fig = self.data_quality_enhancer.plot_distribution(selected_col)
                        if fig:
                            st.pyplot(fig)
                    else:
                        st.info("No distribution analysis available. Run data quality enhancement first.")

                with quality_tabs[2]:
                    st.subheader("Recommended Transformations")

                    if hasattr(self.data_quality_enhancer, 'recommended_transformations') and self.data_quality_enhancer.recommended_transformations:
                        # Create summary of transformation recommendations
                        transform_summary = []
                        for col, info in self.data_quality_enhancer.recommended_transformations.items():
                            transform_summary.append({
                                'Column': col,
                                'Transformation': info['transformation'],
                                'Skewness': info['skewness'],
                                'Reason': info['reason']
                            })

                        transform_summary_df = pd.DataFrame(transform_summary)
                        st.dataframe(transform_summary_df, use_container_width=True)

                        # Count transformations by type
                        transform_counts = transform_summary_df['Transformation'].value_counts().reset_index()
                        transform_counts.columns = ['Transformation', 'Count']

                        fig = px.pie(
                            transform_counts,
                            values='Count',
                            names='Transformation',
                            title='Recommended Transformation Types'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Select a column to view transformation
                        if hasattr(self.data_quality_enhancer, 'transformed_data') and self.data_quality_enhancer.transformed_data is not None:
                            transformed_cols = [col for col in self.data_quality_enhancer.recommended_transformations
                                              if self.data_quality_enhancer.recommended_transformations[col]['transformation'] != 'none']

                            if transformed_cols:
                                selected_col = st.selectbox(
                                    "Select Column to View Transformation",
                                    options=transformed_cols
                                )

                                # Display transformation result
                                fig = self.data_quality_enhancer.plot_distribution(selected_col)
                                if fig:
                                    st.pyplot(fig)
                    else:
                        st.info("No transformation recommendations available. Run data quality enhancement first.")

                with quality_tabs[3]:
                    st.subheader("Enhanced Data Preview")

                    if self.enhanced_data is not None:
                        st.dataframe(self.enhanced_data.head(10), use_container_width=True)
                        st.write(f"Showing 10 of {len(self.enhanced_data)} rows")

                        # Compare original vs enhanced
                        st.subheader("Original vs Enhanced")

                        numeric_cols = data_to_use.select_dtypes(include=['float64', 'int64']).columns
                        common_cols = [col for col in numeric_cols if col in self.enhanced_data.columns]

                        if common_cols:
                            selected_col = st.selectbox(
                                "Select Column to Compare",
                                options=common_cols
                            )

                            if selected_col:
                                # Create comparison DataFrame
                                compare_df = pd.DataFrame({
                                    'Original': data_to_use[selected_col],
                                    'Enhanced': self.enhanced_data[selected_col]
                                })

                                # Create side-by-side histograms
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(
                                    x=compare_df['Original'],
                                    opacity=0.75,
                                    name='Original',
                                    marker_color='blue'
                                ))
                                fig.add_trace(go.Histogram(
                                    x=compare_df['Enhanced'],
                                    opacity=0.75,
                                    name='Enhanced',
                                    marker_color='red'
                                ))

                                fig.update_layout(
                                    title_text=f'Comparison of {selected_col}: Original vs Enhanced',
                                    xaxis_title=selected_col,
                                    yaxis_title='Count',
                                    barmode='overlay'
                                )

                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No common numeric columns to compare.")
                    else:
                        st.info("No enhanced data available. Run data quality enhancement first.")

    def _render_treatment_analysis(self):
        """Render treatment group analysis."""
        st.header("💊 Treatment Group Analysis")
        st.markdown("""
        This section analyzes treatment groups to understand differences in outcomes
        between different interventions.
        """)

        # Get appropriate data to use
        if self.enhanced_data is not None and st.session_state.pipeline_stage['data_quality_enhanced']:
            data_to_use = self.enhanced_data
        elif self.reduced_data is not None and st.session_state.pipeline_stage['dimensions_reduced']:
            data_to_use = self.reduced_data
        elif self.screened_data is not None and st.session_state.pipeline_stage['variables_screened']:
            data_to_use = self.screened_data
        elif self.imputed_data is not None and st.session_state.pipeline_stage['data_imputed']:
            data_to_use = self.imputed_data
        else:
            data_to_use = self.data

        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Treatment Analysis Settings")

            # Data processing options
            data_state = st.radio(
                "Select data state for analysis",
                options=["Original", "Imputed", "Screened", "Reduced", "Enhanced"],
                disabled=[
                    False,
                    not st.session_state.pipeline_stage['data_imputed'],
                    not st.session_state.pipeline_stage['variables_screened'],
                    not st.session_state.pipeline_stage['dimensions_reduced'],
                    not st.session_state.pipeline_stage['data_quality_enhanced']
                ]
            )

            # Get the appropriate data based on selection
            if data_state == "Imputed" and self.imputed_data is not None:
                data_for_groups = self.imputed_data
            elif data_state == "Screened" and self.screened_data is not None:
                data_for_groups = self.screened_data
            elif data_state == "Reduced" and self.reduced_data is not None:
                data_for_groups = self.reduced_data
            elif data_state == "Enhanced" and self.enhanced_data is not None:
                data_for_groups = self.enhanced_data
            else:
                data_for_groups = self.data

            # Analyze treatment groups button
            if st.button("Analyze Treatment Groups"):
                with st.spinner("Analyzing treatment groups..."):
                    # Set data temporarily in data_loader
                    original_data = self.data_loader.data
                    self.data_loader.data = data_for_groups

                    # Get treatment groups
                    self.treatment_groups = self.data_loader.get_treatment_groups()

                    # Restore original data
                    self.data_loader.data = original_data

                    # Update session state
                    st.session_state.pipeline_stage['treatment_groups_analyzed'] = True

                    # Show success message
                    if self.treatment_groups:
                        group_counts = {group: len(df) for group, df in self.treatment_groups.items()}
                        st.success(f"Treatment groups analyzed: {group_counts}")
                    else:
                        st.error("Could not identify treatment groups. Check that treatment columns exist.")

        with col2:
            # Display treatment group results
            if self.treatment_groups is not None and st.session_state.pipeline_stage['treatment_groups_analyzed']:
                # Create tabs for different treatment analysis views
                treatment_tabs = st.tabs(["Group Overview", "Outcome Analysis", "Group Comparison", "Variable Analysis"])

                with treatment_tabs[0]:
                    st.subheader("Treatment Group Overview")

                    # Display group sizes
                    group_sizes = {group: len(df) for group, df in self.treatment_groups.items()}
                    group_size_df = pd.DataFrame({
                        'Group': list(group_sizes.keys()),
                        'Size': list(group_sizes.values())
                    })

                    # Plot group sizes
                    fig = px.bar(
                        group_size_df,
                        x='Group',
                        y='Size',
                        color='Group',
                        title='Treatment Group Sizes'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display group summary statistics
                    st.write("Group Summary Statistics:")

                    # Get common numeric columns
                    numeric_cols = data_for_groups.select_dtypes(include=['float64', 'int64']).columns

                    # Allow user to select variables to summarize
                    selected_vars = st.multiselect(
                        "Select variables to summarize",
                        options=numeric_cols,
                        default=[col for col in numeric_cols if 'baseline' in col.lower() or 'outcome' in col.lower()][:3]
                    )

                    if selected_vars:
                        # Create summary statistics for each group
                        group_stats = []

                        for group, df in self.treatment_groups.items():
                            for var in selected_vars:
                                if var in df.columns:
                                    stats = df[var].describe()
                                    group_stats.append({
                                        'Group': group,
                                        'Variable': var,
                                        'Mean': stats['mean'],
                                        'Std': stats['std'],
                                        'Min': stats['min'],
                                        'Median': stats['50%'],
                                        'Max': stats['max'],
                                        'N': stats['count']
                                    })

                        group_stats_df = pd.DataFrame(group_stats)
                        st.dataframe(group_stats_df, use_container_width=True)

                with treatment_tabs[1]:
                    st.subheader("Outcome Analysis")

                    # Identify potential outcome variables
                    outcome_keywords = ['outcome', 'result', 'score', 'pain', 'womac', 'function', 'follow', '6m', '12m']
                    potential_outcomes = [col for col in data_for_groups.columns
                                         if any(keyword in col.lower() for keyword in outcome_keywords)]

                    col1, col2 = st.columns(2)

                    with col1:
                        # Select outcome variable
                        outcome_var = st.selectbox(
                            "Select outcome variable",
                            options=potential_outcomes if potential_outcomes else data_for_groups.columns,
                            index=0 if potential_outcomes else None
                        )

                    with col2:
                        # Select baseline variable (if available)
                        baseline_vars = [col for col in data_for_groups.columns if 'baseline' in col.lower()]

                        if baseline_vars:
                            baseline_var = st.selectbox(
                                "Select baseline variable",
                                options=['None'] + baseline_vars,
                                index=0
                            )
                        else:
                            baseline_var = 'None'

                    if outcome_var:
                        # Check if outcome variable exists in all groups
                        if all(outcome_var in df.columns for df in self.treatment_groups.values()):
                            # Prepare data for visualization
                            outcome_data = []

                            for group, df in self.treatment_groups.items():
                                for _, row in df.iterrows():
                                    data_point = {'Group': group}

                                    if outcome_var in row:
                                        data_point['Outcome'] = row[outcome_var]

                                        if baseline_var != 'None' and baseline_var in row:
                                            data_point['Baseline'] = row[baseline_var]
                                            data_point['Change'] = row[outcome_var] - row[baseline_var]

                                        outcome_data.append(data_point)

                            outcome_df = pd.DataFrame(outcome_data)

                            # Create visualizations
                            if not outcome_df.empty:
                                # Box plot of outcomes by group
                                fig = px.box(
                                    outcome_df,
                                    x='Group',
                                    y='Outcome',
                                    color='Group',
                                    title=f'{outcome_var} by Treatment Group',
                                    points='all'
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # If baseline is available, show change
                                if baseline_var != 'None' and 'Change' in outcome_df.columns:
                                    fig = px.box(
                                        outcome_df,
                                        x='Group',
                                        y='Change',
                                        color='Group',
                                        title=f'Change in {outcome_var} from {baseline_var}',
                                        points='all'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Calculate mean change for each group
                                    group_changes = outcome_df.groupby('Group')['Change'].agg(['mean', 'std', 'count']).reset_index()

                                    # Calculate standard error
                                    group_changes['se'] = group_changes['std'] / np.sqrt(group_changes['count'])

                                    # Create bar chart with error bars
                                    fig = px.bar(
                                        group_changes,
                                        x='Group',
                                        y='mean',
                                        color='Group',
                                        error_y='se',
                                        title=f'Mean Change in {outcome_var} from {baseline_var}',
                                        labels={'mean': 'Mean Change'}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Variable '{outcome_var}' not found in all treatment groups.")

                with treatment_tabs[2]:
                    st.subheader("Group Comparison")

                    # Allow comparison of any two groups
                    groups = list(self.treatment_groups.keys())

                    col1, col2 = st.columns(2)

                    with col1:
                        group1 = st.selectbox(
                            "Select first group",
                            options=groups,
                            index=0 if groups else None
                        )

                    with col2:
                        remaining_groups = [g for g in groups if g != group1]
                        group2 = st.selectbox(
                            "Select second group",
                            options=remaining_groups,
                            index=0 if remaining_groups else None
                        )

                    if group1 and group2:
                        # Get data for selected groups
                        df1 = self.treatment_groups[group1]
                        df2 = self.treatment_groups[group2]

                        # Find common numeric columns
                        numeric_cols1 = df1.select_dtypes(include=['float64', 'int64']).columns
                        numeric_cols2 = df2.select_dtypes(include=['float64', 'int64']).columns
                        common_cols = [col for col in numeric_cols1 if col in numeric_cols2]

                        # Allow user to select variables to compare
                        selected_vars = st.multiselect(
                            "Select variables to compare",
                            options=common_cols,
                            default=[col for col in common_cols
                                    if any(keyword in col.lower() for keyword in
                                          ['outcome', 'pain', 'womac', 'function'])][:3]
                        )

                        if selected_vars:
                            # Create comparison statistics
                            comparison_stats = []

                            for var in selected_vars:
                                # Group 1 stats
                                mean1 = df1[var].mean()
                                std1 = df1[var].std()
                                n1 = df1[var].count()

                                # Group 2 stats
                                mean2 = df2[var].mean()
                                std2 = df2[var].std()
                                n2 = df2[var].count()

                                # Calculate difference
                                diff = mean2 - mean1

                                # Calculate pooled standard deviation for Cohen's d
                                pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

                                # Calculate Cohen's d (effect size)
                                cohens_d = diff / pooled_std if pooled_std != 0 else 0

                                comparison_stats.append({
                                    'Variable': var,
                                    f'{group1} Mean': mean1,
                                    f'{group1} Std': std1,
                                    f'{group1} N': n1,
                                    f'{group2} Mean': mean2,
                                    f'{group2} Std': std2,
                                    f'{group2} N': n2,
                                    'Difference': diff,
                                    "Cohen's d": cohens_d
                                })

                            comparison_df = pd.DataFrame(comparison_stats)
                            st.dataframe(comparison_df, use_container_width=True)

                            # Create comparison visualizations
                            for var in selected_vars:
                                # Create data for violin plot
                                plot_data = pd.DataFrame({
                                    'Group': [group1] * len(df1) + [group2] * len(df2),
                                    'Value': list(df1[var]) + list(df2[var])
                                })

                                fig = px.violin(
                                    plot_data,
                                    x='Group',
                                    y='Value',
                                    color='Group',
                                    box=True,
                                    points='all',
                                    title=f'Comparison of {var}'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Please select variables to compare.")

                with treatment_tabs[3]:
                    st.subheader("Variable Analysis Across Groups")

                    # Allow analysis of variables across all groups
                    all_cols = data_for_groups.columns

                    # Select variable to analyze
                    analyze_var = st.selectbox(
                        "Select variable to analyze across groups",
                        options=all_cols,
                        index=0 if all_cols.size > 0 else None
                    )

                    if analyze_var:
                        # Check if variable exists in all groups
                        if all(analyze_var in df.columns for df in self.treatment_groups.values()):
                            # Create analysis data
                            analysis_data = []

                            for group, df in self.treatment_groups.items():
                                for _, row in df.iterrows():
                                    if analyze_var in row:
                                        analysis_data.append({
                                            'Group': group,
                                            'Value': row[analyze_var]
                                        })

                            analysis_df = pd.DataFrame(analysis_data)

                            # Create visualizations
                            if not analysis_df.empty:
                                # Determine visualization based on data type
                                if pd.api.types.is_numeric_dtype(analysis_df['Value']):
                                    # Numeric variable

                                    # Create histogram by group
                                    fig = px.histogram(
                                        analysis_df,
                                        x='Value',
                                        color='Group',
                                        marginal='box',
                                        barmode='overlay',
                                        opacity=0.7,
                                        title=f'Distribution of {analyze_var} by Treatment Group'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Create box plot
                                    fig = px.box(
                                        analysis_df,
                                        x='Group',
                                        y='Value',
                                        color='Group',
                                        title=f'{analyze_var} by Treatment Group',
                                        points='all'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Calculate statistics
                                    stats = analysis_df.groupby('Group')['Value'].agg(['mean', 'std', 'count']).reset_index()
                                    stats['se'] = stats['std'] / np.sqrt(stats['count'])

                                    # Create bar chart with error bars
                                    fig = px.bar(
                                        stats,
                                        x='Group',
                                        y='mean',
                                        color='Group',
                                        error_y='se',
                                        title=f'Mean {analyze_var} by Treatment Group',
                                        labels={'mean': f'Mean {analyze_var}'}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    # Categorical variable

                                    # Create contingency table
                                    contingency = pd.crosstab(analysis_df['Group'], analysis_df['Value'])

                                    # Convert to percentage
                                    contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100

                                    # Create heatmap
                                    fig = px.imshow(
                                        contingency_pct.values,
                                        x=contingency_pct.columns,
                                        y=contingency_pct.index,
                                        color_continuous_scale='Blues',
                                        labels=dict(x='Value', y='Group', color='Percentage'),
                                        title=f'Distribution of {analyze_var} by Treatment Group (%)'
                                    )

                                    # Add text annotations
                                    annotations = []
                                    for i, row in enumerate(contingency_pct.values):
                                        for j, value in enumerate(row):
                                            count = contingency.values[i, j]
                                            annotations.append(
                                                dict(
                                                    x=j, y=i,
                                                    text=f"{value:.1f}%<br>({count})",
                                                    showarrow=False,
                                                    font=dict(color="white" if value > 50 else "black")
                                                )
                                            )

                                    fig.update_layout(annotations=annotations)
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Create bar chart
                                    contingency_long = contingency.reset_index().melt(
                                        id_vars='Group',
                                        var_name='Value',
                                        value_name='Count'
                                    )

                                    fig = px.bar(
                                        contingency_long,
                                        x='Group',
                                        y='Count',
                                        color='Value',
                                        barmode='group',
                                        title=f'Count of {analyze_var} by Treatment Group'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Variable '{analyze_var}' not found in all treatment groups.")

    def _render_export_data(self):
        """Render interface for exporting processed data."""
        st.header("💾 Export Processed Data")
        st.markdown("""
        This section allows you to export the processed data for further analysis or use in other applications.
        """)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Export Settings")

            # Data state selector
            data_state = st.radio(
                "Select data to export",
                options=["Original", "Imputed", "Screened", "Reduced", "Enhanced", "Treatment Groups"],
                disabled=[
                    False,
                    not st.session_state.pipeline_stage['data_imputed'],
                    not st.session_state.pipeline_stage['variables_screened'],
                    not st.session_state.pipeline_stage['dimensions_reduced'],
                    not st.session_state.pipeline_stage['data_quality_enhanced'],
                    not st.session_state.pipeline_stage['treatment_groups_analyzed']
                ]
            )

            # Treatment group selector if applicable
            if data_state == "Treatment Groups" and st.session_state.pipeline_stage['treatment_groups_analyzed']:
                if self.treatment_groups:
                    treatment_group = st.selectbox(
                        "Select treatment group to export",
                        options=list(self.treatment_groups.keys())
                    )
                else:
                    st.warning("No treatment groups available.")
                    treatment_group = None
            else:
                treatment_group = None

            # File format
            file_format = st.radio(
                "Select export format",
                options=["CSV", "Excel"],
                horizontal=True
            )

            # Filename
            default_filename = f"te_koa_{data_state.lower()}"
            if treatment_group:
                default_filename += f"_{treatment_group.replace(' ', '_').lower()}"
            default_filename += ".csv" if file_format == "CSV" else ".xlsx"

            filename = st.text_input(
                "Enter filename",
                value=default_filename
            )

            # Export options
            col1, col2 = st.columns(2)
            with col1:
                include_index = st.checkbox("Include index", value=False)
            with col2:
                include_header = st.checkbox("Include header", value=True)

            # Export button
            export_clicked = st.button("Export Data")

        with col2:
            st.subheader("Data Preview")

            # Get the data to export
            if data_state == "Imputed" and self.imputed_data is not None:
                export_data = self.imputed_data
            elif data_state == "Screened" and self.screened_data is not None:
                export_data = self.screened_data
            elif data_state == "Reduced" and self.reduced_data is not None:
                export_data = self.reduced_data
            elif data_state == "Enhanced" and self.enhanced_data is not None:
                export_data = self.enhanced_data
            elif data_state == "Treatment Groups" and self.treatment_groups is not None and treatment_group:
                export_data = self.treatment_groups[treatment_group]
            else:
                export_data = self.data

            # Show data preview
            if export_data is not None:
                st.dataframe(export_data.head(10), use_container_width=True)
                st.write(f"Showing 10 of {len(export_data)} rows")

            # Export data if button clicked
            if export_clicked and export_data is not None:
                try:
                    if file_format == "CSV":
                        # Export as CSV
                        csv = export_data.to_csv(index=include_index, header=include_header)

                        # Provide download button
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=filename,
                            mime="text/csv"
                        )

                        st.success(f"Data exported as CSV with {len(export_data)} rows and {len(export_data.columns)} columns.")
                    else:
                        # Export as Excel
                        # First save to a temporary file
                        temp_file = f"temp_{filename}"
                        export_data.to_excel(temp_file, index=include_index, header=include_header)

                        # Read the file and provide download button
                        with open(temp_file, "rb") as f:
                            excel_data = f.read()

                        st.download_button(
                            label="Download Excel",
                            data=excel_data,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                        # Clean up the temporary file
                        import os
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                        st.success(f"Data exported as Excel with {len(export_data)} rows and {len(export_data.columns)} columns.")
                except Exception as e:
                    st.error(f"Error exporting data: {e}")


def main() -> None:
    """Main entry point for the dashboard application."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create and run dashboard
        dashboard = Dashboard()
        dashboard.render()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Dashboard error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
