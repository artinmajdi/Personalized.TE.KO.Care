"""TE-KOA-C Dataset Dashboard Component.

This module provides a specialized dashboard for visualizing and exploring the TE-KOA-C dataset.
"""

import logging
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from te_koa.io.data_loader import DataLoader
from te_koa.configurations.params import DatasetNames

logger = logging.getLogger(__name__)


class Dashboard:
    """Dashboard component for visualizing the TE-KOA-C clinical research dataset."""

    def __init__(self):
        """Initialize the TE-KOA dashboard component."""
        self.data_loader = DataLoader()
        self.dataset_name = DatasetNames.TE_KOA.value
        self.data = None
        self.dictionary = None

    def load_data(self):
        """Load the TE-KOA-C dataset."""
        try:
            self.data, self.dictionary = self.data_loader.load_data()
            return True
        except Exception as e:
            logger.error(f"Error loading TE-KOA-C dataset: {e}")
            st.error(f"Error loading dataset: {e}")
            return False

    def render(self):
        """Render the TE-KOA-C dashboard."""
        st.title("TE-KOA-C Clinical Research Dashboard")

        # Load data if not already loaded
        if self.data is None or self.dictionary is None:
            with st.spinner("Loading dataset..."):
                if not self.load_data():
                    st.stop()

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data Explorer", "Dictionary", "Visualizations"])

        with tab1:
            self._render_overview()

        with tab2:
            self._render_data_explorer()

        with tab3:
            self._render_dictionary()

        with tab4:
            self._render_visualizations()

    def _render_overview(self):
        """Render overview information about the dataset."""
        st.header("Dataset Overview")

        # Display basic information
        st.subheader("Basic Information")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Number of Participants", len(self.data))
            st.metric("Number of Variables", len(self.data.columns))

        with col2:
            # Count missing values
            missing_values = self.data.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
            st.metric("Dictionary Entries", len(self.dictionary))

        # Display data types summary
        st.subheader("Data Types")
        dtypes_df = pd.DataFrame({
            'Data Type': self.data.dtypes.value_counts().index.astype(str),
            'Count': self.data.dtypes.value_counts().values
        })
        st.bar_chart(dtypes_df.set_index('Data Type'))

    def _render_data_explorer(self):
        """Render data exploration tools."""
        st.header("Data Explorer")

        # Column selector
        selected_columns = st.multiselect(
            "Select columns to display",
            options=list(self.data.columns),
            default=list(self.data.columns)[:5]  # Default to first 5 columns
        )

        if selected_columns:
            # Display filtered data
            st.dataframe(self.data[selected_columns])

            # Display statistics for selected columns
            st.subheader("Statistics for Selected Columns")
            numeric_cols = self.data[selected_columns].select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.write(self.data[numeric_cols].describe())
            else:
                st.info("No numeric columns selected for statistics.")
        else:
            st.info("Please select at least one column to display.")

    def _render_dictionary(self):
        """Render the data dictionary."""
        st.header("Data Dictionary")

        # Search functionality
        search_term = st.text_input("Search dictionary", "")

        # Filter dictionary based on search term
        if search_term:
            # Search across all columns
            filtered_dict = self.dictionary[self.dictionary.astype(str).apply(
                lambda row: row.str.contains(search_term, case=False).any(), axis=1
            )]
        else:
            filtered_dict = self.dictionary

        # Display the dictionary
        st.dataframe(filtered_dict)

    def _render_visualizations(self):
        """Render visualizations for the dataset."""
        st.header("Data Visualizations")

        # Only use numeric columns for visualizations
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns available for visualization.")
            return

        # Visualization type selector
        viz_type = st.selectbox(
            "Select visualization type",
            options=["Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap"]
        )

        if viz_type == "Histogram":
            col = st.selectbox("Select column for histogram", options=numeric_cols)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(self.data[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)

        elif viz_type == "Box Plot":
            col = st.selectbox("Select column for box plot", options=numeric_cols)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(y=self.data[col].dropna(), ax=ax)
            ax.set_title(f"Box Plot of {col}")
            st.pyplot(fig)

        elif viz_type == "Scatter Plot":
            col1 = st.selectbox("Select X-axis column", options=numeric_cols)
            col2 = st.selectbox("Select Y-axis column", options=numeric_cols, index=min(1, len(numeric_cols)-1))

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=self.data[col1], y=self.data[col2], ax=ax)
            ax.set_title(f"Scatter Plot: {col1} vs {col2}")
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            st.pyplot(fig)

        elif viz_type == "Correlation Heatmap":
            # Let user select columns for correlation
            selected_cols = st.multiselect(
                "Select columns for correlation heatmap",
                options=numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]  # Default to first 5 numeric columns
            )

            if selected_cols:
                corr = self.data[selected_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)
            else:
                st.info("Please select at least one column for the correlation heatmap.")


def main() -> None:
    """Main entry point for the dashboard application."""
    try:
        # Create and run dashboard
        dashboard = Dashboard()
        dashboard.render()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Dashboard error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
