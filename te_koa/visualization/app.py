"""TE-KOA-C Dataset Dashboard Component.

This module provides a specialized dashboard for visualizing and exploring the TE-KOA-C dataset.
It includes functionality for data exploration, visualization, imputation, treatment group analysis, and saving processed data.
"""

import logging
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

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
        self.imputed_data = None
        self.treatment_groups = None
        self.missing_data_report = None

    def load_data(self):
        """Load the TE-KOA-C dataset."""
        try:
            self.data, self.dictionary = self.data_loader.load_data()
            self.missing_data_report = self.data_loader.get_missing_data_report()
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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Overview", 
            "Data Explorer", 
            "Dictionary", 
            "Visualizations", 
            "Missing Data & Imputation", 
            "Treatment Groups", 
            "Save Processed Data"
        ])

        with tab1:
            self._render_overview()

        with tab2:
            self._render_data_explorer()

        with tab3:
            self._render_dictionary()

        with tab4:
            self._render_visualizations()
            
        with tab5:
            self._render_missing_data_imputation()
            
        with tab6:
            self._render_treatment_groups()
            
        with tab7:
            self._render_save_processed_data()

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
            missing_percent = (missing_values / (len(self.data) * len(self.data.columns))) * 100
            st.metric("Missing Values", missing_values, f"{missing_percent:.2f}%")
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


    def _render_missing_data_imputation(self):
        """Render missing data analysis and imputation tools."""
        st.header("Missing Data & Imputation")
        
        # Display missing data report
        st.subheader("Missing Data Report")
        if self.missing_data_report is not None:
            # Only show columns with missing values
            missing_report = self.missing_data_report[self.missing_data_report['Missing Values'] > 0]
            if len(missing_report) > 0:
                st.dataframe(missing_report)
            else:
                st.success("No missing values found in the dataset!")
        else:
            st.warning("Missing data report not available.")
        
        # Imputation options
        st.subheader("Impute Missing Values")
        
        col1, col2 = st.columns(2)
        with col1:
            imputation_method = st.selectbox(
                "Select imputation method",
                options=["knn", "mean", "median"],
                index=0
            )
        
        with col2:
            if imputation_method == "knn":
                knn_neighbors = st.slider("Number of neighbors for KNN", 1, 20, 5)
            else:
                knn_neighbors = 5  # Default value, not used for mean/median
        
        # Column exclusion
        st.subheader("Exclude Columns from Imputation")
        cols_to_exclude = st.multiselect(
            "Select columns to exclude from imputation",
            options=list(self.data.columns)
        )
        
        # Impute button
        if st.button("Impute Missing Values"):
            with st.spinner("Imputing missing values..."):
                try:
                    self.imputed_data = self.data_loader.impute_missing_values(
                        method=imputation_method,
                        knn_neighbors=knn_neighbors,
                        cols_to_exclude=cols_to_exclude
                    )
                    st.success("Successfully imputed missing values!")
                    
                    # Show comparison of before/after
                    if self.imputed_data is not None:
                        st.subheader("Before/After Imputation Comparison")
                        # Select a column with missing values to compare
                        missing_cols = [col for col in self.data.columns 
                                       if col not in cols_to_exclude and 
                                       self.data[col].isnull().sum() > 0 and
                                       pd.api.types.is_numeric_dtype(self.data[col])]
                        
                        if missing_cols:
                            compare_col = st.selectbox(
                                "Select column to compare",
                                options=missing_cols
                            )
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Before Imputation")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                sns.histplot(self.data[compare_col].dropna(), kde=True, ax=ax)
                                ax.set_title(f"{compare_col} (Before)")
                                st.pyplot(fig)
                                
                                st.metric(
                                    "Missing Values", 
                                    self.data[compare_col].isnull().sum(),
                                    f"{(self.data[compare_col].isnull().sum() / len(self.data)) * 100:.2f}%"
                                )
                                
                            with col2:
                                st.write("After Imputation")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                sns.histplot(self.imputed_data[compare_col], kde=True, ax=ax)
                                ax.set_title(f"{compare_col} (After)")
                                st.pyplot(fig)
                                
                                st.metric(
                                    "Missing Values", 
                                    self.imputed_data[compare_col].isnull().sum(),
                                    f"{(self.imputed_data[compare_col].isnull().sum() / len(self.imputed_data)) * 100:.2f}%"
                                )
                        else:
                            st.info("No numeric columns with missing values to compare.")
                except Exception as e:
                    st.error(f"Error during imputation: {e}")
                    logger.error(f"Imputation error: {e}", exc_info=True)
        
        # Display imputed data if available
        if self.imputed_data is not None:
            if st.checkbox("Show imputed dataset"):
                st.dataframe(self.imputed_data)
    
    def _render_treatment_groups(self):
        """Render treatment group analysis."""
        st.header("Treatment Groups Analysis")
        
        # Get treatment groups button
        if st.button("Get Treatment Groups"):
            with st.spinner("Analyzing treatment groups..."):
                try:
                    # Use imputed data if available, otherwise use original data
                    if self.imputed_data is not None and st.checkbox("Use imputed data for treatment groups", value=True):
                        # Need to set the data in the data_loader first
                        original_data = self.data_loader.data
                        self.data_loader.data = self.imputed_data
                        self.treatment_groups = self.data_loader.get_treatment_groups()
                        # Restore original data
                        self.data_loader.data = original_data
                    else:
                        self.treatment_groups = self.data_loader.get_treatment_groups()
                    
                    st.success("Successfully analyzed treatment groups!")
                except Exception as e:
                    st.error(f"Error analyzing treatment groups: {e}")
                    logger.error(f"Treatment group analysis error: {e}", exc_info=True)
        
        # Display treatment groups if available
        if self.treatment_groups is not None:
            st.subheader("Treatment Group Summary")
            
            # Display group sizes
            group_sizes = {group: len(df) for group, df in self.treatment_groups.items()}
            group_size_df = pd.DataFrame({
                'Group': list(group_sizes.keys()),
                'Size': list(group_sizes.values())
            })
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.dataframe(group_size_df)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x='Group', y='Size', data=group_size_df, ax=ax)
                ax.set_title("Treatment Group Sizes")
                ax.set_ylabel("Number of Participants")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            # Select a group to explore
            selected_group = st.selectbox(
                "Select a treatment group to explore",
                options=list(self.treatment_groups.keys())
            )
            
            if selected_group:
                st.subheader(f"{selected_group} Group Data")
                st.dataframe(self.treatment_groups[selected_group])
                
                # Analyze outcomes for the selected group
                st.subheader(f"{selected_group} Outcome Analysis")
                
                # Find potential outcome variables (e.g., those with 'outcome', 'result', etc. in name)
                outcome_keywords = ['outcome', 'result', 'score', 'pain', 'womac', 'function']
                potential_outcomes = [col for col in self.treatment_groups[selected_group].columns 
                                    if any(keyword in col.lower() for keyword in outcome_keywords)]
                
                if potential_outcomes:
                    selected_outcome = st.selectbox(
                        "Select an outcome variable",
                        options=potential_outcomes
                    )
                    
                    if selected_outcome and pd.api.types.is_numeric_dtype(self.treatment_groups[selected_group][selected_outcome]):
                        # Calculate statistics
                        outcome_stats = self.treatment_groups[selected_group][selected_outcome].describe()
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write("Outcome Statistics:")
                            st.write(outcome_stats)
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sns.histplot(self.treatment_groups[selected_group][selected_outcome].dropna(), kde=True, ax=ax)
                            ax.set_title(f"{selected_outcome} Distribution for {selected_group}")
                            st.pyplot(fig)
                    else:
                        st.info("Selected outcome is not numeric or contains no data.")
                else:
                    st.info("No potential outcome variables identified.")
                
                # Compare with other groups
                st.subheader("Compare Across Treatment Groups")
                
                # Find common numeric columns across all groups
                common_numeric_cols = []
                for col in self.treatment_groups[selected_group].columns:
                    if all(col in group_df.columns and pd.api.types.is_numeric_dtype(group_df[col]) 
                           for group_df in self.treatment_groups.values()):
                        common_numeric_cols.append(col)
                
                if common_numeric_cols:
                    compare_var = st.selectbox(
                        "Select variable to compare across groups",
                        options=common_numeric_cols
                    )
                    
                    if compare_var:
                        # Create comparison dataframe
                        comparison_data = []
                        for group_name, group_df in self.treatment_groups.items():
                            group_values = group_df[compare_var].dropna()
                            for value in group_values:
                                comparison_data.append({
                                    'Group': group_name,
                                    'Value': value
                                })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(x='Group', y='Value', data=comparison_df, ax=ax)
                        ax.set_title(f"{compare_var} Comparison Across Treatment Groups")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        
                        # ANOVA or statistical test could be added here
                else:
                    st.info("No common numeric variables found across all treatment groups.")
    
    def _render_save_processed_data(self):
        """Render interface for saving processed data."""
        st.header("Save Processed Data")
        
        # Select which dataset to save
        data_to_save = st.radio(
            "Select data to save:",
            options=["Original Data", "Imputed Data", "Treatment Group"],
            index=1 if self.imputed_data is not None else 0
        )
        
        # Determine which data to use
        save_data = None
        if data_to_save == "Original Data":
            save_data = self.data
        elif data_to_save == "Imputed Data":
            if self.imputed_data is not None:
                save_data = self.imputed_data
            else:
                st.warning("Imputed data not available. Please impute data first.")
                return
        elif data_to_save == "Treatment Group":
            if self.treatment_groups is not None:
                selected_group = st.selectbox(
                    "Select treatment group to save",
                    options=list(self.treatment_groups.keys())
                )
                save_data = self.treatment_groups[selected_group]
            else:
                st.warning("Treatment groups not available. Please analyze treatment groups first.")
                return
        
        # File name input
        filename = st.text_input(
            "Enter filename (CSV):",
            value=f"te_koa_{data_to_save.lower().replace(' ', '_')}.csv"
        )
        
        # Additional options
        col1, col2 = st.columns(2)
        with col1:
            include_index = st.checkbox("Include index", value=False)
        with col2:
            include_header = st.checkbox("Include header", value=True)
        
        # Save button
        if st.button("Save Data"):
            if save_data is not None:
                try:
                    # Ensure filename has .csv extension
                    if not filename.endswith('.csv'):
                        filename += '.csv'
                    
                    # Save the data
                    self.data_loader.save_processed_data(
                        df=save_data,
                        filename=filename,
                        index=include_index,
                        header=include_header
                    )
                    
                    st.success(f"Successfully saved data to {filename}!")
                    
                    # Provide download link
                    file_path = self.data_loader.data_dir / filename
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label="Download saved file",
                                data=f,
                                file_name=filename,
                                mime="text/csv"
                            )
                except Exception as e:
                    st.error(f"Error saving data: {e}")
                    logger.error(f"Data saving error: {e}", exc_info=True)
            else:
                st.error("No data available to save.")


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
