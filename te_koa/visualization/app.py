"""
Enhanced TE-KOA-C Dataset Dashboard.

This module provides a comprehensive dashboard for the TE-KOA-C dataset,
implementing all Phase I components including:
- Data loading and exploration
- Missing data analysis and imputation
- Variable screening
- Dimensionality reduction
- Data quality enhancement
- Treatment group analysis
"""

import logging
import os
from typing import Optional
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorcet as cc
import time
import networkx as nx
from wordcloud import WordCloud
from scipy import stats as scipy_stats
import json

from te_koa.io.data_loader import DataLoader
from te_koa.utils.variable_screener import VariableScreener
from te_koa.utils.dimensionality_reducer import DimensionalityReducer
from te_koa.utils.data_quality_enhancer import DataQualityEnhancer
from te_koa.configurations.params import DatasetNames

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define app configuration
APP_TITLE = "TE-KOA Clinical Research Dashboard"
APP_SUBTITLE = "Phenotyping and Heterogeneity of Treatment Effects in Knee Osteoarthritis"
PHASE_TITLE = "Phase I: Data Preparation"

# Define color scheme
COLOR_PALETTE = {
	'primary': '#4472C4',  # Blue
	'secondary': '#ED7D31',  # Orange
	'tertiary': '#A5A5A5',  # Gray
	'success': '#70AD47',  # Green
	'warning': '#FFC000',  # Yellow
	'danger': '#FF0000',  # Red
	'highlight': '#5B9BD5',  # Light blue
	'background': '#F5F5F5',  # Light gray
	'text': '#333333'  # Dark gray
}

# Treatment group colors
TREATMENT_COLORS = {
	'Control (Sham)': '#90CAF9',  # Light blue
	'Experimental': '#FF8A65',  # Light orange
	'tDCS': '#81C784',  # Light green
	'Meditation': '#E1BEE7',  # Light purple
	'Control (No tDCS, No Meditation)': '#9FA8DA',  # Indigo
	'tDCS Only': '#A5D6A7',  # Green
	'Meditation Only': '#CE93D8',  # Purple
	'tDCS + Meditation': '#FFAB91'  # Deep orange
}


class Dashboard:
	"""Enhanced dashboard for the TE-KOA-C clinical research dataset."""

	def __init__(self):
		"""Initialize the TE-KOA dashboard component."""
		self.data_loader: Optional[DataLoader] = None
		self.dataset_name = DatasetNames.TE_KOA.value
		self.data = None
		self.dictionary = None
		self.imputed_data = None
		self.treatment_groups = None
		self.missing_data_report = None

		# Phase I components
		self.variable_screener = None
		self.dimensionality_reducer = None
		self.data_quality_enhancer = None

		# Session state initialization
		if 'processed_data' not in st.session_state:
			st.session_state.processed_data = None
		if 'pipeline_results' not in st.session_state:
			st.session_state.pipeline_results = {}
		if 'phenotypes' not in st.session_state:
			st.session_state.phenotypes = None

	def load_data(self):
		"""Load the TE-KOA-C dataset."""

		# Check if file is uploaded in session state
		if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
			uploaded_file = st.session_state.uploaded_file
			logger.info(f"Loading data from uploaded file: {uploaded_file.name}")

			# Create DataLoader with uploaded file
			self.data_loader = DataLoader(uploaded_file=uploaded_file)

		if self.data_loader is None:
			return False


		# Load data using the data loader
		self.data, self.dictionary = self.data_loader.load_data()
		self.missing_data_report = self.data_loader.get_missing_data_report()

		# Initialize Phase I components if data loaded successfully
		if self.data is not None:
			st.session_state.processed_data = self.data.copy()
			st.session_state.data_loaded_time = pd.Timestamp.now()

			logger.info(f"Successfully loaded data: {len(self.data)} rows, {len(self.data.columns)} columns")

			# Check for required treatment column
			if 'tx.group' not in self.data.columns:
				st.warning("Dataset doesn't contain the expected 'tx.group' column for treatment groups. Some functionality may be limited.")
				logger.warning("tx.group column not found in dataset")

			return True



	def _analyze_missing_data(self) -> pd.DataFrame:
		"""
		Analyze missing data in the dataset.

		Returns:
			DataFrame containing missing data statistics
		"""
		if self.data is None:
			return None

		# Calculate missing values statistics
		missing = self.data.isnull().sum()
		missing_percent = (self.data.isnull().sum() / len(self.data)) * 100
		data_types = self.data.dtypes

		# Create a report
		missing_data_report = pd.DataFrame({
			'Missing Values': missing,
			'Percentage': missing_percent,
			'Data Type': data_types
		})

		# Sort by percentage of missing values, descending
		missing_data_report = missing_data_report.sort_values('Percentage', ascending=False)

		return missing_data_report

	def render(self):
		"""Render the TE-KOA-C dashboard."""
		# Set page config
		st.set_page_config(
			page_title=APP_TITLE,
			page_icon="🦵",
			layout="wide",
			initial_sidebar_state="expanded"
		)

		# Apply custom CSS
		self._apply_custom_css()

		# Display header
		self._render_header()

		# Sidebar navigation
		self._render_sidebar()

		# Load data if not already loaded
		if self.data is None or self.dictionary is None:
			with st.spinner("Loading dataset..."):
				# Check if we have an uploaded file first
				if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
					if not self.load_data():
						st.stop()
					st.success(f"Loaded dataset from uploaded file: {st.session_state.uploaded_file.name}")
				else:
					# Try to load from default path
					if not self.load_data():
						# Show instructions for uploading file if loading fails
						st.error("Could not load dataset from default path.")
						st.info("Please upload an Excel file using the sidebar. The file should have two sheets: 'Sheet1' with data and 'dictionary' with variable descriptions.")
						st.stop()

		# Get current page from session state
		current_page = st.session_state.get('current_page', 'Overview')

		# Render current page
		if current_page == 'Overview':
			self._render_overview()
		elif current_page == 'Data Explorer':
			self._render_data_explorer()
		elif current_page == 'Data Dictionary':
			self._render_dictionary()
		elif current_page == 'Missing Data & Imputation':
			self._render_missing_data_imputation()
		elif current_page == 'Variable Screening':
			self._render_variable_screening()
		elif current_page == 'Dimensionality Reduction':
			self._render_dimensionality_reduction()
		elif current_page == 'Data Quality':
			self._render_data_quality()
		elif current_page == 'Treatment Groups':
			self._render_treatment_groups()
		elif current_page == 'Pipeline & Export':
			self._render_pipeline_export()
		else:
			st.warning(f"Unknown page: {current_page}")

	def _apply_custom_css(self):
		"""Apply custom CSS styles to the dashboard."""
		st.markdown("""
		<style>
		.main-header {
			background-color: #4472C4;
			padding: 1rem;
			color: white;
			border-radius: 5px;
			margin-bottom: 1rem;
		}
		.phase-indicator {
			background-color: #ED7D31;
			padding: 0.3rem 0.6rem;
			color: white;
			border-radius: 3px;
			font-size: 0.8rem;
			margin-bottom: 0.5rem;
			display: inline-block;
		}
		.section-header {
			background-color: #F5F5F5;
			padding: 0.5rem;
			border-left: 4px solid #4472C4;
			margin-top: 1rem;
			margin-bottom: 1rem;
		}
		.card {
			background-color: white;
			padding: 1rem;
			border-radius: 5px;
			box-shadow: 0 0 5px rgba(0,0,0,0.1);
			margin-bottom: 1rem;
		}
		.metric-container {
			display: flex;
			flex-wrap: wrap;
			gap: 10px;
		}
		.metric-card {
			background-color: white;
			padding: 1rem;
			border-radius: 5px;
			box-shadow: 0 0 5px rgba(0,0,0,0.1);
			flex: 1;
			min-width: 200px;
			text-align: center;
		}
		.metric-value {
			font-size: 2rem;
			font-weight: bold;
			color: #4472C4;
		}
		.metric-label {
			font-size: 0.9rem;
			color: #666;
		}
		.stButton>button {
			background-color: #4472C4;
			color: white;
		}
		.sidebar-section {
			margin-bottom: 2rem;
		}
		</style>
		""", unsafe_allow_html=True)

	def _render_header(self):
		"""Render the dashboard header."""
		st.image("https://www.nursing.arizona.edu/sites/default/files/styles/max_width_full/public/2023-04/tc%20banner%20home%20hero.png?itok=nJQD6jVY", use_column_width=True)

		# Check data status and display indicator
		data_loaded = self.data is not None

		# Two-column layout for title and status
		col1, col2 = st.columns([0.7, 0.3])

		with col1:
			st.title(APP_TITLE)

		with col2:
			st.markdown("<div style='height: 25px'></div>", unsafe_allow_html=True)  # Vertical spacing
			data_source = ""

			# Show source info
			if data_loaded:
				if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
					data_source = f"📄 {st.session_state.uploaded_file.name}"
				else:
					data_source = "🔍 Default dataset"

			# Create status indicator
			data_status = "✅" if data_loaded else "❌"
			status_text = f"Data Loaded: {data_status}"
			if data_source:
				status_text += f" | Source: {data_source}"

			st.markdown(
				f"<div style='text-align: right; padding: 10px; "
				f"border-radius: 5px; background-color: {'#E8F0FE' if data_loaded else '#FFF3CD'}; "
				f"color: {'#000080' if data_loaded else '#856404'}; font-weight: bold;'>"
				f"{status_text}</div>",
				unsafe_allow_html=True
			)

	def _render_sidebar(self):
		"""Render the sidebar navigation."""
		with st.sidebar:
			st.image("https://www.nursing.arizona.edu/sites/default/files/styles/uaqs_large/public/2023-05/Primary-logo-Nursing.png?itok=l84uKF2Z", width=200)

			# Add file upload section at the top
			st.markdown("### Data Source")

			# Option to use demo dataset
			use_demo = st.checkbox("Use demo dataset", value=True,
			                      help="Use the included TE-KOA dataset for demonstration")

			if use_demo:
				if 'uploaded_file' in st.session_state:
					del st.session_state.uploaded_file
				if self.data is None:
					if st.button("Load Demo Dataset", key="load_demo_btn"):
						# Reset data to force reloading
						self.data = None
						self.dictionary = None
						st.rerun()
			else:
				uploaded_file = st.file_uploader("Upload Excel dataset", type=["xlsx", "xls"])

				if uploaded_file is not None:
					# Store the uploaded file in session state
					if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file != uploaded_file:
						st.session_state.uploaded_file = uploaded_file
						# Reset data to force reloading with the new file
						self.data = None
						self.dictionary = None
						st.success("File uploaded! Click 'Load Dataset' to process it.")

				# Add a load button
				if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
					if st.button("Load Uploaded Dataset", key="load_dataset_btn"):
						# This will trigger data loading in the render method
						if self.data is None or self.dictionary is None:
							st.rerun()

			st.markdown("### Navigation")

			# Initialize current page in session state if not present
			if 'current_page' not in st.session_state:
				st.session_state.current_page = 'Overview'

			# Navigation buttons
			if st.button("📊 Overview", use_container_width=True):
				st.session_state.current_page = 'Overview'
				st.rerun()

			if st.button("🔍 Data Explorer", use_container_width=True):
				st.session_state.current_page = 'Data Explorer'
				st.rerun()

			if st.button("📖 Data Dictionary", use_container_width=True):
				st.session_state.current_page = 'Data Dictionary'
				st.rerun()

			st.markdown("#### Phase I Components")

			if st.button("🧩 Missing Data & Imputation", use_container_width=True):
				st.session_state.current_page = 'Missing Data & Imputation'
				st.rerun()

			if st.button("🧠 Variable Screening", use_container_width=True):
				st.session_state.current_page = 'Variable Screening'
				st.rerun()

			if st.button("📉 Dimensionality Reduction", use_container_width=True):
				st.session_state.current_page = 'Dimensionality Reduction'
				st.rerun()

			if st.button("📈 Data Quality", use_container_width=True):
				st.session_state.current_page = 'Data Quality'
				st.rerun()

			if st.button("👥 Treatment Groups", use_container_width=True):
				st.session_state.current_page = 'Treatment Groups'
				st.rerun()

			if st.button("💾 Pipeline & Export", use_container_width=True):
				st.session_state.current_page = 'Pipeline & Export'
				st.rerun()

			# Status indicators
			st.markdown("### Status")

			# Data loaded indicator
			data_loaded = self.data is not None
			st.markdown(f"Data Loaded: {'✅' if data_loaded else '❌'}")

			# Processing status
			processing_done = st.session_state.processed_data is not None
			st.markdown(f"Processing Complete: {'✅' if processing_done else '❌'}")

			# Pipeline indicators
			if 'pipeline_results' in st.session_state and st.session_state.pipeline_results:
				pipeline = st.session_state.pipeline_results
				st.markdown("### Pipeline Steps")

				imputation_done = 'imputation' in pipeline
				st.markdown(f"- Imputation: {'✅' if imputation_done else '❌'}")

				screening_done = 'variable_screening' in pipeline
				st.markdown(f"- Variable Screening: {'✅' if screening_done else '❌'}")

				dim_reduction_done = 'dimensionality_reduction' in pipeline
				st.markdown(f"- Dimensionality Reduction: {'✅' if dim_reduction_done else '❌'}")

				quality_done = 'data_quality' in pipeline
				st.markdown(f"- Data Quality: {'✅' if quality_done else '❌'}")

			# Dataset info
			if data_loaded:
				st.markdown("### Dataset Info")
				st.markdown(f"Participants: {len(self.data)}")
				st.markdown(f"Variables: {len(self.data.columns)}")

				# Show data types summary
				dtypes = self.data.dtypes.value_counts()
				st.markdown("#### Data Types")
				for dtype, count in dtypes.items():
					st.markdown(f"- {dtype}: {count}")

	def _render_overview(self):
		"""Render overview information about the dataset."""
		st.header("Dataset Overview")

		# Basic dataset information
		st.markdown('<div class="section-header"><h3>Basic Information</h3></div>', unsafe_allow_html=True)

		col1, col2, col3, col4 = st.columns(4)

		with col1:
			st.markdown('<div class="metric-card">', unsafe_allow_html=True)
			st.markdown(f'<div class="metric-value">{len(self.data)}</div>', unsafe_allow_html=True)
			st.markdown('<div class="metric-label">Participants</div>', unsafe_allow_html=True)
			st.markdown('</div>', unsafe_allow_html=True)

		with col2:
			st.markdown('<div class="metric-card">', unsafe_allow_html=True)
			st.markdown(f'<div class="metric-value">{len(self.data.columns)}</div>', unsafe_allow_html=True)
			st.markdown('<div class="metric-label">Variables</div>', unsafe_allow_html=True)
			st.markdown('</div>', unsafe_allow_html=True)

		with col3:
			# Count missing values
			missing_values = self.data.isnull().sum().sum()
			missing_percent = (missing_values / (len(self.data) * len(self.data.columns))) * 100

			st.markdown('<div class="metric-card">', unsafe_allow_html=True)
			st.markdown(f'<div class="metric-value">{missing_percent:.1f}%</div>', unsafe_allow_html=True)
			st.markdown('<div class="metric-label">Missing Values</div>', unsafe_allow_html=True)
			st.markdown('</div>', unsafe_allow_html=True)

		with col4:
			# Count variables with any missing values
			vars_with_missing = sum(self.data.isnull().sum() > 0)
			percent_vars_with_missing = (vars_with_missing / len(self.data.columns)) * 100

			st.markdown('<div class="metric-card">', unsafe_allow_html=True)
			st.markdown(f'<div class="metric-value">{percent_vars_with_missing:.1f}%</div>', unsafe_allow_html=True)
			st.markdown('<div class="metric-label">Variables with Missing Data</div>', unsafe_allow_html=True)
			st.markdown('</div>', unsafe_allow_html=True)

		# Treatment group summary
		st.markdown('<div class="section-header"><h3>Treatment Groups</h3></div>', unsafe_allow_html=True)

		# Get treatment groups if not already done
		if self.treatment_groups is None:
			self.treatment_groups = self.data_loader.get_treatment_groups()

		if self.treatment_groups:
			# Create a bar chart of treatment group sizes
			treatment_sizes = {group: len(df) for group, df in self.treatment_groups.items()}

			# Skip 'Experimental' since it's equivalent to 'tDCS + Meditation'
			if 'Experimental' in treatment_sizes:
				del treatment_sizes['Experimental']

			# Focus on the 2x2 factorial design groups
			factorial_groups = {
				'Control (No tDCS, No Meditation)': treatment_sizes.get('Control (No tDCS, No Meditation)', 0),
				'tDCS Only': treatment_sizes.get('tDCS Only', 0),
				'Meditation Only': treatment_sizes.get('Meditation Only', 0),
				'tDCS + Meditation': treatment_sizes.get('tDCS + Meditation', 0)
			}

			# Create a bar chart
			fig = px.bar(
				x=list(factorial_groups.keys()),
				y=list(factorial_groups.values()),
				color=list(factorial_groups.keys()),
				color_discrete_map={
					'Control (No tDCS, No Meditation)': TREATMENT_COLORS['Control (No tDCS, No Meditation)'],
					'tDCS Only': TREATMENT_COLORS['tDCS Only'],
					'Meditation Only': TREATMENT_COLORS['Meditation Only'],
					'tDCS + Meditation': TREATMENT_COLORS['tDCS + Meditation']
				},
				labels={'x': 'Treatment Group', 'y': 'Number of Participants'},
				title='Treatment Group Sizes'
			)

			fig.update_layout(showlegend=False)
			st.plotly_chart(fig, use_container_width=True)

			# 2x2 grid showing treatment groups
			st.markdown('<div class="section-header"><h3>Treatment Group Design (2×2 Factorial)</h3></div>', unsafe_allow_html=True)

			col1, col2, col3 = st.columns([1, 2, 1])

			with col2:
				# Create a DataFrame for the 2x2 grid
				grid_data = pd.DataFrame(
					[
						['Control\n(n=' + str(factorial_groups['Control (No tDCS, No Meditation)']) + ')',
						 'tDCS Only\n(n=' + str(factorial_groups['tDCS Only']) + ')'],
						['Meditation Only\n(n=' + str(factorial_groups['Meditation Only']) + ')',
						 'tDCS + Meditation\n(n=' + str(factorial_groups['tDCS + Meditation']) + ')']
					],
					index=['No Meditation', 'Meditation'],
					columns=['No tDCS', 'tDCS']
				)

				# Create a heatmap
				fig = px.imshow(
					[[factorial_groups['Control (No tDCS, No Meditation)'], factorial_groups['tDCS Only']],
					 [factorial_groups['Meditation Only'], factorial_groups['tDCS + Meditation']]],
					x=['No tDCS', 'tDCS'],
					y=['No Meditation', 'Meditation'],
					color_continuous_scale='Blues',
					labels=dict(x="tDCS Treatment", y="Meditation Treatment", color="Participants"),
					text_auto=True
				)

				fig.update_layout(title='2×2 Factorial Design')
				st.plotly_chart(fig, use_container_width=True)

		# Variable categories
		st.markdown('<div class="section-header"><h3>Variable Categories</h3></div>', unsafe_allow_html=True)

		variable_categories = self.data_loader.get_variable_categories()

		if variable_categories:
			# Count variables in each category
			category_counts = {category: len(variables) for category, variables in variable_categories.items()}

			# Create a horizontal bar chart
			fig = px.bar(
				y=list(category_counts.keys()),
				x=list(category_counts.values()),
				orientation='h',
				labels={'y': 'Category', 'x': 'Number of Variables'},
				title='Variables by Category',
				color=list(category_counts.keys()),
				color_discrete_sequence=px.colors.qualitative.Set3
			)

			fig.update_layout(showlegend=False)
			st.plotly_chart(fig, use_container_width=True)

			# Allow exploration of categories
			selected_category = st.selectbox(
				"Explore variables in category:",
				options=list(variable_categories.keys()),
				index=0
			)

			if selected_category:
				variables_in_category = variable_categories[selected_category]

				if variables_in_category:
					st.write(f"**{len(variables_in_category)} variables in {selected_category} category:**")

					# Show variables in a multi-column layout
					cols = st.columns(3)
					for i, variable in enumerate(sorted(variables_in_category)):
						cols[i % 3].markdown(f"- {variable}")
				else:
					st.info(f"No variables in {selected_category} category.")

		# Phase I pipeline summary
		st.markdown('<div class="section-header"><h3>Phase I: Data Preparation</h3></div>', unsafe_allow_html=True)

		# Create a roadmap of Phase I
		col1, col2 = st.columns(2)

		with col1:
			st.markdown("#### Data Preparation Pipeline")
			st.markdown("""
			1. **Data Loading & Exploration**
			   - Import dataset
			   - Explore variables and data dictionary
			   - Analyze data quality and completeness

			2. **Missing Data & Imputation**
			   - Identify patterns of missing data
			   - Apply appropriate imputation methods
			   - Validate imputed values

			3. **Variable Screening**
			   - Detect near-zero variance variables
			   - Identify collinear variables
			   - Calculate Variance Inflation Factor (VIF)
			   - Select optimal variable subset

			4. **Dimensionality Reduction**
			   - Apply Factor Analysis of Mixed Data (FAMD)
			   - Identify optimal component structure
			   - Retain ~75% of total information
			   - Interpret component meaning

			5. **Data Quality Enhancement**
			   - Detect and handle outliers
			   - Analyze variable distributions
			   - Apply transformations when needed
			   - Standardize variables

			6. **Treatment Group Analysis**
			   - Analyze 2×2 factorial design
			   - Compare baseline characteristics
			   - Assess balance between groups
			""")

		with col2:
			st.markdown("#### Outcomes & Deliverables")
			st.markdown("""
			- **Cleaned Dataset**
			  - Imputed missing values
			  - Removed redundant variables
			  - Enhanced data quality

			- **Dimensionality-Reduced Dataset**
			  - 8-10 synthetic components
			  - Reduced noise in the data
			  - Prepared for clustering in Phase II

			- **Variable Importance Assessment**
			  - Statistical importance rankings
			  - Clinical relevance assessment
			  - Optimal variable subset

			- **Data Quality Report**
			  - Missing data patterns
			  - Outlier detection summary
			  - Distribution analysis
			  - Transformation recommendations

			- **Treatment Group Profile**
			  - Baseline characteristics by group
			  - Group balance assessment
			  - Key outcome variables by group
			""")

		# Get started button
		st.markdown("### Ready to Begin?")
		col1, col2, col3 = st.columns([1, 2, 1])
		with col2:
			if st.button("Start Data Preparation Pipeline", use_container_width=True):
				st.session_state.current_page = 'Data Explorer'
				st.rerun()

	def _render_data_explorer(self):
		"""Render data exploration tools."""
		st.header("Data Explorer")

		# Tab navigation for data explorer
		tabs = st.tabs(["Data Preview", "Summary Statistics", "Variable Distribution", "Correlation Analysis"])

		with tabs[0]:  # Data Preview
			st.subheader("Data Preview")

			# Column selector
			selected_columns = st.multiselect(
				"Select columns to display",
				options=list(self.data.columns),
				default=list(self.data.columns)[:5]  # Default to first 5 columns
			)

			# Number of rows to display
			n_rows = st.slider("Number of rows to display", 5, 100, 10)

			if selected_columns:
				# Display filtered data
				st.dataframe(self.data[selected_columns].head(n_rows))

				# Download button for selected data
				csv = self.data[selected_columns].to_csv(index=False)
				b64 = base64.b64encode(csv.encode()).decode()
				href = f'<a href="data:file/csv;base64,{b64}" download="selected_data.csv">Download Selected Data</a>'
				st.markdown(href, unsafe_allow_html=True)
			else:
				st.info("Please select at least one column to display.")

		with tabs[1]:  # Summary Statistics
			st.subheader("Summary Statistics")

			# Select numeric columns for statistics
			numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()

			selected_numeric_cols = st.multiselect(
				"Select numeric columns for statistics",
				options=numeric_cols,
				default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
			)

			if selected_numeric_cols:
				# Display statistics
				stats = self.data[selected_numeric_cols].describe().T

				# Add additional statistics
				if len(self.data) > 0:
					stats['missing'] = self.data[selected_numeric_cols].isnull().sum()
					stats['missing_pct'] = self.data[selected_numeric_cols].isnull().sum() / len(self.data) * 100
					stats['skew'] = self.data[selected_numeric_cols].skew()
					stats['kurtosis'] = self.data[selected_numeric_cols].kurtosis()

				st.dataframe(stats.style.format({
					'mean': '{:.2f}',
					'std': '{:.2f}',
					'min': '{:.2f}',
					'25%': '{:.2f}',
					'50%': '{:.2f}',
					'75%': '{:.2f}',
					'max': '{:.2f}',
					'missing_pct': '{:.1f}%',
					'skew': '{:.2f}',
					'kurtosis': '{:.2f}'
				}))

				# Download button for statistics
				csv = stats.to_csv()
				b64 = base64.b64encode(csv.encode()).decode()
				href = f'<a href="data:file/csv;base64,{b64}" download="statistics.csv">Download Statistics</a>'
				st.markdown(href, unsafe_allow_html=True)
			else:
				st.info("Please select at least one numeric column for statistics.")

		with tabs[2]:  # Variable Distribution
			st.subheader("Variable Distribution")

			# Select a variable to visualize
			col1, col2 = st.columns(2)

			with col1:
				selected_var = st.selectbox(
					"Select a variable to visualize",
					options=numeric_cols,
					index=0 if numeric_cols else None
				)

			with col2:
				plot_type = st.selectbox(
					"Select plot type",
					options=["Histogram", "Box Plot", "Violin Plot", "KDE Plot"],
					index=0
				)

			if selected_var:
				# Filter out missing values
				data_to_plot = self.data[selected_var].dropna()

				if len(data_to_plot) > 0:
					# Create the plot
					if plot_type == "Histogram":
						fig = px.histogram(
							self.data, x=selected_var,
							title=f'Histogram of {selected_var}',
							color_discrete_sequence=[COLOR_PALETTE['primary']],
							marginal="box"
						)
					elif plot_type == "Box Plot":
						fig = px.box(
							self.data, y=selected_var,
							title=f'Box Plot of {selected_var}',
							color_discrete_sequence=[COLOR_PALETTE['primary']]
						)
					elif plot_type == "Violin Plot":
						fig = px.violin(
							self.data, y=selected_var,
							title=f'Violin Plot of {selected_var}',
							color_discrete_sequence=[COLOR_PALETTE['primary']],
							box=True
						)
					elif plot_type == "KDE Plot":
						fig = px.histogram(
							self.data, x=selected_var,
							title=f'KDE Plot of {selected_var}',
							color_discrete_sequence=[COLOR_PALETTE['primary']],
							marginal="rug",
							histnorm='probability density',
							nbins=50
						)

						fig.update_traces(fillcolor=COLOR_PALETTE['primary'], opacity=0.5)

					# Show plot
					st.plotly_chart(fig, use_container_width=True)

					# Display basic statistics
					st.markdown("##### Basic Statistics")
					col1, col2, col3, col4 = st.columns(4)

					col1.metric("Mean", f"{data_to_plot.mean():.2f}")
					col2.metric("Median", f"{data_to_plot.median():.2f}")
					col3.metric("Std Dev", f"{data_to_plot.std():.2f}")
					col4.metric("Missing", f"{self.data[selected_var].isnull().sum()} ({self.data[selected_var].isnull().sum() / len(self.data) * 100:.1f}%)")

					# Get variable description
					description = self.data_loader.get_variable_description(selected_var)
					if description:
						st.markdown(f"**Description:** {description}")
				else:
					st.warning(f"No non-missing values for {selected_var}.")
			else:
				st.info("Please select a variable to visualize.")

		with tabs[3]:  # Correlation Analysis
			st.subheader("Correlation Analysis")

			# Select variables for correlation analysis
			selected_corr_vars = st.multiselect(
				"Select variables for correlation analysis",
				options=numeric_cols,
				default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
			)

			if selected_corr_vars and len(selected_corr_vars) > 1:
				# Calculate correlation matrix
				corr_matrix = self.data[selected_corr_vars].corr()

				# Create heatmap
				fig = px.imshow(
					corr_matrix,
					text_auto=True,
					color_continuous_scale='RdBu_r',
					zmin=-1, zmax=1,
					title='Correlation Matrix'
				)

				# Show plot
				st.plotly_chart(fig, use_container_width=True)

				# Display pair plot option
				if len(selected_corr_vars) <= 5 and st.checkbox("Show Scatter Plot Matrix"):
					fig = px.scatter_matrix(
						self.data[selected_corr_vars].dropna(),
						dimensions=selected_corr_vars,
						color_discrete_sequence=[COLOR_PALETTE['primary']]
					)

					fig.update_layout(title='Scatter Plot Matrix')
					st.plotly_chart(fig, use_container_width=True)

				# Download correlation matrix
				csv = corr_matrix.to_csv()
				b64 = base64.b64encode(csv.encode()).decode()
				href = f'<a href="data:file/csv;base64,{b64}" download="correlation_matrix.csv">Download Correlation Matrix</a>'
				st.markdown(href, unsafe_allow_html=True)
			else:
				st.info("Please select at least two variables for correlation analysis.")

			# Add correlation network graph option
			if selected_corr_vars and len(selected_corr_vars) > 2 and st.checkbox("Show Correlation Network Graph"):
				# Correlation threshold
				corr_threshold = st.slider(
					"Correlation threshold",
					min_value=0.0,
					max_value=1.0,
					value=0.5,
					step=0.05
				)

				# Calculate correlation matrix
				corr_matrix = self.data[selected_corr_vars].corr().abs()

				# Create network graph
				edges = []
				for i in range(len(corr_matrix.columns)):
					for j in range(i+1, len(corr_matrix.columns)):
						if corr_matrix.iloc[i, j] >= corr_threshold:
							edges.append((
								corr_matrix.columns[i],
								corr_matrix.columns[j],
								corr_matrix.iloc[i, j]
							))

				if edges:
					G = nx.Graph()
					for var in selected_corr_vars:
						G.add_node(var)

					for source, target, weight in edges:
						G.add_edge(source, target, weight=weight)

					# Get positions using a layout algorithm
					pos = nx.spring_layout(G, seed=42)

					# Create edges trace
					edge_x = []
					edge_y = []
					edge_weights = []

					for edge in G.edges(data=True):
						x0, y0 = pos[edge[0]]
						x1, y1 = pos[edge[1]]
						edge_x.extend([x0, x1, None])
						edge_y.extend([y0, y1, None])
						edge_weights.append(edge[2]['weight'])

					# Normalize edge widths
					min_width = 1
					max_width = 10
					if edge_weights:
						normalized_weights = [
							min_width + (w - min(edge_weights)) * (max_width - min_width) / (max(edge_weights) - min(edge_weights))
							if max(edge_weights) > min(edge_weights) else 5
							for w in edge_weights
						]
					else:
						normalized_weights = []

					# Create nodes trace
					node_x = []
					node_y = []
					node_text = []

					for node in G.nodes():
						x, y = pos[node]
						node_x.append(x)
						node_y.append(y)
						node_text.append(node)

					# Calculate node degree for node size
					node_degrees = dict(G.degree())
					node_sizes = [30 + 10 * node_degrees[node] for node in G.nodes()]

					# Create the figure
					fig = go.Figure()

					# Add edges with varying widths based on correlation
					edge_segments = np.reshape(edge_x, (-1, 3))
					for i, segment in enumerate(edge_segments):
						if i < len(normalized_weights):
							fig.add_trace(go.Scatter(
								x=segment,
								y=edge_y[i*3:(i+1)*3],
								mode='lines',
								line=dict(width=normalized_weights[i], color='rgba(68, 114, 196, 0.5)'),
								hoverinfo='none'
							))

					# Add nodes
					fig.add_trace(go.Scatter(
						x=node_x, y=node_y,
						mode='markers+text',
						marker=dict(
							size=node_sizes,
							color=COLOR_PALETTE['primary'],
							line=dict(width=2, color='white')
						),
						text=node_text,
						textposition='top center',
						hoverinfo='text',
						hovertext=[f"{node}<br>Connections: {node_degrees[node]}" for node in G.nodes()]
					))

					# Update layout
					fig.update_layout(
						title='Correlation Network Graph',
						showlegend=False,
						hovermode='closest',
						margin=dict(b=20, l=5, r=5, t=40),
						xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
						yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
						width=800,
						height=600
					)

					# Show plot
					st.plotly_chart(fig, use_container_width=True)

					# Display edge information
					if st.checkbox("Show correlation details"):
						edge_df = pd.DataFrame(edges, columns=['Variable 1', 'Variable 2', 'Correlation'])
						edge_df = edge_df.sort_values('Correlation', ascending=False)
						st.dataframe(edge_df.style.format({'Correlation': '{:.3f}'}))
				else:
					st.info(f"No correlations above threshold ({corr_threshold:.2f}). Try lowering the threshold.")

	def _render_dictionary(self):
		"""Render the data dictionary."""
		st.header("Data Dictionary")

		# Tab navigation for dictionary
		tabs = st.tabs(["Dictionary Explorer", "Variable Categories", "Dictionary Visualization"])

		with tabs[0]:  # Dictionary Explorer
			st.subheader("Dictionary Explorer")

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
			st.dataframe(filtered_dict, height=400)

			# Allow looking up a specific variable
			st.subheader("Variable Lookup")

			lookup_var = st.selectbox(
				"Select a variable to look up",
				options=list(self.data.columns),
				index=0
			)

			if lookup_var:
				# Get variable description
				description = self.data_loader.get_variable_description(lookup_var)

				if description:
					st.markdown(f"**{lookup_var}:** {description}")

					# Get variable statistics
					if lookup_var in self.data.columns:
						var_data = self.data[lookup_var]

						# Check if numeric
						if pd.api.types.is_numeric_dtype(var_data):
							col1, col2, col3, col4 = st.columns(4)
							col1.metric("Mean", f"{var_data.mean():.2f}")
							col2.metric("Median", f"{var_data.median():.2f}")
							col3.metric("Std Dev", f"{var_data.std():.2f}")
							col4.metric("Missing", f"{var_data.isnull().sum()} ({var_data.isnull().sum() / len(self.data) * 100:.1f}%)")

							# Create histogram
							fig = px.histogram(
								var_data.dropna(),
								title=f'Distribution of {lookup_var}',
								color_discrete_sequence=[COLOR_PALETTE['primary']]
							)

							st.plotly_chart(fig, use_container_width=True)
						else:
							# For categorical variables
							value_counts = var_data.value_counts()

							col1, col2 = st.columns(2)
							col1.metric("Unique Values", var_data.nunique())
							col2.metric("Missing", f"{var_data.isnull().sum()} ({var_data.isnull().sum() / len(self.data) * 100:.1f}%)")

							# Create bar chart
							fig = px.bar(
								x=value_counts.index,
								y=value_counts.values,
								title=f'Value Counts for {lookup_var}',
								labels={'x': 'Value', 'y': 'Count'},
								color_discrete_sequence=[COLOR_PALETTE['primary']]
							)

							st.plotly_chart(fig, use_container_width=True)
				else:
					st.warning(f"No description found for {lookup_var}.")

		with tabs[1]:  # Variable Categories
			st.subheader("Variable Categories")

			# Get variable categories
			variable_categories = self.data_loader.get_variable_categories()

			if variable_categories:
				# Select a category
				selected_category = st.selectbox(
					"Select a variable category",
					options=list(variable_categories.keys()),
					index=0
				)

				if selected_category:
					variables_in_category = variable_categories[selected_category]

					if variables_in_category:
						st.write(f"**{len(variables_in_category)} variables in {selected_category} category:**")

						# Create a DataFrame with variable details
						var_details = []
						for var in sorted(variables_in_category):
							description = self.data_loader.get_variable_description(var)
							missing_count = self.data[var].isnull().sum() if var in self.data.columns else None
							missing_pct = missing_count / len(self.data) * 100 if missing_count is not None else None

							var_details.append({
								'Variable': var,
								'Description': description,
								'Missing Values': missing_count,
								'Missing %': missing_pct
							})

						var_details_df = pd.DataFrame(var_details)

						# Display the table
						st.dataframe(var_details_df.style.format({
							'Missing %': '{:.1f}%' if '{:.1f}%' is not None else None
						}), height=400)

						# Allow downloading the category variables
						csv = var_details_df.to_csv(index=False)
						b64 = base64.b64encode(csv.encode()).decode()
						href = f'<a href="data:file/csv;base64,{b64}" download="{selected_category}_variables.csv">Download {selected_category} Variables</a>'
						st.markdown(href, unsafe_allow_html=True)
					else:
						st.info(f"No variables in {selected_category} category.")
			else:
				st.info("No variable categories available.")

		with tabs[2]:  # Dictionary Visualization
			st.subheader("Dictionary Visualization")

			# Create a treemap of variables by category
			if variable_categories:
				# Prepare data for treemap
				treemap_data = []
				for category, variables in variable_categories.items():
					for var in variables:
						missing_count = self.data[var].isnull().sum() if var in self.data.columns else 0
						missing_pct = missing_count / len(self.data) * 100 if var in self.data.columns else 0

						treemap_data.append({
							'Category': category,
							'Variable': var,
							'Missing %': missing_pct
						})

				treemap_df = pd.DataFrame(treemap_data)

				# Create treemap
				fig = px.treemap(
					treemap_df,
					path=['Category', 'Variable'],
					color='Missing %',
					color_continuous_scale='RdYlGn_r',
					title='Variables by Category and Missing Data',
					height=600
				)

				fig.update_traces(marker=dict(cornerradius=5))
				st.plotly_chart(fig, use_container_width=True)

				# Create a sunburst chart alternative
				fig = px.sunburst(
					treemap_df,
					path=['Category', 'Variable'],
					color='Missing %',
					color_continuous_scale='RdYlGn_r',
					title='Variables by Category and Missing Data (Sunburst)',
					height=600
				)

				st.plotly_chart(fig, use_container_width=True)
			else:
				st.info("No variable categories available for visualization.")

			# Word cloud of variable descriptions
			if st.checkbox("Show Variable Description Word Cloud"):
				try:


					# Combine all descriptions
					all_descriptions = " ".join([
						str(desc) for desc in self.dictionary.iloc[:, 1] if desc and not pd.isna(desc)
					])

					if all_descriptions:
						# Create word cloud
						wordcloud = WordCloud(
							width=800,
							height=400,
							background_color='white',
							colormap='viridis',
							max_words=100,
							contour_width=1,
							contour_color='steelblue'
						).generate(all_descriptions)

						# Display word cloud
						fig, ax = plt.subplots(figsize=(10, 5))
						ax.imshow(wordcloud, interpolation='bilinear')
						ax.axis('off')
						plt.tight_layout()

						st.pyplot(fig)
					else:
						st.info("No descriptions available for word cloud.")
				except ImportError:
					st.warning("WordCloud package not installed. Please install it to use this feature.")

	def _render_missing_data_imputation(self):
		"""Render missing data analysis and imputation tools."""
		st.header("Missing Data & Imputation")

		# Tab navigation for missing data
		tabs = st.tabs(["Missing Data Analysis", "Missing Data Patterns", "Imputation", "Imputation Validation"])

		with tabs[0]:  # Missing Data Analysis
			st.subheader("Missing Data Analysis")

			# Display missing data report
			if self.missing_data_report is not None:
				# Only show columns with missing values
				missing_report = self.missing_data_report[self.missing_data_report['Missing Values'] > 0]

				if len(missing_report) > 0:
					# Create bar chart of missing percentages
					fig = px.bar(
						missing_report.sort_values('Percentage', ascending=False).head(20),
						y=missing_report.sort_values('Percentage', ascending=False).head(20).index,
						x='Percentage',
						orientation='h',
						title='Top 20 Variables with Missing Values',
						labels={'y': 'Variable', 'x': 'Missing (%)'},
						color='Percentage',
						color_continuous_scale='RdYlGn_r'
					)

					st.plotly_chart(fig, use_container_width=True)

					# Display the full report
					st.dataframe(missing_report.style.format({'Percentage': '{:.1f}%'}), height=400)

					# Download button for missing data report
					csv = missing_report.to_csv()
					b64 = base64.b64encode(csv.encode()).decode()
					href = f'<a href="data:file/csv;base64,{b64}" download="missing_data_report.csv">Download Missing Data Report</a>'
					st.markdown(href, unsafe_allow_html=True)

					# Summary statistics
					col1, col2, col3 = st.columns(3)

					col1.metric("Variables with Missing Data", f"{len(missing_report)} ({len(missing_report) / len(self.data.columns) * 100:.1f}%)")
					col2.metric("Total Missing Values", f"{self.data.isnull().sum().sum()} ({self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns)) * 100:.1f}%)")
					col3.metric("Complete Cases", f"{len(self.data.dropna())} ({len(self.data.dropna()) / len(self.data) * 100:.1f}%)")
				else:
					st.success("No missing values found in the dataset!")
			else:
				st.warning("Missing data report not available.")

		with tabs[1]:  # Missing Data Patterns
			st.subheader("Missing Data Patterns")

			# Create a heatmap of missing values
			if self.data is not None:
				# Select top variables with missing values
				n_vars = st.slider("Number of variables to show", 5, 50, 20)

				# Get top n variables with most missing values
				missing_counts = self.data.isnull().sum()
				top_missing_vars = missing_counts[missing_counts > 0].sort_values(ascending=False).head(n_vars).index.tolist()

				if top_missing_vars:
					# Create missing data matrix (True for missing, False for present)
					missing_matrix = self.data[top_missing_vars].isnull()

					# Create heatmap
					fig = px.imshow(
						missing_matrix.values,
						x=top_missing_vars,
						color_continuous_scale='RdBu',
						title=f'Missing Data Patterns (Top {n_vars} Variables)',
						height=600
					)

					fig.update_xaxes(tickangle=45)
					st.plotly_chart(fig, use_container_width=True)

					# Create a correlation heatmap of missing values
					st.subheader("Missing Value Correlation")
					st.markdown("This shows which variables tend to be missing together.")

					# Calculate correlation of missing indicators
					missing_corr = missing_matrix.corr()

					# Create heatmap
					fig = px.imshow(
						missing_corr.values,
						x=missing_corr.columns,
						y=missing_corr.index,
						color_continuous_scale='RdBu_r',
						title='Correlation of Missing Values',
						labels=dict(x="Variable", y="Variable", color="Correlation")
					)

					fig.update_xaxes(tickangle=45)
					st.plotly_chart(fig, use_container_width=True)

					# Find pairs of variables with highly correlated missingness
					high_corr_threshold = 0.5
					high_corr_pairs = []

					for i in range(len(missing_corr.columns)):
						for j in range(i+1, len(missing_corr.columns)):
							if missing_corr.iloc[i, j] >= high_corr_threshold:
								high_corr_pairs.append((
									missing_corr.columns[i],
									missing_corr.columns[j],
									missing_corr.iloc[i, j]
								))

					if high_corr_pairs:
						st.markdown(f"**Variables with correlated missingness (correlation ≥ {high_corr_threshold}):**")

						# Create DataFrame for display
						corr_pairs_df = pd.DataFrame(high_corr_pairs, columns=['Variable 1', 'Variable 2', 'Correlation'])
						corr_pairs_df = corr_pairs_df.sort_values('Correlation', ascending=False)

						st.dataframe(corr_pairs_df.style.format({'Correlation': '{:.3f}'}))
				else:
					st.info("No variables with missing values found.")

				# Missing data by groups
				st.subheader("Missing Data by Groups")
				st.markdown("This shows if certain groups have more missing data than others.")

				# Get treatment groups if not already done
				if self.treatment_groups is None:
					self.treatment_groups = self.data_loader.get_treatment_groups()

				if self.treatment_groups:
					# Calculate missing percentages by group
					group_missing = {}

					for group, group_data in self.treatment_groups.items():
						if 'Experimental' in group:  # Skip 'Experimental' since it's equivalent to 'tDCS + Meditation'
							continue

						group_missing[group] = group_data[top_missing_vars].isnull().mean() * 100

					# Combine into a DataFrame
					group_missing_df = pd.DataFrame(group_missing)

					# Create heatmap
					fig = px.imshow(
						group_missing_df.values.T,  # Transpose to have groups as rows
						x=group_missing_df.index,
						y=group_missing_df.columns,
						color_continuous_scale='RdYlGn_r',
						title='Missing Data Percentage by Treatment Group',
						labels=dict(x="Variable", y="Treatment Group", color="Missing (%)"),
						text_auto='.1f'
					)

					fig.update_xaxes(tickangle=45)
					st.plotly_chart(fig, use_container_width=True)
			else:
				st.warning("Data not available.")

		with tabs[2]:  # Imputation
			st.subheader("Imputation Methods")

			# Imputation options
			st.markdown("### Impute Missing Values")

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

			# Variables with missing values
			variables_with_missing = self.data.columns[self.data.isnull().any()].tolist()

			# Column exclusion
			if variables_with_missing:
				st.markdown("### Variables with Missing Values")
				st.markdown("Select variables to exclude from imputation (they will remain missing):")

				# Organize variables by category for easier selection
				variable_categories = self.data_loader.get_variable_categories()

				if variable_categories:
					# Group missing variables by category
					missing_by_category = {}
					for category, vars_in_category in variable_categories.items():
						missing_in_category = [var for var in vars_in_category if var in variables_with_missing]
						if missing_in_category:
							missing_by_category[category] = missing_in_category

					# Create multiselect for each category with missing variables
					cols_to_exclude = []

					for category, missing_vars in missing_by_category.items():
						with st.expander(f"{category} ({len(missing_vars)} variables)"):
							selected = st.multiselect(
								f"Select variables to exclude from {category}",
								options=missing_vars,
								default=[]
							)
							cols_to_exclude.extend(selected)
				else:
					# Simple multiselect if no categories
					cols_to_exclude = st.multiselect(
						"Select variables to exclude from imputation",
						options=variables_with_missing,
						default=[]
					)
			else:
				st.success("No missing values to impute!")
				cols_to_exclude = []

			# Impute button
			if variables_with_missing and st.button("Impute Missing Values"):
				with st.spinner("Imputing missing values..."):
					try:
						self.imputed_data = self.data_loader.impute_missing_values(
							method=imputation_method,
							knn_neighbors=knn_neighbors,
							cols_to_exclude=cols_to_exclude
						)

						# Store imputed data in session state
						st.session_state.processed_data = self.imputed_data

						# Store imputation details in pipeline results
						if 'pipeline_results' not in st.session_state:
							st.session_state.pipeline_results = {}

						st.session_state.pipeline_results['imputation'] = {
							'method': imputation_method,
							'knn_neighbors': knn_neighbors,
							'cols_excluded': cols_to_exclude,
							'original_missing': self.data.isnull().sum().sum(),
							'remaining_missing': self.imputed_data.isnull().sum().sum()
						}

						st.success("Successfully imputed missing values!")

						# Display imputation summary
						original_missing = self.data.isnull().sum().sum()
						remaining_missing = self.imputed_data.isnull().sum().sum()
						imputed_count = original_missing - remaining_missing

						col1, col2, col3 = st.columns(3)
						col1.metric("Original Missing Values", original_missing)
						col2.metric("Imputed Values", imputed_count)
						col3.metric("Remaining Missing Values", remaining_missing)
					except Exception as e:
						st.error(f"Error during imputation: {e}")
						logger.error(f"Imputation error: {e}", exc_info=True)

		with tabs[3]:  # Imputation Validation
			st.subheader("Imputation Validation")

			# Check if imputed data is available
			if self.imputed_data is not None or st.session_state.processed_data is not None:
				# Use either imputed_data or processed_data from session state
				imputed_data = self.imputed_data if self.imputed_data is not None else st.session_state.processed_data

				# Select a variable with imputed values
				variables_with_imputed = []

				for col in self.data.columns:
					if self.data[col].isnull().sum() > 0 and imputed_data[col].isnull().sum() < self.data[col].isnull().sum():
						variables_with_imputed.append(col)

				if variables_with_imputed:
					selected_var = st.selectbox(
						"Select a variable with imputed values to validate",
						options=variables_with_imputed
					)

					if selected_var:
						# Create before/after comparison
						col1, col2 = st.columns(2)

						with col1:
							st.markdown("### Before Imputation")

							# Statistics before imputation
							orig_data = self.data[selected_var].dropna()

							st.metric("Missing Values", self.data[selected_var].isnull().sum())
							st.metric("Mean", f"{orig_data.mean():.2f}")
							st.metric("Median", f"{orig_data.median():.2f}")
							st.metric("Std Dev", f"{orig_data.std():.2f}")

							# Histogram before imputation
							fig = px.histogram(
								orig_data,
								title=f'{selected_var} (Before Imputation)',
								color_discrete_sequence=[COLOR_PALETTE['primary']]
							)

							st.plotly_chart(fig, use_container_width=True)

						with col2:
							st.markdown("### After Imputation")

							# Statistics after imputation
							imp_data = imputed_data[selected_var]

							st.metric("Missing Values", imp_data.isnull().sum())
							st.metric("Mean", f"{imp_data.dropna().mean():.2f}")
							st.metric("Median", f"{imp_data.dropna().median():.2f}")
							st.metric("Std Dev", f"{imp_data.dropna().std():.2f}")

							# Histogram after imputation
							fig = px.histogram(
								imp_data.dropna(),
								title=f'{selected_var} (After Imputation)',
								color_discrete_sequence=[COLOR_PALETTE['secondary']]
							)

							st.plotly_chart(fig, use_container_width=True)

						# Compare original vs imputed values
						st.subheader("Comparison of Original vs. Imputed")

						# Create a scatter plot of original vs imputed where possible
						if pd.api.types.is_numeric_dtype(self.data[selected_var]):
							# Create a combined dataset with flags for imputed values
							combined = pd.DataFrame({
								'Value': imputed_data[selected_var],
								'Status': ['Original' if not pd.isna(self.data.iloc[i][selected_var]) else 'Imputed'
										 for i in range(len(self.data))]
							})

							# Create box plots for comparison
							fig = px.box(
								combined,
								x='Status',
								y='Value',
								title=f'Original vs. Imputed Values for {selected_var}',
								color='Status',
								color_discrete_map={
									'Original': COLOR_PALETTE['primary'],
									'Imputed': COLOR_PALETTE['secondary']
								},
								notched=True,
								points='all'
							)

							st.plotly_chart(fig, use_container_width=True)

							# Add statistical comparison
							if 'Imputed' in combined['Status'].values and 'Original' in combined['Status'].values:
								original_values = combined[combined['Status'] == 'Original']['Value']
								imputed_values = combined[combined['Status'] == 'Imputed']['Value']

								# Basic statistics comparison
								stats_df = pd.DataFrame({
									'Original': [
										len(original_values),
										original_values.mean(),
										original_values.median(),
										original_values.std(),
										original_values.min(),
										original_values.max()
									],
									'Imputed': [
										len(imputed_values),
										imputed_values.mean(),
										imputed_values.median(),
										imputed_values.std(),
										imputed_values.min(),
										imputed_values.max()
									]
								}, index=['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'])

								st.dataframe(stats_df.style.format({
									'Original': '{:.2f}',
									'Imputed': '{:.2f}'
								}))

								# Perform t-test if enough samples
								if len(imputed_values) >= 5 and len(original_values) >= 5:


									t_stat, p_value = scipy_stats.ttest_ind(
										original_values.dropna(),
										imputed_values.dropna(),
										equal_var=False  # Welch's t-test
									)

									st.markdown(f"**T-test results:** t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")

									if p_value < 0.05:
										st.warning("The distributions of original and imputed values are significantly different (p < 0.05).")
									else:
										st.success("The distributions of original and imputed values are not significantly different (p ≥ 0.05).")
						else:
							st.info(f"Variable {selected_var} is not numeric. Cannot create comparative visualization.")
				else:
					st.info("No variables with imputed values found.")
			else:
				st.warning("No imputed data available. Please impute missing values first.")

	def _render_variable_screening(self):
		"""Render variable screening tools."""
		st.header("Variable Screening")

		# Initialize variable screener if not done and data is available
		if self.variable_screener is None and (self.imputed_data is not None or st.session_state.processed_data is not None):
			# Use imputed data if available, otherwise use original data
			data_for_screening = self.imputed_data if self.imputed_data is not None else \
								 st.session_state.processed_data if st.session_state.processed_data is not None else \
								 self.data

			self.variable_screener = VariableScreener(data_for_screening)

		# Tab navigation for variable screening
		tabs = st.tabs(["Near-Zero Variance", "Collinearity Analysis", "VIF Analysis", "Variable Recommendations"])

		with tabs[0]:  # Near-Zero Variance
			st.subheader("Near-Zero Variance Detection")

			st.markdown("""
			Near-zero variance variables have very little variation in their values and provide minimal information for analysis.
			They can cause instability in machine learning models and are generally safe to remove.
			""")

			# Thresholds for near-zero variance
			threshold = st.slider(
				"Unique value threshold (% of total observations)",
				min_value=0.1,
				max_value=10.0,
				value=1.0,
				step=0.1,
				help="Variables with unique values less than this percentage of total observations will be flagged."
			) / 100  # Convert to proportion

			# Detect near-zero variance variables
			if self.variable_screener is not None:
				if st.button("Detect Near-Zero Variance Variables"):
					with st.spinner("Detecting near-zero variance variables..."):
						near_zero_vars = self.variable_screener.identify_near_zero_variance(threshold=threshold)

						# Display results
						if near_zero_vars:
							st.warning(f"Found {len(near_zero_vars)} variables with near-zero variance.")

							# Get variable details
							var_stats = self.variable_screener.results['near_zero_variance']

							# Create a DataFrame for display
							var_details = []
							for var in near_zero_vars:
								stats = var_stats[var]
								description = self.data_loader.get_variable_description(var)

								var_details.append({
									'Variable': var,
									'Unique Values': stats['n_unique'],
									'Most Common (%)': stats['most_common_freq'] * 100,
									'Description': description or 'No description available'
								})

							var_details_df = pd.DataFrame(var_details).sort_values('Unique Values')

							# Display the table
							st.dataframe(var_details_df.style.format({
								'Most Common (%)': '{:.1f}%'
							}), height=400)

							# Display distribution of a selected variable
							selected_var = st.selectbox(
								"Select a variable to examine:",
								options=near_zero_vars
							)

							if selected_var:
								# Check if numeric
								if pd.api.types.is_numeric_dtype(self.data[selected_var]):
									fig = px.histogram(
										self.data[selected_var].dropna(),
										title=f'Distribution of {selected_var}',
										color_discrete_sequence=[COLOR_PALETTE['primary']]
									)

									st.plotly_chart(fig, use_container_width=True)
								else:
									# For categorical variables
									value_counts = self.data[selected_var].value_counts()

									fig = px.bar(
										x=value_counts.index,
										y=value_counts.values,
										title=f'Value Counts for {selected_var}',
										labels={'x': 'Value', 'y': 'Count'},
										color_discrete_sequence=[COLOR_PALETTE['primary']]
									)

									st.plotly_chart(fig, use_container_width=True)
						else:
							st.success("No variables with near-zero variance found.")
			else:
				st.warning("Data not available for variable screening. Please complete data imputation first.")

		with tabs[1]:  # Collinearity Analysis
			st.subheader("Collinearity Analysis")

			st.markdown("""
			Collinearity occurs when two or more variables are highly correlated. Highly collinear variables can cause
			instability in statistical models and may not provide unique information. It's often beneficial to
			remove one variable from each highly correlated pair.
			""")

			# Threshold for high correlation
			corr_threshold = st.slider(
				"Correlation threshold",
				min_value=0.5,
				max_value=1.0,
				value=0.85,
				step=0.05,
				help="Variable pairs with absolute correlation above this threshold will be flagged."
			)

			# Analyze collinearity
			if self.variable_screener is not None:
				if st.button("Analyze Collinearity"):
					with st.spinner("Analyzing collinearity..."):
						high_corr_pairs = self.variable_screener.analyze_collinearity(threshold=corr_threshold)

						# Display results
						if high_corr_pairs:
							st.warning(f"Found {len(high_corr_pairs)} pairs of highly correlated variables.")

							# Create a DataFrame for display
							pairs_df = pd.DataFrame(high_corr_pairs, columns=['Variable 1', 'Variable 2', 'Correlation'])
							pairs_df = pairs_df.sort_values('Correlation', ascending=False)

							# Display the table
							st.dataframe(pairs_df.style.format({
								'Correlation': '{:.3f}'
							}), height=400)

							# Visualize correlation matrix
							corr_matrix = self.variable_screener.get_correlation_matrix()

							# Get variables in high correlation pairs
							corr_vars = set()
							for var1, var2, _ in high_corr_pairs:
								corr_vars.add(var1)
								corr_vars.add(var2)

							# Create heatmap of correlation matrix for these variables
							if corr_vars:
								corr_vars = sorted(list(corr_vars))

								fig = px.imshow(
									corr_matrix.loc[corr_vars, corr_vars].values,
									x=corr_vars,
									y=corr_vars,
									color_continuous_scale='RdBu_r',
									zmin=-1, zmax=1,
									title='Correlation Matrix for Highly Correlated Variables'
								)

								fig.update_xaxes(tickangle=45)
								st.plotly_chart(fig, use_container_width=True)

								# Correlation network graph
								st.subheader("Correlation Network Graph")

								# Create network graph
								G = nx.Graph()
								for var in corr_vars:
									G.add_node(var)

								for var1, var2, corr in high_corr_pairs:
									G.add_edge(var1, var2, weight=corr)

								# Get positions using a layout algorithm
								pos = nx.spring_layout(G, seed=42)

								# Create edges trace
								edge_x = []
								edge_y = []
								edge_weights = []

								for edge in G.edges(data=True):
									x0, y0 = pos[edge[0]]
									x1, y1 = pos[edge[1]]
									edge_x.extend([x0, x1, None])
									edge_y.extend([y0, y1, None])
									edge_weights.append(edge[2]['weight'])

								# Normalize edge widths
								min_width = 1
								max_width = 10
								if edge_weights:
									normalized_weights = [
										min_width + (w - min(edge_weights)) * (max_width - min_width) / (max(edge_weights) - min(edge_weights))
										if max(edge_weights) > min(edge_weights) else 5
										for w in edge_weights
									]
								else:
									normalized_weights = []

								# Create nodes trace
								node_x = []
								node_y = []
								node_text = []

								for node in G.nodes():
									x, y = pos[node]
									node_x.append(x)
									node_y.append(y)
									node_text.append(node)

								# Calculate node degree for node size
								node_degrees = dict(G.degree())
								node_sizes = [30 + 10 * node_degrees[node] for node in G.nodes()]

								# Create the figure
								fig = go.Figure()

								# Add edges with varying widths based on correlation
								for i, (x0, y0, x1, y1) in enumerate(zip(edge_x[::3], edge_y[::3], edge_x[1::3], edge_y[1::3])):
									if i < len(normalized_weights):
										fig.add_trace(go.Scatter(
											x=[x0, x1, None],
											y=[y0, y1, None],
											mode='lines',
											line=dict(width=normalized_weights[i], color='rgba(68, 114, 196, 0.5)'),
											hoverinfo='none'
										))

								# Add nodes
								fig.add_trace(go.Scatter(
									x=node_x, y=node_y,
									mode='markers+text',
									marker=dict(
										size=node_sizes,
										color=COLOR_PALETTE['primary'],
										line=dict(width=2, color='white')
									),
									text=node_text,
									textposition='top center',
									hoverinfo='text',
									hovertext=[f"{node}<br>Connections: {node_degrees[node]}" for node in G.nodes()]
								))

								# Update layout
								fig.update_layout(
									title='Correlation Network Graph',
									showlegend=False,
									hovermode='closest',
									margin=dict(b=20, l=5, r=5, t=40),
									xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
									yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
									width=800,
									height=600
								)

								st.plotly_chart(fig, use_container_width=True)

								# Select a pair to examine
								st.subheader("Examine Correlated Variable Pair")

								# Create a list of pairs for selection
								pair_strings = [f"{var1} & {var2} (r={corr:.2f})" for var1, var2, corr in high_corr_pairs]

								selected_pair_str = st.selectbox(
									"Select a pair to examine:",
									options=pair_strings
								)

								if selected_pair_str:
									# Extract pair from string
									selected_pair_index = pair_strings.index(selected_pair_str)
									var1, var2, _ = high_corr_pairs[selected_pair_index]

									# Check if both variables are numeric
									if pd.api.types.is_numeric_dtype(self.data[var1]) and pd.api.types.is_numeric_dtype(self.data[var2]):
										# Create scatter plot
										fig = px.scatter(
											x=self.data[var1],
											y=self.data[var2],
											title=f'Correlation between {var1} and {var2}',
											labels={'x': var1, 'y': var2},
											color_discrete_sequence=[COLOR_PALETTE['primary']],
											trendline='ols'
										)

										st.plotly_chart(fig, use_container_width=True)

										# Get variable descriptions
										desc1 = self.data_loader.get_variable_description(var1)
										desc2 = self.data_loader.get_variable_description(var2)

										# Display descriptions if available
										if desc1:
											st.markdown(f"**{var1}:** {desc1}")
										if desc2:
											st.markdown(f"**{var2}:** {desc2}")
									else:
										st.info("One or both variables are not numeric. Cannot create scatter plot.")
						else:
							st.success("No variables with high collinearity found.")
			else:
				st.warning("Data not available for collinearity analysis. Please complete data imputation first.")

		with tabs[2]:  # VIF Analysis
			st.subheader("Variance Inflation Factor (VIF) Analysis")

			st.markdown("""
			The Variance Inflation Factor (VIF) measures how much the variance of an estimated regression coefficient
			increases if your predictors are correlated. It's another way to detect multicollinearity.

			- VIF = 1: No multicollinearity
			- VIF between 1-5: Moderate multicollinearity
			- VIF > 5: High multicollinearity
			- VIF > 10: Very high multicollinearity
			""")

			# Threshold for high VIF
			vif_threshold = st.slider(
				"VIF threshold",
				min_value=2.0,
				max_value=10.0,
				value=5.0,
				step=0.5,
				help="Variables with VIF above this threshold may have multicollinearity issues."
			)

			# Calculate VIF
			if self.variable_screener is not None:
				if st.button("Calculate VIF"):
					with st.spinner("Calculating VIF..."):
						try:
							vif_values = self.variable_screener.calculate_vif(max_vif=vif_threshold)

							# Display results
							if vif_values:
								# Create a DataFrame for display
								vif_df = pd.DataFrame({
									'Variable': list(vif_values.keys()),
									'VIF': list(vif_values.values())
								}).sort_values('VIF', ascending=False)

								# Identify high VIF variables
								high_vif_vars = vif_df[vif_df['VIF'] > vif_threshold]

								if len(high_vif_vars) > 0:
									st.warning(f"Found {len(high_vif_vars)} variables with high VIF (> {vif_threshold}).")
								else:
									st.success(f"No variables with high VIF (> {vif_threshold}) found.")

								# Display the table
								st.dataframe(vif_df.style.format({
									'VIF': '{:.2f}'
								}), height=400)

								# Create bar chart
								fig = px.bar(
									vif_df.head(20),  # Top 20 variables by VIF
									x='VIF',
									y='Variable',
									orientation='h',
									title='Top 20 Variables by VIF',
									color='VIF',
									color_continuous_scale='RdYlGn_r'
								)

								# Add threshold line
								fig.add_vline(
									x=vif_threshold,
									line_dash="dash",
									line_color="red",
									annotation_text=f"Threshold: {vif_threshold}",
									annotation_position="top right"
								)

								st.plotly_chart(fig, use_container_width=True)
							else:
								st.info("VIF calculation not possible with the current data.")
						except Exception as e:
							st.error(f"Error calculating VIF: {e}")
							logger.error(f"VIF calculation error: {e}", exc_info=True)
			else:
				st.warning("Data not available for VIF analysis. Please complete data imputation first.")

		with tabs[3]:  # Variable Recommendations
			st.subheader("Variable Recommendations")

			st.markdown("""
			Based on the analyses of near-zero variance, collinearity, and VIF, we can recommend a subset of variables
			to use for further analysis. This helps reduce dimensionality and improve model stability.
			""")

			# Recommendation settings
			col1, col2, col3 = st.columns(3)

			with col1:
				near_zero_threshold = st.slider(
					"Near-zero variance threshold (%)",
					min_value=0.1,
					max_value=10.0,
					value=1.0,
					step=0.1
				) / 100  # Convert to proportion

			with col2:
				collinearity_threshold = st.slider(
					"Collinearity threshold",
					min_value=0.5,
					max_value=1.0,
					value=0.85,
					step=0.05
				)

			with col3:
				vif_threshold = st.slider(
					"VIF threshold",
					min_value=2.0,
					max_value=10.0,
					value=5.0,
					step=0.5
				)

			# Force include certain variables
			st.markdown("### Force Include Variables")
			st.markdown("Select variables to always include regardless of screening criteria:")

			# Organize variables by category for easier selection
			variable_categories = self.data_loader.get_variable_categories()
			force_include = []

			if variable_categories:
				for category in ['outcome', 'treatment', 'demographic']:
					if category in variable_categories:
						with st.expander(f"{category.capitalize()} Variables"):
							selected = st.multiselect(
								f"Select {category} variables to force include",
								options=variable_categories[category],
								default=variable_categories[category] if category in ['treatment', 'outcome'] else []
							)
							force_include.extend(selected)

			# Button to get recommendations
			if self.variable_screener is not None:
				if st.button("Get Variable Recommendations"):
					with st.spinner("Analyzing variables and generating recommendations..."):
						# Get recommendations
						recommendations = self.variable_screener.recommend_variables(
							near_zero_threshold=near_zero_threshold,
							collinearity_threshold=collinearity_threshold,
							vif_threshold=vif_threshold,
							force_include=force_include
						)

						# Store in session state
						if 'pipeline_results' not in st.session_state:
							st.session_state.pipeline_results = {}

						st.session_state.pipeline_results['variable_screening'] = self.variable_screener.get_results()

						# Display results
						st.subheader("Variable Screening Results")

						col1, col2, col3, col4 = st.columns(4)

						col1.metric("Total Variables", recommendations['total_variables'])
						col2.metric("Near-Zero Variables", recommendations['near_zero_variables'])
						col3.metric("Highly Correlated Pairs", recommendations['highly_correlated_pairs'])
						col4.metric("High VIF Variables", recommendations['high_vif_variables'])

						# Display recommended variables
						st.subheader("Recommended Variables")
						st.success(f"Recommended {len(recommendations['recommended_variable_list'])} variables out of {recommendations['total_variables']} total variables.")

						# Organize by category for easier viewing
						if variable_categories:
							recommended_by_category = {}
							uncategorized = []

							for var in recommendations['recommended_variable_list']:
								categorized = False
								for category, vars_in_category in variable_categories.items():
									if var in vars_in_category:
										if category not in recommended_by_category:
											recommended_by_category[category] = []
										recommended_by_category[category].append(var)
										categorized = True
										break

								if not categorized:
									uncategorized.append(var)

							# Display by category
							for category, vars_in_category in recommended_by_category.items():
								with st.expander(f"{category.capitalize()} Variables ({len(vars_in_category)})"):
									# Create columns
									cols = st.columns(3)
									for i, var in enumerate(sorted(vars_in_category)):
										cols[i % 3].markdown(f"- {var}")

							# Display uncategorized
							if uncategorized:
								with st.expander(f"Uncategorized Variables ({len(uncategorized)})"):
									# Create columns
									cols = st.columns(3)
									for i, var in enumerate(sorted(uncategorized)):
										cols[i % 3].markdown(f"- {var}")
						else:
							# Simple list if no categories
							with st.expander(f"Recommended Variables ({len(recommendations['recommended_variable_list'])})"):
								# Create columns
								cols = st.columns(3)
								for i, var in enumerate(sorted(recommendations['recommended_variable_list'])):
									cols[i % 3].markdown(f"- {var}")

						# Apply recommendations
						if st.button("Apply Recommendations to Dataset"):
							# Filter the dataset to keep only recommended variables
							if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
								recommended_vars = recommendations['recommended_variable_list']

								# Ensure all recommended variables exist in the dataset
								existing_vars = [var for var in recommended_vars if var in st.session_state.processed_data.columns]

								# Update processed data
								st.session_state.processed_data = st.session_state.processed_data[existing_vars]

								st.success(f"Dataset updated to include only the {len(existing_vars)} recommended variables.")
								st.rerun()
			else:
				st.warning("Data not available for variable recommendations. Please complete data imputation first.")

	def _render_dimensionality_reduction(self):
		"""Render dimensionality reduction tools."""
		st.header("Dimensionality Reduction")

		# Initialize dimensionality reducer if not done and data is available
		if self.dimensionality_reducer is None and st.session_state.processed_data is not None:
			self.dimensionality_reducer = DimensionalityReducer(st.session_state.processed_data)

		# Tab navigation for dimensionality reduction
		tabs = st.tabs(["PCA Analysis", "FAMD Analysis", "Component Interpretation", "Transformed Data"])

		with tabs[0]:  # PCA Analysis
			st.subheader("Principal Component Analysis (PCA)")

			st.markdown("""
			Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms the original variables
			into a set of linearly uncorrelated variables called principal components. PCA works best with numeric variables.
			""")

			# PCA settings
			col1, col2 = st.columns(2)

			with col1:
				# Select variables for PCA
				if self.dimensionality_reducer is not None:
					numeric_vars = self.dimensionality_reducer.numeric_vars

					# Check if variable screening was performed
					if 'variable_screening' in st.session_state.pipeline_results:
						screening_results = st.session_state.pipeline_results['variable_screening']
						recommended_vars = screening_results['recommended_variables']

						# Default to recommended numeric variables
						default_vars = [var for var in recommended_vars if var in numeric_vars]
					else:
						default_vars = numeric_vars

					selected_vars = st.multiselect(
						"Select variables for PCA",
						options=numeric_vars,
						default=default_vars[:20] if len(default_vars) > 20 else default_vars
					)
				else:
					selected_vars = []
					st.warning("Data not available for PCA. Please complete data imputation first.")

			with col2:
				# Number of components
				if selected_vars:
					max_components = min(len(selected_vars), len(st.session_state.processed_data) - 1)

					n_components = st.slider(
						"Number of components",
						min_value=2,
						max_value=min(max_components, 20),
						value=min(8, max_components)
					)

					# Standardization option
					standardize = st.checkbox("Standardize variables before PCA", value=True)
				else:
					n_components = 2
					standardize = True

			# Perform PCA
			if self.dimensionality_reducer is not None and selected_vars:
				if st.button("Perform PCA"):
					with st.spinner("Performing PCA..."):
						# Run PCA
						pca_results = self.dimensionality_reducer.perform_pca(
							variables=selected_vars,
							n_components=n_components,
							standardize=standardize
						)

# Store in session state
						if 'pipeline_results' not in st.session_state:
							st.session_state.pipeline_results = {}

						st.session_state.pipeline_results['dimensionality_reduction'] = {
							'method': 'pca',
							'n_components': n_components,
							'standardize': standardize,
							'variables': selected_vars,
							'explained_variance': pca_results['explained_variance_ratio'].tolist(),
							'cumulative_explained_variance': pca_results['cumulative_explained_variance'].tolist()
						}

						# Display results
						st.success(f"PCA successfully performed with {n_components} components!")

						# Scree plot
						st.subheader("Scree Plot")
						fig, _ = self.dimensionality_reducer.plot_scree(method='pca')
						st.pyplot(fig)

						# Get optimal number of components
						variance_threshold = 0.75  # 75% explained variance
						optimal_components = self.dimensionality_reducer.get_optimal_components(
							method='pca',
							variance_threshold=variance_threshold
						)

						st.markdown(f"**Optimal number of components:** {optimal_components} "
								   f"(explaining {variance_threshold*100:.0f}% of variance)")

						# Variance explained table
						variance_df = self.dimensionality_reducer.get_variance_explained(method='pca')
						variance_df = variance_df.head(min(10, len(variance_df)))  # Show first 10 components max

						st.subheader("Explained Variance by Component")
						st.dataframe(variance_df.style.format({
							'Explained_Variance': '{:.3f}',
							'Cumulative_Variance': '{:.3f}'
						}))

						# Biplot for first two components
						st.subheader("PCA Biplot (First Two Components)")
						fig, _ = self.dimensionality_reducer.plot_biplot(pc1=1, pc2=2, method='pca')
						st.pyplot(fig)
			else:
				st.warning("Please select variables for PCA.")

		with tabs[1]:  # FAMD Analysis
			st.subheader("Factor Analysis of Mixed Data (FAMD)")

			st.markdown("""
			Factor Analysis of Mixed Data (FAMD) is a dimensionality reduction technique designed for datasets
			containing both numeric and categorical variables. It's an extension of PCA that can handle mixed data types.
			""")

			# FAMD settings
			if self.dimensionality_reducer is not None:
				# Check if we have both numeric and categorical variables
				has_numeric = len(self.dimensionality_reducer.numeric_vars) > 0
				has_categorical = len(self.dimensionality_reducer.categorical_vars) > 0

				if has_numeric or has_categorical:
					col1, col2 = st.columns(2)

					with col1:
						# Number of components
						max_components = min(20, len(st.session_state.processed_data.columns))

						n_components = st.slider(
							"Number of components (FAMD)",
							min_value=2,
							max_value=max_components,
							value=min(8, max_components)
						)

					with col2:
						# Variance threshold
						variance_threshold = st.slider(
							"Variance threshold (%)",
							min_value=50,
							max_value=95,
							value=75,
							step=5
						) / 100  # Convert to proportion

					# Perform FAMD
					if st.button("Perform FAMD"):
						with st.spinner("Performing FAMD..."):
							try:
								# Run FAMD
								famd_results = self.dimensionality_reducer.perform_famd(
									n_components=n_components
								)

								# Store in session state
								if 'pipeline_results' not in st.session_state:
									st.session_state.pipeline_results = {}

								st.session_state.pipeline_results['dimensionality_reduction'] = {
									'method': 'famd',
									'n_components': n_components,
									'variables': list(st.session_state.processed_data.columns),
									'explained_variance': famd_results['explained_variance'].tolist()
										if isinstance(famd_results.get('explained_variance', None), (list, np.ndarray)) else None,
									'cumulative_explained_variance': famd_results['cumulative_explained_variance'].tolist()
										if isinstance(famd_results.get('cumulative_explained_variance', None), (list, np.ndarray)) else None
								}

								# Display results
								st.success(f"FAMD successfully performed with {n_components} components!")

								# Scree plot
								st.subheader("Scree Plot")
								fig, _ = self.dimensionality_reducer.plot_scree(method='famd')
								st.pyplot(fig)

								# Get optimal number of components
								optimal_components = self.dimensionality_reducer.get_optimal_components(
									method='famd',
									variance_threshold=variance_threshold
								)

								st.markdown(f"**Optimal number of components:** {optimal_components} "
										  f"(explaining {variance_threshold*100:.0f}% of variance)")

								# Variance explained table
								variance_df = self.dimensionality_reducer.get_variance_explained(method='famd')
								variance_df = variance_df.head(min(10, len(variance_df)))  # Show first 10 components max

								st.subheader("Explained Variance by Component")
								st.dataframe(variance_df.style.format({
									'Explained_Variance': '{:.3f}',
									'Cumulative_Variance': '{:.3f}'
								}))
							except Exception as e:
								st.error(f"Error during FAMD: {e}")
								logger.error(f"FAMD error: {e}", exc_info=True)
				else:
					if not has_numeric:
						st.warning("No numeric variables available for FAMD.")
					if not has_categorical:
						st.warning("No categorical variables available for FAMD.")
			else:
				st.warning("Data not available for FAMD. Please complete data imputation first.")

		with tabs[2]:  # Component Interpretation
			st.subheader("Component Interpretation")

			st.markdown("""
			Understanding what each component represents is crucial for interpreting the dimensionality
			reduction results. This tab helps visualize and interpret the principal components.
			""")

			# Select method and component
			if self.dimensionality_reducer is not None:
				col1, col2 = st.columns(2)

				with col1:
					# Select method
					method = st.selectbox(
						"Select method",
						options=["pca", "famd"],
						index=0,
						format_func=lambda x: "PCA" if x == "pca" else "FAMD"
					)

				with col2:
					# Select component to interpret
					max_components = 10  # Limit to first 10 components
					component = st.slider(
						"Select component to interpret",
						min_value=1,
						max_value=max_components,
						value=1
					)

				# Check if method results are available
				method_results_available = False

				if method == "pca" and hasattr(self.dimensionality_reducer, 'pca_results') and self.dimensionality_reducer.pca_results:
					method_results_available = True
				elif method == "famd" and hasattr(self.dimensionality_reducer, 'famd_results') and self.dimensionality_reducer.famd_results:
					method_results_available = True

				if method_results_available:
					# Number of variables to show in loading plot
					n_top = st.slider(
						"Number of top variables to show",
						min_value=5,
						max_value=20,
						value=10
					)

					# Plot component loadings
					st.subheader(f"Component {component} Loadings")
					try:
						fig, _ = self.dimensionality_reducer.plot_component_loadings(
							component=component,
							method=method,
							n_top=n_top
						)
						st.pyplot(fig)

						# Get loading values
						loadings = self.dimensionality_reducer.get_component_loadings(method=method)

						if not loadings.empty:
							component_name = f'PC{component}' if method == 'pca' else f'Dim{component}'

							if component_name in loadings.columns:
								# Sort loadings by absolute value
								abs_loadings = loadings[component_name].abs().sort_values(ascending=False)

								# Create a DataFrame with variable descriptions
								loading_df = pd.DataFrame({
									'Variable': abs_loadings.index,
									'Loading': [loadings.loc[var, component_name] for var in abs_loadings.index],
									'Abs_Loading': abs_loadings.values
								}).head(n_top)

								# Add descriptions
								loading_df['Description'] = loading_df['Variable'].apply(
									lambda var: self.data_loader.get_variable_description(var) or "No description available"
								)

								# Display the table
								st.dataframe(loading_df.style.format({
									'Loading': '{:.3f}',
									'Abs_Loading': '{:.3f}'
								}))

								# Suggest component interpretation
								st.subheader("Suggested Interpretation")

								# Get positive and negative loadings
								pos_loadings = loading_df[loading_df['Loading'] > 0].sort_values('Loading', ascending=False)
								neg_loadings = loading_df[loading_df['Loading'] < 0].sort_values('Loading')

								if not pos_loadings.empty:
									st.markdown("**Positive association:**")
									st.markdown(", ".join(pos_loadings['Variable'].head(5).tolist()))

								if not neg_loadings.empty:
									st.markdown("**Negative association:**")
									st.markdown(", ".join(neg_loadings['Variable'].head(5).tolist()))
							else:
								st.warning(f"Component {component_name} not found in loadings.")
						else:
							st.warning("No loadings available.")
					except Exception as e:
						st.error(f"Error plotting component loadings: {e}")
						logger.error(f"Component loadings error: {e}", exc_info=True)
				else:
					st.warning(f"No {method.upper()} results available. Please run {method.upper()} first.")
			else:
				st.warning("Data not available for component interpretation. Please complete dimensionality reduction first.")

		with tabs[3]:  # Transformed Data
			st.subheader("Transformed Data")

			st.markdown("""
			This tab allows you to transform the original data into the reduced dimensionality space
			and examine the resulting dataset.
			""")

			# Select method and number of components
			if self.dimensionality_reducer is not None:
				col1, col2 = st.columns(2)

				with col1:
					# Select method
					method = st.selectbox(
						"Select transformation method",
						options=["pca", "famd"],
						index=0,
						format_func=lambda x: "PCA" if x == "pca" else "FAMD"
					)

				with col2:
					# Check if method results are available
					method_results_available = False
					max_components = 10  # Default max

					if method == "pca" and hasattr(self.dimensionality_reducer, 'pca_results') and self.dimensionality_reducer.pca_results:
						method_results_available = True
						max_components = min(10, len(self.dimensionality_reducer.pca_results.get('explained_variance_ratio', [])))
					elif method == "famd" and hasattr(self.dimensionality_reducer, 'famd_results') and self.dimensionality_reducer.famd_results:
						method_results_available = True
						max_components = min(10, len(self.dimensionality_reducer.famd_results.get('explained_variance', [])))

					# Select number of components
					if method_results_available:
						n_components = st.slider(
							"Number of components to keep",
							min_value=2,
							max_value=max_components,
							value=min(5, max_components)
						)
					else:
						n_components = 5  # Default value

				if method_results_available:
					# Get optimal components based on variance threshold
					variance_threshold = 0.75  # Default 75% threshold

					optimal_components = self.dimensionality_reducer.get_optimal_components(
						method=method,
						variance_threshold=variance_threshold
					)

					st.markdown(f"**Optimal number of components:** {optimal_components} "
							   f"(explaining {variance_threshold*100:.0f}% of variance)")

					# Transform data
					if st.button("Transform Data"):
						with st.spinner("Transforming data..."):
							try:
								# Transform data to components
								transformed_data = self.dimensionality_reducer.transform_data(
									method=method,
									n_components=n_components
								)

								if not transformed_data.empty:
									# Display transformed data
									st.subheader("Transformed Data (First 10 rows)")
									st.dataframe(transformed_data.head(10))

									# Visualize transformed data (first two components)
									st.subheader("Visualization of First Two Components")

									if len(transformed_data.columns) >= 2:
										# Create scatter plot
										fig = px.scatter(
											transformed_data,
											x=transformed_data.columns[0],
											y=transformed_data.columns[1],
											title=f'First Two Components ({method.upper()})',
											color_discrete_sequence=[COLOR_PALETTE['primary']]
										)

										st.plotly_chart(fig, use_container_width=True)

										# Add treatment group coloring if available
										if self.treatment_groups is not None and 'tx.group' in self.data.columns:
											# Copy transformed data and add treatment group
											vis_data = transformed_data.copy()
											vis_data['Treatment Group'] = self.data['tx.group'].map({
												0: 'Control (Sham)',
												1: 'tDCS + Meditation',
												2: 'tDCS Only',
												3: 'Meditation Only'
											})

											# Create scatter plot with treatment groups
											fig = px.scatter(
												vis_data,
												x=vis_data.columns[0],
												y=vis_data.columns[1],
												color='Treatment Group',
												title=f'First Two Components by Treatment Group ({method.upper()})',
												color_discrete_map=TREATMENT_COLORS
											)

											st.plotly_chart(fig, use_container_width=True)

									# Option to save transformed data
									if st.button("Save Transformed Data to Session"):
										# Store transformed data in session state
										st.session_state.processed_data = transformed_data

										# Update pipeline results
										if 'pipeline_results' not in st.session_state:
											st.session_state.pipeline_results = {}

										st.session_state.pipeline_results['dimensionality_reduction']['transformed_data_shape'] = transformed_data.shape

										st.success(f"Transformed data with {n_components} components saved to session. "
												f"Shape: {transformed_data.shape}")
								else:
									st.warning("No transformed data generated.")
							except Exception as e:
								st.error(f"Error transforming data: {e}")
								logger.error(f"Data transformation error: {e}", exc_info=True)
				else:
					st.warning(f"No {method.upper()} results available. Please run {method.upper()} first.")
			else:
				st.warning("Data not available for transformation. Please complete dimensionality reduction first.")

	def _render_data_quality(self):
		"""Render data quality enhancement tools."""
		st.header("Data Quality Enhancement")

		# Initialize data quality enhancer if not done and data is available
		if self.data_quality_enhancer is None and st.session_state.processed_data is not None:
			self.data_quality_enhancer = DataQualityEnhancer(st.session_state.processed_data)

		# Tab navigation for data quality
		tabs = st.tabs(["Outlier Detection", "Distribution Analysis", "Transformations", "Standardization"])

		with tabs[0]:  # Outlier Detection
			st.subheader("Outlier Detection")

			st.markdown("""
			Outliers are observations that deviate significantly from other observations and can affect statistical
			analyses. This tool helps identify outliers using different methods.
			""")

			# Outlier detection settings
			if self.data_quality_enhancer is not None:
				col1, col2 = st.columns(2)

				with col1:
					# Select method
					method = st.selectbox(
						"Select outlier detection method",
						options=["iqr", "zscore", "modified_zscore"],
						index=0,
						format_func=lambda x: {
							"iqr": "Interquartile Range (IQR)",
							"zscore": "Z-Score",
							"modified_zscore": "Modified Z-Score"
						}.get(x, x)
					)

				with col2:
					# Threshold for outlier detection
					if method == "iqr":
						threshold = st.slider(
							"IQR multiplier",
							min_value=1.0,
							max_value=3.0,
							value=1.5,
							step=0.1,
							help="Values beyond median ± (threshold × IQR) are considered outliers."
						)
					elif method == "zscore":
						threshold = st.slider(
							"Z-score threshold",
							min_value=2.0,
							max_value=4.0,
							value=3.0,
							step=0.1,
							help="Values with absolute z-score above threshold are considered outliers."
						)
					else:  # modified_zscore
						threshold = st.slider(
							"Modified z-score threshold",
							min_value=2.0,
							max_value=5.0,
							value=3.5,
							step=0.1,
							help="Values with absolute modified z-score above threshold are considered outliers."
						)

				# Select variables for outlier detection
				numeric_vars = self.data_quality_enhancer.numeric_vars

				selected_vars = st.multiselect(
					"Select variables for outlier detection",
					options=numeric_vars,
					default=numeric_vars[:5] if len(numeric_vars) >= 5 else numeric_vars
				)

				# Detect outliers
				if selected_vars and st.button("Detect Outliers"):
					with st.spinner("Detecting outliers..."):
						try:
							# Detect outliers for selected variables
							outliers = self.data_quality_enhancer.detect_outliers(
								method=method,
								threshold=threshold,
								variables=selected_vars
							)

							# Store in pipeline results
							if 'pipeline_results' not in st.session_state:
								st.session_state.pipeline_results = {}

							if 'data_quality' not in st.session_state.pipeline_results:
								st.session_state.pipeline_results['data_quality'] = {}

							st.session_state.pipeline_results['data_quality']['outliers'] = {
								'method': method,
								'threshold': threshold,
								'variables': selected_vars,
								'summary': outliers.get('summary', {})
							}

							# Display results
							summary = outliers.get('summary', {})

							if 'total_outliers_detected' in summary:
								outlier_count = summary['total_outliers_detected']

								if outlier_count > 0:
									st.warning(f"Detected {outlier_count} outliers across {summary.get('variables_with_outliers', 0)} variables.")
								else:
									st.success("No outliers detected.")

								# Display outlier counts by variable
								if 'variable_outlier_counts' in summary:
									counts = summary['variable_outlier_counts']

									if counts:
										# Create a DataFrame for display
										counts_df = pd.DataFrame({
											'Variable': list(counts.keys()),
											'Outliers': list(counts.values())
										}).sort_values('Outliers', ascending=False)

										st.dataframe(counts_df)

										# Create bar chart of outlier counts
										fig = px.bar(
											counts_df,
											x='Outliers',
											y='Variable',
											orientation='h',
											title='Outlier Counts by Variable',
											color='Outliers',
											color_continuous_scale='RdYlGn_r'
										)

										st.plotly_chart(fig, use_container_width=True)

										# Select a variable to examine
										st.subheader("Examine Variable Outliers")

										selected_outlier_var = st.selectbox(
											"Select a variable to examine outliers",
											options=[var for var in counts_df['Variable'] if counts_df.loc[counts_df['Variable'] == var, 'Outliers'].iloc[0] > 0],
											index=0 if len(counts_df[counts_df['Outliers'] > 0]) > 0 else None
										)

										if selected_outlier_var:
											# Plot outliers for selected variable
											fig, _ = self.data_quality_enhancer.plot_outliers(
												variable=selected_outlier_var,
												method=method,
												threshold=threshold
											)

											st.pyplot(fig)

											# Get details about outliers
											details = outliers.get('details', {}).get(selected_outlier_var, {})

											if details:
												# Display outlier indices and values
												outlier_indices = details.get('outlier_indices', [])

												if outlier_indices:
													# Get outlier values
													outlier_values = [st.session_state.processed_data.iloc[i][selected_outlier_var]
																	 for i in outlier_indices if i < len(st.session_state.processed_data)]

													# Create DataFrame
													outlier_df = pd.DataFrame({
														'Index': outlier_indices,
														'Value': outlier_values
													})

													st.dataframe(outlier_df)
							else:
								st.info("No outlier summary available.")
						except Exception as e:
							st.error(f"Error detecting outliers: {e}")
							logger.error(f"Outlier detection error: {e}", exc_info=True)
			else:
				st.warning("Data not available for outlier detection. Please complete previous steps first.")

		with tabs[1]:  # Distribution Analysis
			st.subheader("Distribution Analysis")

			st.markdown("""
			Analyzing variable distributions helps understand data characteristics and identify transformations
			that may improve model performance. This tool evaluates normality, skewness, and other distribution properties.
			""")

			# Distribution analysis settings
			if self.data_quality_enhancer is not None:
				# Select variables for analysis
				numeric_vars = self.data_quality_enhancer.numeric_vars

				selected_vars = st.multiselect(
					"Select variables for distribution analysis",
					options=numeric_vars,
					default=numeric_vars[:5] if len(numeric_vars) >= 5 else numeric_vars
				)

				# Analyze distributions
				if selected_vars and st.button("Analyze Distributions"):
					with st.spinner("Analyzing distributions..."):
						try:
							# Analyze distributions for selected variables
							distributions = self.data_quality_enhancer.analyze_distributions(
								variables=selected_vars
							)

							# Store in pipeline results
							if 'pipeline_results' not in st.session_state:
								st.session_state.pipeline_results = {}

							if 'data_quality' not in st.session_state.pipeline_results:
								st.session_state.pipeline_results['data_quality'] = {}

							st.session_state.pipeline_results['data_quality']['distributions'] = {
								'variables': selected_vars,
								'summary': distributions.get('summary', {})
							}

							# Display results
							summary = distributions.get('summary', {})
							details = distributions.get('details', {})

							if details:
								# Create a DataFrame for display
								dist_df = pd.DataFrame({
									'Variable': list(details.keys()),
									'Mean': [details[var]['mean'] for var in details],
									'Median': [details[var]['median'] for var in details],
									'Std Dev': [details[var]['std'] for var in details],
									'Skewness': [details[var]['skewness'] for var in details],
									'Kurtosis': [details[var]['kurtosis'] for var in details],
									'Normal (p>0.05)': [details[var]['is_normal'] for var in details],
									'Shapiro p-value': [details[var]['shapiro_p'] for var in details]
								}).sort_values('Skewness', key=abs, ascending=False)

								# Display the table
								st.dataframe(dist_df.style.format({
									'Mean': '{:.3f}',
									'Median': '{:.3f}',
									'Std Dev': '{:.3f}',
									'Skewness': '{:.3f}',
									'Kurtosis': '{:.3f}',
									'Shapiro p-value': '{:.4f}'
								}))

								# Summary statistics
								col1, col2, col3 = st.columns(3)

								col1.metric(
									"Variables with Normal Distribution",
									f"{summary.get('normal_variables', 0)} ({summary.get('normal_variables', 0) / len(details) * 100:.1f}%)"
								)

								col2.metric(
									"Variables with Skewness > 1",
									f"{summary.get('skewed_variables', 0)} ({summary.get('skewed_variables', 0) / len(details) * 100:.1f}%)"
								)

								col3.metric(
									"Variables with Skewness > 2",
									f"{summary.get('highly_skewed_variables', 0)} ({summary.get('highly_skewed_variables', 0) / len(details) * 100:.1f}%)"
								)

								# Select a variable to examine
								st.subheader("Examine Variable Distribution")

								selected_dist_var = st.selectbox(
									"Select a variable to examine distribution",
									options=list(details.keys()),
									index=0
								)

								if selected_dist_var:
									# Plot distribution for selected variable
									fig, _ = self.data_quality_enhancer.plot_distribution(
										variable=selected_dist_var,
										original=True,
										transformed=False
									)

									st.pyplot(fig)

									# Display variable details
									var_details = details.get(selected_dist_var, {})

									if var_details:
										st.subheader("Distribution Statistics")

										col1, col2 = st.columns(2)

										with col1:
											st.markdown("**Basic Statistics:**")
											st.markdown(f"- Mean: {var_details.get('mean', 0):.3f}")
											st.markdown(f"- Median: {var_details.get('median', 0):.3f}")
											st.markdown(f"- Std Dev: {var_details.get('std', 0):.3f}")
											st.markdown(f"- Min: {var_details.get('min', 0):.3f}")
											st.markdown(f"- Max: {var_details.get('max', 0):.3f}")

										with col2:
											st.markdown("**Distribution Shape:**")
											st.markdown(f"- Skewness: {var_details.get('skewness', 0):.3f}")
											st.markdown(f"- Kurtosis: {var_details.get('kurtosis', 0):.3f}")
											st.markdown(f"- Normality (Shapiro p-value): {var_details.get('shapiro_p', 0):.4f}")

											is_normal = var_details.get('is_normal', False)
											st.markdown(f"- Normal Distribution: {'✅ Yes' if is_normal else '❌ No'}")
							else:
								st.info("No distribution details available.")
						except Exception as e:
							st.error(f"Error analyzing distributions: {e}")
							logger.error(f"Distribution analysis error: {e}", exc_info=True)
			else:
				st.warning("Data not available for distribution analysis. Please complete previous steps first.")

		with tabs[2]:  # Transformations
			st.subheader("Variable Transformations")

			st.markdown("""
			Variable transformations can improve model performance by making distributions more symmetric or
			linear relationships more apparent. This tool recommends and applies appropriate transformations.
			""")

			# Variable transformation settings
			if self.data_quality_enhancer is not None:
				# Get transformation recommendations
				if st.button("Get Transformation Recommendations"):
					with st.spinner("Analyzing variables and recommending transformations..."):
						try:
							# Get recommendations
							transformations = self.data_quality_enhancer.recommend_transformations()

							# Store in pipeline results
							if 'pipeline_results' not in st.session_state:
								st.session_state.pipeline_results = {}

							if 'data_quality' not in st.session_state.pipeline_results:
								st.session_state.pipeline_results['data_quality'] = {}

							st.session_state.pipeline_results['data_quality']['transformations'] = {
								'summary': transformations.get('summary', {})
							}

							# Display results
							recommendations = transformations.get('recommendations', {})
							summary = transformations.get('summary', {})

							if recommendations:
								# Display summary
								col1, col2 = st.columns(2)

								col1.metric(
									"Variables Analyzed",
									summary.get('variables_analyzed', 0)
								)

								col2.metric(
									"Variables Needing Transformation",
									summary.get('variables_needing_transformation', 0)
								)

								# Create a DataFrame for display
								rec_df = pd.DataFrame({
									'Variable': list(recommendations.keys()),
									'Recommendation': [recommendations[var]['recommendation'] for var in recommendations],
									'Reason': [recommendations[var]['reason'] for var in recommendations],
									'Skewness': [recommendations[var]['skewness'] for var in recommendations],
									'Is Normal': [recommendations[var]['is_normal'] for var in recommendations]
								})

								# Filter to show only variables needing transformation
								transform_vars = rec_df[rec_df['Recommendation'] != 'none']

								if not transform_vars.empty:
									st.subheader("Recommended Transformations")
									st.dataframe(transform_vars.style.format({
										'Skewness': '{:.3f}'
									}))

									# Display transformation counts
									counts = summary.get('recommendation_counts', {})

									if counts:
										# Create a DataFrame for display
										counts_df = pd.DataFrame({
											'Transformation': list(counts.keys()),
											'Count': list(counts.values())
										}).sort_values('Count', ascending=False)

										# Create a pie chart
										fig = px.pie(
											counts_df,
											values='Count',
											names='Transformation',
											title='Recommendation Distribution',
											color_discrete_sequence=px.colors.qualitative.Set3
										)

										st.plotly_chart(fig, use_container_width=True)

									# Apply transformations
									if st.button("Apply Recommended Transformations"):
										with st.spinner("Applying transformations..."):
											# Get transformations to apply
											transformations_to_apply = {
												var: rec['recommendation']
												for var, rec in recommendations.items()
												if rec['recommendation'] != 'none'
											}

											# Apply transformations
											transformed_data = self.data_quality_enhancer.apply_transformations(
												transformations=transformations_to_apply
											)

											# Update session state
											st.session_state.processed_data = transformed_data

											# Update pipeline results
											applied = self.data_quality_enhancer.results['transformations'].get('applied', {})

											if 'details' in applied:
												st.session_state.pipeline_results['data_quality']['transformations']['applied'] = {
													'variables': list(applied['details'].keys()),
													'summary': applied.get('summary', {})
												}

											st.success(f"Successfully applied transformations to {len(transformations_to_apply)} variables.")
								else:
									st.success("No variables require transformation.")

								# Select a variable to examine
								st.subheader("Examine Transformation Effect")

								selected_transform_var = st.selectbox(
									"Select a variable to examine transformation",
									options=[var for var in recommendations if recommendations[var]['recommendation'] != 'none'],
									index=0 if transform_vars.shape[0] > 0 else None
								)

								if selected_transform_var:
									# Get recommended transformation
									transform_type = recommendations[selected_transform_var]['recommendation']

									# Plot transformation effect
									fig, _ = self.data_quality_enhancer.plot_distribution(
										variable=selected_transform_var,
										original=True,
										transformed=True,
										transformation_type=transform_type
									)

									st.pyplot(fig)
							else:
								st.info("No transformation recommendations available.")
						except Exception as e:
							st.error(f"Error recommending transformations: {e}")
							logger.error(f"Transformation recommendation error: {e}", exc_info=True)
			else:
				st.warning("Data not available for transformations. Please complete previous steps first.")

		with tabs[3]:  # Standardization
			st.subheader("Variable Standardization")

			st.markdown("""
			Standardization rescales variables to have similar ranges, which is important for many machine learning
			algorithms. This tool provides different standardization methods.
			""")

			# Standardization settings
			if self.data_quality_enhancer is not None:
				col1, col2 = st.columns(2)

				with col1:
					# Select method
					method = st.selectbox(
						"Select standardization method",
						options=["zscore", "robust", "minmax"],
						index=0,
						format_func=lambda x: {
							"zscore": "Z-Score (mean=0, std=1)",
							"robust": "Robust (median=0, IQR=1)",
							"minmax": "Min-Max Scaling (0-1)"
						}.get(x, x)
					)

				with col2:
					# Select variables to standardize
					standardize_all = st.checkbox("Standardize all numeric variables", value=True)

				# Select specific variables if not standardizing all
				numeric_vars = self.data_quality_enhancer.numeric_vars

				if not standardize_all:
					selected_vars = st.multiselect(
						"Select variables to standardize",
						options=numeric_vars,
						default=numeric_vars[:5] if len(numeric_vars) >= 5 else numeric_vars
					)
				else:
					selected_vars = numeric_vars

				# Standardize variables
				if selected_vars and st.button("Standardize Variables"):
					with st.spinner("Standardizing variables..."):
						try:
							# Standardize selected variables
							standardized_data = self.data_quality_enhancer.standardize_variables(
								variables=selected_vars,
								method=method
							)

							# Store in pipeline results
							if 'pipeline_results' not in st.session_state:
								st.session_state.pipeline_results = {}

							if 'data_quality' not in st.session_state.pipeline_results:
								st.session_state.pipeline_results['data_quality'] = {}

							st.session_state.pipeline_results['data_quality']['standardization'] = {
								'method': method,
								'variables': selected_vars
							}

							# Update session state
							st.session_state.processed_data = standardized_data

							st.success(f"Successfully standardized {len(selected_vars)} variables using {method} method.")

							# Display before/after comparison for a variable
							st.subheader("Before/After Standardization")

							# Select a variable to examine
							sample_var = st.selectbox(
								"Select a variable to examine",
								options=selected_vars,
								index=0
							)

							if sample_var:
								col1, col2 = st.columns(2)

								with col1:
									st.markdown("### Before Standardization")

									# Original statistics
									orig_data = self.data[sample_var].dropna()

									st.metric("Mean", f"{orig_data.mean():.2f}")
									st.metric("Median", f"{orig_data.median():.2f}")
									st.metric("Std Dev", f"{orig_data.std():.2f}")
									st.metric("Min", f"{orig_data.min():.2f}")
									st.metric("Max", f"{orig_data.max():.2f}")

									# Histogram before standardization
									fig = px.histogram(
										orig_data,
										title=f'{sample_var} (Before)',
										color_discrete_sequence=[COLOR_PALETTE['primary']]
									)

									st.plotly_chart(fig, use_container_width=True)

								with col2:
									st.markdown("### After Standardization")

									# Standardized statistics
									std_data = standardized_data[sample_var].dropna()

									st.metric("Mean", f"{std_data.mean():.2f}")
									st.metric("Median", f"{std_data.median():.2f}")
									st.metric("Std Dev", f"{std_data.std():.2f}")
									st.metric("Min", f"{std_data.min():.2f}")
									st.metric("Max", f"{std_data.max():.2f}")

									# Histogram after standardization
									fig = px.histogram(
										std_data,
										title=f'{sample_var} (After {method})',
										color_discrete_sequence=[COLOR_PALETTE['secondary']]
									)

									st.plotly_chart(fig, use_container_width=True)
						except Exception as e:
							st.error(f"Error standardizing variables: {e}")
							logger.error(f"Standardization error: {e}", exc_info=True)
			else:
				st.warning("Data not available for standardization. Please complete previous steps first.")

	def _render_treatment_groups(self):
		"""Render treatment group analysis tools."""
		st.header("Treatment Group Analysis")

		# Get treatment groups if not already done
		if self.treatment_groups is None:
			self.treatment_groups = self.data_loader.get_treatment_groups()

		# Tab navigation for treatment groups
		tabs = st.tabs(["Group Overview", "Baseline Characteristics", "Outcome Comparison", "Group Balance"])

		with tabs[0]:  # Group Overview
			st.subheader("Treatment Group Overview")

			if self.treatment_groups:
				# Create a bar chart of treatment group sizes
				treatment_sizes = {group: len(df) for group, df in self.treatment_groups.items()}

				# Skip 'Experimental' since it's equivalent to 'tDCS + Meditation'
				if 'Experimental' in treatment_sizes:
					del treatment_sizes['Experimental']

				# Focus on the 2x2 factorial design groups
				factorial_groups = {
					'Control (No tDCS, No Meditation)': treatment_sizes.get('Control (No tDCS, No Meditation)', 0),
					'tDCS Only': treatment_sizes.get('tDCS Only', 0),
					'Meditation Only': treatment_sizes.get('Meditation Only', 0),
					'tDCS + Meditation': treatment_sizes.get('tDCS + Meditation', 0)
				}

				# Create a bar chart
				fig = px.bar(
					x=list(factorial_groups.keys()),
					y=list(factorial_groups.values()),
					color=list(factorial_groups.keys()),
					color_discrete_map={
						'Control (No tDCS, No Meditation)': TREATMENT_COLORS['Control (No tDCS, No Meditation)'],
						'tDCS Only': TREATMENT_COLORS['tDCS Only'],
						'Meditation Only': TREATMENT_COLORS['Meditation Only'],
						'tDCS + Meditation': TREATMENT_COLORS['tDCS + Meditation']
					},
					labels={'x': 'Treatment Group', 'y': 'Number of Participants'},
					title='Treatment Group Sizes'
				)

				fig.update_layout(showlegend=False)
				st.plotly_chart(fig, use_container_width=True)

				# 2x2 grid showing treatment groups
				st.subheader("2×2 Factorial Design")

				col1, col2, col3 = st.columns([1, 2, 1])

				with col2:
					# Create a heatmap
					fig = px.imshow(
						[[factorial_groups['Control (No tDCS, No Meditation)'], factorial_groups['tDCS Only']],
						 [factorial_groups['Meditation Only'], factorial_groups['tDCS + Meditation']]],
						x=['No tDCS', 'tDCS'],
						y=['No Meditation', 'Meditation'],
						color_continuous_scale='Blues',
						labels=dict(x="tDCS Treatment", y="Meditation Treatment", color="Participants"),
						text_auto=True
					)

					fig.update_layout(title='2×2 Factorial Design')
					st.plotly_chart(fig, use_container_width=True)

				# Group descriptions
				st.subheader("Treatment Group Descriptions")

				col1, col2 = st.columns(2)

				with col1:
					st.markdown("**Control Group (No tDCS, No Meditation)**")
					st.markdown("Participants in this group received neither tDCS nor meditation intervention.")

					st.markdown("**tDCS Only Group**")
					st.markdown("Participants in this group received transcranial direct current stimulation (tDCS) without meditation intervention.")

				with col2:
					st.markdown("**Meditation Only Group**")
					st.markdown("Participants in this group received meditation intervention without tDCS.")

					st.markdown("**tDCS + Meditation Group**")
					st.markdown("Participants in this group received both tDCS and meditation interventions.")
			else:
				st.warning("Treatment group data not available.")

		with tabs[1]:  # Baseline Characteristics
			st.subheader("Baseline Characteristics by Treatment Group")

			if self.treatment_groups:
				# Select variables to compare
				st.markdown("### Select Variables")
				st.markdown("Choose baseline characteristics to compare across treatment groups:")

				# Get baseline variables
				variable_categories = self.data_loader.get_variable_categories()

				if variable_categories:
					baseline_vars = variable_categories.get('baseline', [])
					demographic_vars = variable_categories.get('demographic', [])

					# Combine and remove duplicates
					compare_vars = list(set(baseline_vars + demographic_vars))

					# Add variables containing ".0" or "_0" as they're likely baseline
					for var in self.data.columns:
						if ".0" in var or "_0" in var:
							compare_vars.append(var)

					# Remove duplicates and sort
					compare_vars = sorted(list(set(compare_vars)))
				else:
					# Default to variables with "0" in the name (likely baseline)
					compare_vars = [var for var in self.data.columns if "0" in var]

				selected_vars = st.multiselect(
					"Select baseline variables to compare",
					options=compare_vars,
					default=[var for var in ['Age', 'History.Age', 'History.Gender', 'WOMAC.Pain.0', 'WOMAC.Total1.0', 'NRS.Average.Daily.0']
							 if var in compare_vars][:5]
				)

				# Select groups to compare
				groups_to_compare = st.multiselect(
					"Select groups to compare",
					options=list(self.treatment_groups.keys()),
					default=[group for group in self.treatment_groups.keys()
							if group not in ['Experimental']]  # Exclude 'Experimental' by default
				)

				# Compare baseline characteristics
				if selected_vars and groups_to_compare and st.button("Compare Baseline Characteristics"):
					with st.spinner("Analyzing baseline characteristics..."):
						try:
							# Create a DataFrame to store comparison results
							comparison_data = []

							for var in selected_vars:
								var_data = {'Variable': var}

								for group in groups_to_compare:
									group_df = self.treatment_groups[group]

									if var in group_df.columns:
										# Check if numeric
										if pd.api.types.is_numeric_dtype(group_df[var]):
											# Calculate mean and std
											mean = group_df[var].mean()
											std = group_df[var].std()
											var_data[group] = f"{mean:.2f} ± {std:.2f}"
										else:
											# For categorical, get most common value and percentage
											value_counts = group_df[var].value_counts()
											if not value_counts.empty:
												top_value = value_counts.index[0]
												top_count = value_counts.iloc[0]
												pct = top_count / len(group_df) * 100
												var_data[group] = f"{top_value} ({pct:.1f}%)"
											else:
												var_data[group] = "N/A"
									else:
										var_data[group] = "N/A"

								comparison_data.append(var_data)

							# Create a DataFrame
							comparison_df = pd.DataFrame(comparison_data)

							# Display the table
							st.dataframe(comparison_df)

							# Create visualizations for selected variables
							st.subheader("Visualizations")

							for var in selected_vars:
								if var in self.data.columns:
									# Check if numeric
									if pd.api.types.is_numeric_dtype(self.data[var]):
										# Create a box plot
										var_data = []

										for group in groups_to_compare:
											group_df = self.treatment_groups[group]

											if var in group_df.columns:
												for value in group_df[var].dropna():
													var_data.append({
														'Group': group,
														'Value': value
													})

										if var_data:
											var_df = pd.DataFrame(var_data)

											# Get variable description
											description = self.data_loader.get_variable_description(var) or var

											fig = px.box(
												var_df,
												x='Group',
												y='Value',
												color='Group',
												title=f'{description} by Treatment Group',
												color_discrete_map=TREATMENT_COLORS
											)

											st.plotly_chart(fig, use_container_width=True)
									else:
										# For categorical, create a stacked bar chart
										var_data = []

										for group in groups_to_compare:
											group_df = self.treatment_groups[group]

											if var in group_df.columns:
												value_counts = group_df[var].value_counts(normalize=True)

												for value, count in value_counts.items():
													var_data.append({
														'Group': group,
														'Value': str(value),
														'Percentage': count * 100
													})

										if var_data:
											var_df = pd.DataFrame(var_data)

											# Get variable description
											description = self.data_loader.get_variable_description(var) or var

											fig = px.bar(
												var_df,
												x='Group',
												y='Percentage',
												color='Value',
												title=f'{description} by Treatment Group',
												color_discrete_sequence=px.colors.qualitative.Set3
											)

											st.plotly_chart(fig, use_container_width=True)
						except Exception as e:
							st.error(f"Error comparing baseline characteristics: {e}")
							logger.error(f"Baseline comparison error: {e}", exc_info=True)
			else:
				st.warning("Treatment group data not available.")

		with tabs[2]:  # Outcome Comparison
			st.subheader("Outcome Comparison Across Treatment Groups")

			if self.treatment_groups:
				# Select outcome variables
				st.markdown("### Select Outcome Variables")
				st.markdown("Choose outcome variables to compare across treatment groups:")

				# Get outcome variables
				variable_categories = self.data_loader.get_variable_categories()

				if variable_categories:
					outcome_vars = variable_categories.get('outcome', [])
				else:
					# Default to variables with "differ" or "change" in the name
					outcome_vars = [var for var in self.data.columns if "differ" in var.lower() or "change" in var.lower()]

				# Add variables with specific patterns that might indicate outcomes
				for pattern in ['M1', 'M2', 'M3', '10', 'differ', 'Differ']:
					for var in self.data.columns:
						if pattern in var:
							outcome_vars.append(var)

				# Remove duplicates and sort
				outcome_vars = sorted(list(set(outcome_vars)))

				selected_outcomes = st.multiselect(
					"Select outcome variables to compare",
					options=outcome_vars,
					default=[var for var in ['WOMAC.Pain.Differ', 'WOMAC.Total.differ', 'NRS.Average.differ']
							 if var in outcome_vars][:3]
				)

				# Select groups to compare
				groups_to_compare = st.multiselect(
					"Select groups to compare",
					options=list(self.treatment_groups.keys()),
					default=[group for group in self.treatment_groups.keys()
							if group not in ['Experimental']]  # Exclude 'Experimental' by default
				)

				# Compare outcomes
				if selected_outcomes and groups_to_compare and st.button("Compare Outcomes"):
					with st.spinner("Analyzing outcomes..."):
						try:
							# Create visualizations for selected outcomes
							for var in selected_outcomes:
								if var in self.data.columns:
									# Check if numeric
									if pd.api.types.is_numeric_dtype(self.data[var]):
										# Create a box plot
										var_data = []

										for group in groups_to_compare:
											group_df = self.treatment_groups[group]

											if var in group_df.columns:
												for value in group_df[var].dropna():
													var_data.append({
														'Group': group,
														'Value': value
													})

										if var_data:
											var_df = pd.DataFrame(var_data)

											# Get variable description
											description = self.data_loader.get_variable_description(var) or var

											fig = px.box(
												var_df,
												x='Group',
												y='Value',
												color='Group',
												title=f'{description} by Treatment Group',
												color_discrete_map=TREATMENT_COLORS
											)

											st.plotly_chart(fig, use_container_width=True)

											# Add statistics
											st.subheader(f"Statistics for {var}")

											stats_data = []
											for group in groups_to_compare:
												group_df = self.treatment_groups[group]

												if var in group_df.columns:
													values = group_df[var].dropna()

													stats_data.append({
														'Group': group,
														'N': len(values),
														'Mean': values.mean(),
														'Median': values.median(),
														'Std Dev': values.std(),
														'Min': values.min(),
														'Max': values.max()
													})

											stats_df = pd.DataFrame(stats_data)

											st.dataframe(stats_df.style.format({
												'Mean': '{:.2f}',
												'Median': '{:.2f}',
												'Std Dev': '{:.2f}',
												'Min': '{:.2f}',
												'Max': '{:.2f}'
											}))

											# Simple statistical test (ANOVA)
											if len(groups_to_compare) > 1:
												try:


													# Prepare data for ANOVA
													anova_data = []
													for group in groups_to_compare:
														group_df = self.treatment_groups[group]
														if var in group_df.columns:
															values = group_df[var].dropna().tolist()
															if values:
																anova_data.append(values)

													if len(anova_data) > 1 and all(len(data) > 0 for data in anova_data):
														# Perform one-way ANOVA
														f_stat, p_value = scipy_stats.f_oneway(*anova_data)

														st.markdown(f"**One-way ANOVA:** F = {f_stat:.3f}, p-value = {p_value:.4f}")

														if p_value < 0.05:
															st.markdown("**Result:** There is a statistically significant difference between groups (p < 0.05).")
														else:
															st.markdown("**Result:** There is no statistically significant difference between groups (p ≥ 0.05).")
												except Exception as e:
													st.warning(f"Could not perform statistical test: {e}")
									else:
										st.warning(f"Variable {var} is not numeric and cannot be visualized as an outcome.")
								else:
									st.warning(f"Variable {var} not found in the dataset.")
						except Exception as e:
							st.error(f"Error comparing outcomes: {e}")
							logger.error(f"Outcome comparison error: {e}", exc_info=True)
			else:
				st.warning("Treatment group data not available.")

		with tabs[3]:  # Group Balance
			st.subheader("Treatment Group Balance Assessment")

			if self.treatment_groups:
				st.markdown("""
				Assessing balance between treatment groups is crucial in randomized controlled trials.
				This tool helps evaluate if baseline characteristics are well-balanced across groups.
				""")

				# Select variables for balance assessment
				st.markdown("### Select Variables")
				st.markdown("Choose baseline characteristics to assess balance:")

				# Get baseline variables (similar to baseline characteristics tab)
				variable_categories = self.data_loader.get_variable_categories()

				if variable_categories:
					baseline_vars = variable_categories.get('baseline', [])
					demographic_vars = variable_categories.get('demographic', [])

					# Combine and remove duplicates
					balance_vars = list(set(baseline_vars + demographic_vars))

					# Add variables containing ".0" or "_0" as they're likely baseline
					for var in self.data.columns:
						if ".0" in var or "_0" in var:
							balance_vars.append(var)

					# Remove duplicates and sort
					balance_vars = sorted(list(set(balance_vars)))
				else:
					# Default to variables with "0" in the name (likely baseline)
					balance_vars = [var for var in self.data.columns if "0" in var]

				selected_vars = st.multiselect(
					"Select baseline variables to assess balance",
					options=balance_vars,
					default=[var for var in ['Age', 'History.Age', 'History.Gender', 'WOMAC.Pain.0', 'WOMAC.Total1.0', 'NRS.Average.Daily.0']
							 if var in balance_vars][:5]
				)

				# Select groups to compare
				groups_to_compare = st.multiselect(
					"Select groups to compare",
					options=list(self.treatment_groups.keys()),
					default=[group for group in self.treatment_groups.keys()
							if group not in ['Experimental']]  # Exclude 'Experimental' by default
				)

				# Assess balance
				if selected_vars and groups_to_compare and st.button("Assess Group Balance"):
					with st.spinner("Assessing group balance..."):
						try:
							# Calculate standardized mean differences
							balance_results = []

							for var in selected_vars:
								if var in self.data.columns and pd.api.types.is_numeric_dtype(self.data[var]):
									var_data = {'Variable': var}

									# Get reference group (first group)
									ref_group = groups_to_compare[0]
									ref_values = self.treatment_groups[ref_group][var].dropna()

									if len(ref_values) > 0:
										ref_mean = ref_values.mean()
										ref_std = ref_values.std()

										var_data['Reference Group'] = ref_group
										var_data['Reference Mean'] = ref_mean
										var_data['Reference Std'] = ref_std

										# Calculate SMD for each comparison group
										for group in groups_to_compare[1:]:
											group_values = self.treatment_groups[group][var].dropna()

											if len(group_values) > 0:
												group_mean = group_values.mean()
												group_std = group_values.std()

												# Calculate pooled standard deviation
												n1 = len(ref_values)
												n2 = len(group_values)
												pooled_std = np.sqrt(((n1 - 1) * ref_std**2 + (n2 - 1) * group_std**2) / (n1 + n2 - 2))

												# Calculate standardized mean difference
												if pooled_std > 0:
													smd = abs((group_mean - ref_mean) / pooled_std)
												else:
													smd = 0

												var_data[f'SMD vs {group}'] = smd
											else:
												var_data[f'SMD vs {group}'] = np.nan

										balance_results.append(var_data)

							if balance_results:
								# Create a DataFrame
								balance_df = pd.DataFrame(balance_results)

								# Display the table
								st.dataframe(balance_df.style.format({
									'Reference Mean': '{:.2f}',
									'Reference Std': '{:.2f}',
									**{f'SMD vs {group}': '{:.3f}' for group in groups_to_compare[1:]}
								}))

								# Visualize SMDs
								smd_data = []

								for _, row in balance_df.iterrows():
									var = row['Variable']

									for group in groups_to_compare[1:]:
										col_name = f'SMD vs {group}'

										if col_name in row and not pd.isna(row[col_name]):
											smd_data.append({
												'Variable': var,
												'Group': group,
												'SMD': row[col_name]
											})

								if smd_data:
									smd_df = pd.DataFrame(smd_data)

									# Create a heatmap
									fig = px.imshow(
										smd_df.pivot(index='Variable', columns='Group', values='SMD'),
										color_continuous_scale='RdYlGn_r',
										title='Standardized Mean Differences',
										labels=dict(x="Comparison Group", y="Variable", color="SMD")
									)

									st.plotly_chart(fig, use_container_width=True)

									# Create a categorical assessment
									st.subheader("Balance Assessment")

									balance_assessment = []

									for _, row in smd_df.iterrows():
										smd = row['SMD']

										if smd < 0.1:
											balance = "Good balance"
											color = "green"
										elif smd < 0.2:
											balance = "Acceptable balance"
											color = "orange"
										else:
											balance = "Poor balance"
											color = "red"

										balance_assessment.append({
											'Variable': row['Variable'],
											'Group': row['Group'],
											'SMD': smd,
											'Assessment': balance,
											'Color': color
										})

									if balance_assessment:
										assessment_df = pd.DataFrame(balance_assessment)

										# Display as a styled table
										st.dataframe(assessment_df.style.format({
											'SMD': '{:.3f}'
										}).applymap(lambda _: 'color: green', subset=['Assessment']))

										# Summary
										good_count = len(assessment_df[assessment_df['Assessment'] == "Good balance"])
										acceptable_count = len(assessment_df[assessment_df['Assessment'] == "Acceptable balance"])
										poor_count = len(assessment_df[assessment_df['Assessment'] == "Poor balance"])

										st.markdown("### Balance Summary")

										col1, col2, col3 = st.columns(3)

										col1.metric("Good Balance (SMD < 0.1)", good_count)
										col2.metric("Acceptable Balance (SMD < 0.2)", acceptable_count)
										col3.metric("Poor Balance (SMD ≥ 0.2)", poor_count)

										# Overall assessment
										st.subheader("Overall Balance Assessment")

										if poor_count == 0 and acceptable_count <= len(assessment_df) * 0.2:
											st.success("Overall balance between groups is good.")
										elif poor_count <= len(assessment_df) * 0.2:
											st.warning("Overall balance between groups is acceptable, but some variables show differences.")
										else:
											st.error("Overall balance between groups is poor. Consider adjusting for these variables in your analysis.")
							else:
								st.warning("No numeric variables selected for balance assessment.")
						except Exception as e:
							st.error(f"Error assessing group balance: {e}")
							logger.error(f"Group balance assessment error: {e}", exc_info=True)
			else:
				st.warning("Treatment group data not available.")

	def _render_pipeline_export(self):
		"""Render pipeline results and data export tools."""
		st.header("Pipeline & Export")

		# Check if processed data is available
		if st.session_state.processed_data is not None:
			# Tab navigation for pipeline and export
			tabs = st.tabs(["Pipeline Summary", "Data Export", "Pipeline Report", "Next Steps"])

			with tabs[0]:  # Pipeline Summary
				st.subheader("Data Preparation Pipeline Summary")

				# Get pipeline results
				pipeline_results = st.session_state.pipeline_results

				if pipeline_results:
					# Create a timeline of completed steps
					steps = []

					if 'imputation' in pipeline_results:
						imputation = pipeline_results['imputation']
						steps.append({
							'step': 'Imputation',
							'status': 'Completed',
							'details': f"Method: {imputation.get('method', 'unknown')}, Variables: {len(self.data.columns)}"
						})
					else:
						steps.append({
							'step': 'Imputation',
							'status': 'Not Completed',
							'details': "Missing data not yet imputed"
						})

					if 'variable_screening' in pipeline_results:
						screening = pipeline_results['variable_screening']

						if 'summary' in screening:
							summary = screening['summary']
							steps.append({
								'step': 'Variable Screening',
								'status': 'Completed',
								'details': f"Variables reduced from {summary.get('total_variables', 0)} to {summary.get('recommended_variables', 0)}"
							})
						else:
							steps.append({
								'step': 'Variable Screening',
								'status': 'Completed',
								'details': "Variables screened, but no summary available"
							})
					else:
						steps.append({
							'step': 'Variable Screening',
							'status': 'Not Completed',
							'details': "Variables not yet screened"
						})

					if 'dimensionality_reduction' in pipeline_results:
						dim_reduction = pipeline_results['dimensionality_reduction']
						steps.append({
							'step': 'Dimensionality Reduction',
							'status': 'Completed',
							'details': f"Method: {dim_reduction.get('method', 'unknown')}, Components: {dim_reduction.get('n_components', 0)}"
						})
					else:
						steps.append({
							'step': 'Dimensionality Reduction',
							'status': 'Not Completed',
							'details': "Dimensionality not yet reduced"
						})

					if 'data_quality' in pipeline_results:
						data_quality = pipeline_results['data_quality']
						quality_steps = []

						if 'outliers' in data_quality:
							quality_steps.append("Outlier Detection")

						if 'distributions' in data_quality:
							quality_steps.append("Distribution Analysis")

						if 'transformations' in data_quality:
							quality_steps.append("Transformations")

						if 'standardization' in data_quality:
							quality_steps.append("Standardization")

						if quality_steps:
							steps.append({
								'step': 'Data Quality Enhancement',
								'status': 'Completed',
								'details': f"Completed steps: {', '.join(quality_steps)}"
							})
						else:
							steps.append({
								'step': 'Data Quality Enhancement',
								'status': 'Not Completed',
								'details': "Data quality not yet enhanced"
							})
					else:
						steps.append({
							'step': 'Data Quality Enhancement',
							'status': 'Not Completed',
							'details': "Data quality not yet enhanced"
						})

					# Display the timeline
					for i, step in enumerate(steps):
						col1, col2 = st.columns([1, 3])

						with col1:
							if step['status'] == 'Completed':
								st.success(step['step'])
							else:
								st.warning(step['step'])

						with col2:
							st.markdown(f"**Status:** {step['status']}")
							st.markdown(f"**Details:** {step['details']}")

						if i < len(steps) - 1:
							st.markdown("---")

					# Dataset transformation summary
					st.subheader("Dataset Transformation Summary")

					col1, col2, col3 = st.columns(3)

					with col1:
						# Original dataset
						original_rows = len(self.data)
						original_cols = len(self.data.columns)

						st.metric("Original Dataset", f"{original_rows} × {original_cols}")

					with col2:
						# Current processed dataset
						current_rows = len(st.session_state.processed_data)
						current_cols = len(st.session_state.processed_data.columns)

						st.metric("Current Dataset", f"{current_rows} × {current_cols}")

					with col3:
						# Dimensionality reduction
						dim_reduction = (1 - current_cols / original_cols) * 100

						st.metric("Dimensionality Reduction", f"{dim_reduction:.1f}%")

					# Data preview
					st.subheader("Processed Data Preview")
					st.dataframe(st.session_state.processed_data.head(10))
				else:
					st.warning("No pipeline steps have been completed yet.")

			with tabs[1]:  # Data Export
				st.subheader("Export Processed Data")

				# Export options
				export_format = st.radio(
					"Export format",
					options=["CSV", "Excel"],
					index=0
				)

				# Include metadata
				include_metadata = st.checkbox("Include pipeline metadata", value=True)

				# Export data
				if st.button("Export Data"):
					with st.spinner("Preparing data for export..."):
						try:
							if export_format == "CSV":
								# Export to CSV
								csv = st.session_state.processed_data.to_csv(index=False)
								b64 = base64.b64encode(csv.encode()).decode()

								# Create download link
								href = f'<a href="data:file/csv;base64,{b64}" download="te_koa_processed_data.csv">Download Processed Data (CSV)</a>'
								st.markdown(href, unsafe_allow_html=True)

								# Export metadata if requested
								if include_metadata and pipeline_results:
									# Convert pipeline results to JSON


									# Ensure all numpy arrays and other non-serializable objects are converted
									def convert_for_json(obj):
										if isinstance(obj, (np.ndarray, list)):
											return [convert_for_json(item) for item in obj]
										elif isinstance(obj, dict):
											return {key: convert_for_json(value) for key, value in obj.items()}
										elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
											return float(obj) if np.issubdtype(type(obj), np.floating) else int(obj)
										else:
											return obj

									# Convert pipeline results
									json_results = json.dumps(convert_for_json(pipeline_results), indent=2)
									b64_meta = base64.b64encode(json_results.encode()).decode()

									# Create download link for metadata
									href_meta = f'<a href="data:file/json;base64,{b64_meta}" download="te_koa_pipeline_metadata.json">Download Pipeline Metadata (JSON)</a>'
									st.markdown(href_meta, unsafe_allow_html=True)
							else:  # Excel
								# Export to Excel
								buffer = io.BytesIO()

								with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
									st.session_state.processed_data.to_excel(writer, sheet_name='Processed Data', index=False)

									# Add metadata if requested
									if include_metadata and pipeline_results:
										# Create metadata sheet
										metadata = pd.DataFrame([
											{'Step': 'Imputation', 'Details': str(pipeline_results.get('imputation', 'Not completed'))},
											{'Step': 'Variable Screening', 'Details': str(pipeline_results.get('variable_screening', {}).get('summary', 'Not completed'))},
											{'Step': 'Dimensionality Reduction', 'Details': str(pipeline_results.get('dimensionality_reduction', 'Not completed'))},
											{'Step': 'Data Quality', 'Details': str(pipeline_results.get('data_quality', 'Not completed'))}
										])

										metadata.to_excel(writer, sheet_name='Pipeline Metadata', index=False)

								b64 = base64.b64encode(buffer.getvalue()).decode()

								# Create download link
								href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="te_koa_processed_data.xlsx">Download Processed Data (Excel)</a>'
								st.markdown(href, unsafe_allow_html=True)

							st.success("Data export prepared. Click the link above to download.")
						except Exception as e:
							st.error(f"Error exporting data: {e}")
							logger.error(f"Data export error: {e}", exc_info=True)

			with tabs[2]:  # Pipeline Report
				st.subheader("Data Preparation Pipeline Report")

				# Generate a comprehensive report
				if pipeline_results:
					st.markdown("""
					This report summarizes the data preparation pipeline applied to the TE-KOA dataset,
					including all transformations and their effects on the data.
					""")

					# Report sections
					sections = []

					# 1. Data Overview
					sections.append({
						'title': 'Data Overview',
						'content': f"""
						* Original dataset: {len(self.data)} participants, {len(self.data.columns)} variables
						* Current dataset: {len(st.session_state.processed_data)} participants, {len(st.session_state.processed_data.columns)} variables
						* Dimensionality reduction: {(1 - len(st.session_state.processed_data.columns) / len(self.data.columns)) * 100:.1f}%
						"""
					})

					# 2. Imputation
					if 'imputation' in pipeline_results:
						imputation = pipeline_results['imputation']

						sections.append({
							'title': 'Missing Data & Imputation',
							'content': f"""
							* Imputation method: {imputation.get('method', 'unknown')}
							* KNN neighbors (if applicable): {imputation.get('knn_neighbors', 'N/A')}
							* Original missing values: {imputation.get('original_missing', 'N/A')}
							* Remaining missing values: {imputation.get('remaining_missing', 'N/A')}
							* Excluded variables: {len(imputation.get('cols_excluded', []))}
							"""
						})

					# 3. Variable Screening
					if 'variable_screening' in pipeline_results:
						screening = pipeline_results['variable_screening']

						if 'summary' in screening:
							summary = screening['summary']

							sections.append({
								'title': 'Variable Screening',
								'content': f"""
								* Total variables analyzed: {summary.get('total_variables', 0)}
								* Near-zero variance variables: {summary.get('near_zero_variables', 0)}
								* Highly correlated pairs: {summary.get('highly_correlated_pairs', 0)}
								* High VIF variables: {summary.get('high_vif_variables', 0)}
								* Force-included variables: {summary.get('force_included_variables', 0)}
								* Recommended variables: {summary.get('recommended_variables', 0)}
								"""
							})

					# 4. Dimensionality Reduction
					if 'dimensionality_reduction' in pipeline_results:
						dim_reduction = pipeline_results['dimensionality_reduction']

						optimal_components = pipeline_results.get('optimal_components', {})

						sections.append({
							'title': 'Dimensionality Reduction',
							'content': f"""
							* Method: {dim_reduction.get('method', 'unknown')}
							* Number of components: {dim_reduction.get('n_components', 0)}
							* Variables analyzed: {len(dim_reduction.get('variables', []))}
							* Optimal components: {optimal_components.get('optimal_number', 'N/A')}
							* Variance threshold: {optimal_components.get('variance_threshold', 0) * 100:.0f}%
							* Transformed data shape: {dim_reduction.get('transformed_data_shape', 'N/A')}
							"""
						})

					# 5. Data Quality Enhancement
					if 'data_quality' in pipeline_results:
						data_quality = pipeline_results['data_quality']

						quality_content = ""

						if 'outliers' in data_quality:
							outliers = data_quality['outliers']

							quality_content += f"""
							**Outlier Detection:**
							* Method: {outliers.get('method', 'unknown')}
							* Threshold: {outliers.get('threshold', 'N/A')}
							* Variables analyzed: {len(outliers.get('variables', []))}
							* Total outliers detected: {outliers.get('summary', {}).get('total_outliers_detected', 0)}

							"""

						if 'distributions' in data_quality:
							distributions = data_quality['distributions']

							quality_content += f"""
							**Distribution Analysis:**
							* Variables analyzed: {distributions.get('summary', {}).get('variables_analyzed', 0)}
							* Normal variables: {distributions.get('summary', {}).get('normal_variables', 0)}
							* Skewed variables: {distributions.get('summary', {}).get('skewed_variables', 0)}
							* Highly skewed variables: {distributions.get('summary', {}).get('highly_skewed_variables', 0)}

							"""

						if 'transformations' in data_quality:
							transformations = data_quality['transformations']

							quality_content += f"""
							**Variable Transformations:**
							* Variables analyzed: {transformations.get('summary', {}).get('variables_analyzed', 0)}
							* Variables needing transformation: {transformations.get('summary', {}).get('variables_needing_transformation', 0)}

							"""

							if 'applied' in transformations:
								applied = transformations['applied']

								quality_content += f"""
								**Applied Transformations:**
								* Variables transformed: {applied.get('summary', {}).get('variables_transformed', 0)}
								* Log transformations: {applied.get('summary', {}).get('transformation_counts', {}).get('log', 0)}
								* Square root transformations: {applied.get('summary', {}).get('transformation_counts', {}).get('sqrt', 0)}
								* Square transformations: {applied.get('summary', {}).get('transformation_counts', {}).get('square', 0)}
								* Yeo-Johnson transformations: {applied.get('summary', {}).get('transformation_counts', {}).get('yeo-johnson', 0)}

								"""

						if 'standardization' in data_quality:
							standardization = data_quality['standardization']

							quality_content += f"""
							**Variable Standardization:**
							* Method: {standardization.get('method', 'unknown')}
							* Variables standardized: {len(standardization.get('variables', []))}
							"""

						if quality_content:
							sections.append({
								'title': 'Data Quality Enhancement',
								'content': quality_content
							})

					# Display each section
					for section in sections:
						st.markdown(f"### {section['title']}")
						st.markdown(section['content'])
						st.markdown("---")

					# Download report
					if st.button("Generate Downloadable Report"):
						try:
							# Create a full report in Markdown format
							report = f"# TE-KOA Data Preparation Pipeline Report\n\n"
							report += f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

							for section in sections:
								report += f"## {section['title']}\n\n"
								report += section['content'].strip() + "\n\n"

							# Add a summary table of processed data
							if not st.session_state.processed_data.empty:
								report += "## Processed Data Summary\n\n"

								# Convert DataFrame description to Markdown
								desc = st.session_state.processed_data.describe().to_markdown()
								report += desc + "\n\n"

							# Encode as base64 for download
							b64 = base64.b64encode(report.encode()).decode()

							# Create download link
							href = f'<a href="data:text/markdown;base64,{b64}" download="te_koa_pipeline_report.md">Download Pipeline Report (Markdown)</a>'
							st.markdown(href, unsafe_allow_html=True)

							st.success("Report generated. Click the link above to download.")
						except Exception as e:
							st.error(f"Error generating report: {e}")
							logger.error(f"Report generation error: {e}", exc_info=True)
				else:
					st.warning("No pipeline steps have been completed yet.")

			with tabs[3]:  # Next Steps
				st.subheader("Next Steps")

				st.markdown("""
				### Phase II: Phenotype Discovery (Clustering)

				The next step in the project is to use the prepared dataset for phenotype discovery through clustering:

				1. **Clustering Pipeline**
				   - K-means clustering (fast, simple)
				   - PAM/medoids (robust to outliers)
				   - Gaussian Mixture Model (soft membership)

				2. **Validation Metrics**
				   - Silhouette scores and gap statistic
				   - Bootstrap stability assessment
				   - Minimum cluster size rules

				3. **Phenotype Characterization**
				   - Statistical comparisons between phenotypes
				   - Visualization with radar charts and heatmaps
				   - Clinical interpretation and naming

				### Phase III: Treatment Effect Heterogeneity Analysis

				The final phase will analyze how each phenotype responds to different treatments:

				1. **Linear Mixed Models**
				   - Main effects and interactions with phenotypes

				2. **Causal Machine Learning**
				   - Causal Forest for conditional treatment effects
				   - Bayesian Additive Regression Trees

				3. **Quality Assessment**
				   - Policy value evaluation
				   - Calibration of predictions
				   - Cluster-level treatment effect differences
				""")

				# Add button to continue to Phase II (future development)
				st.button("Proceed to Phase II (Coming Soon)", disabled=True)
		else:
			st.warning("No processed data available. Please complete at least one pipeline step first.")


if __name__ == "__main__":
	# Create and render the dashboard
	dashboard = Dashboard()
	dashboard.render()
