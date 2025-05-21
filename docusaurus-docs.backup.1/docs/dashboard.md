---
sidebar_position: 5
---

# Dashboard Guide

The TE-KOA dashboard provides an interactive interface for exploring and analyzing clinical trial data related to transcranial electrical stimulation for knee osteoarthritis. This guide will walk you through the features and functionality of the dashboard.

## Accessing the Dashboard

You can access the TE-KOA dashboard in two ways:

1. **Live Dashboard**: Visit [https://tekoac.streamlit.app/](https://tekoac.streamlit.app/) to access the publicly hosted version.

2. **Local Installation**: Run the dashboard locally after [installing the project](installation):

   ```bash
   # Using the Python module
   python -m streamlit run tekoa/visualization/app.py

   # Or using the CLI command
   tekoa-dashboard
   ```

## Dashboard Features

The TE-KOA dashboard provides several features for data analysis:

### 1. Data Upload

The dashboard allows you to upload your own Excel files for analysis. The file should contain:

- A data sheet (typically 'Sheet1') with your clinical data
- A 'dictionary' sheet that describes the variables in your dataset

As noted in the project memory, the `DataLoader` class prioritizes loading data from an uploaded file if provided during initialization.

### 2. Data Overview

The overview section provides basic information about your dataset:

- Number of observations and variables
- Data types for each variable
- Summary of missing values

### 3. Data Explorer

The data explorer allows you to:

- View the raw data in a tabular format
- Filter and sort the data
- View basic statistics for each variable

### 4. Data Dictionary

The data dictionary section displays the definitions and metadata for each variable in your dataset, helping you understand what each variable represents.

### 5. Visualizations

The dashboard offers various visualization options:

- Histograms for viewing the distribution of variables
- Box plots for comparing distributions across groups
- Scatter plots for examining relationships between variables
- Correlation heatmaps for identifying relationships among multiple variables

### 6. Missing Data & Imputation

This section helps you:

- Identify variables with missing values
- Visualize patterns of missing data
- Impute missing values using different methods (mean, median, KNN, etc.)

### 7. Variable Screening

The variable screening tools help you identify problematic variables:

- Variables with near-zero variance
- Variables with high collinearity
- Variables with excessive missing values

### 8. Dimensionality Reduction

This section allows you to apply dimensionality reduction techniques like PCA to your data.

### 9. Data Quality

The data quality section helps you assess and enhance the quality of your dataset.

### 10. Treatment Groups

This section provides tools for analyzing treatment groups and their outcomes, including:

- Comparison of baseline characteristics
- Analysis of treatment effects
- Visualization of outcomes by group

### 11. Pipeline & Export

The pipeline section shows you all the processing steps that have been applied to your data, and allows you to export the processed data for further analysis.

## Using the Dashboard

### Uploading Data

1. Navigate to the "Upload Data" section
2. Click the "Upload Excel File" button
3. Select your Excel file containing the clinical data and data dictionary
4. The dashboard will automatically load and process your data

### Analyzing Data

1. Use the sidebar to navigate between different analysis sections
2. Select variables of interest from the dropdown menus
3. Adjust parameters as needed for each analysis
4. View the results in the main panel

### Exporting Results

1. Navigate to the "Pipeline & Export" section
2. Review the processing steps that have been applied
3. Click the "Export Data" button to download the processed data
4. Choose your preferred format (CSV, Excel, etc.)

## Tips for Effective Use

- **Start with Data Overview**: Get a general understanding of your dataset before diving into specific analyses
- **Check Data Quality**: Identify and address data quality issues before proceeding with analysis
- **Explore Visualizations**: Use visualizations to identify patterns and relationships in your data
- **Use Imputation Carefully**: Consider the impact of imputation on your analysis results
- **Document Your Process**: Use the pipeline view to keep track of your analysis steps

## Troubleshooting

If you encounter issues with the dashboard:

- **Data Upload Errors**: Ensure your Excel file has the expected structure (data sheet and dictionary sheet)
- **Visualization Errors**: Check that you've selected appropriate variables for the visualization
- **Performance Issues**: Large datasets may cause performance issues; consider filtering your data
- **Browser Compatibility**: The dashboard works best with modern browsers (Chrome, Firefox, Edge)

For persistent issues, please report them on the [GitHub repository](https://github.com/artinmajdi/tekoa/issues).
