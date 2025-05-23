---
sidebar_position: 6
---

# API Reference

This page documents the key classes and functions in the TE-KOA project that you can use programmatically.

## Data Loading

### DataLoader

The `DataLoader` class is responsible for loading and preprocessing clinical trial data.

```python
from tekoa.io.data_loader import DataLoader
```

#### Constructor

```python
DataLoader(uploaded_file=None, data_dir=None)
```

**Parameters:**

- `uploaded_file`: A file-like object (e.g., from Streamlit's file_uploader) containing the Excel data.
- `data_dir`: Directory path containing the Excel data file (used if uploaded_file is not provided).

As noted in the project memory, the `DataLoader` class prioritizes loading data from an `uploaded_file` object if provided during initialization.

#### Methods

```python
load_data()
```

Loads data from the specified source (uploaded file or data directory).

**Returns:**

- `data`: Pandas DataFrame containing the clinical data.
- `data_dict`: Pandas DataFrame containing the data dictionary.

```python
get_missing_data_report()
```

Generates a report of missing values in the dataset.

**Returns:**

- Pandas DataFrame with missing data statistics for each variable.

```python
impute_missing_values(method='mean', **kwargs)
```

Imputes missing values in the dataset.

**Parameters:**

- `method`: Imputation method to use ('mean', 'median', 'knn', etc.).
- `**kwargs`: Additional parameters for the imputation method.

**Returns:**

- Pandas DataFrame with imputed values.

## Data Analysis

### analyze_dictionary

Functions for analyzing the data dictionary.

```python
from tekoa.io.analyze_dictionary import get_dictionary_summary
```

```python
get_dictionary_summary(data_dict)
```

Generates a summary of the data dictionary.

**Parameters:**

- `data_dict`: Pandas DataFrame containing the data dictionary.

**Returns:**

- Dictionary with summary statistics about the data dictionary.

### analyze_excel_file

Functions for analyzing Excel files.

```python
from tekoa.io.analyze_excel_file import get_excel_structure
```

```python
get_excel_structure(file_path)
```

Analyzes the structure of an Excel file.

**Parameters:**

- `file_path`: Path to the Excel file.

**Returns:**

- Dictionary with information about the Excel file structure.

## Visualization

### app

The main Streamlit application module.

```python
from tekoa.visualization.app import run_app
```

```python
run_app()
```

Runs the Streamlit dashboard application.

## Command Line Interface

### cli

Command-line interface for the TE-KOA project.

```python
from tekoa.cli import main
```

```python
main()
```

Runs the command-line interface.

## Configuration

### settings

Application settings and configuration.

```python
from tekoa.configuration.settings import get_settings
```

```python
get_settings()
```

Returns the application settings.

**Returns:**

- Dictionary containing application settings.

### params

Parameter definitions for the application.

```python
from tekoa.configuration.params import get_params
```

```python
get_params()
```

Returns the application parameters.

**Returns:**

- Dictionary containing application parameters.

## Example Usage

Here's an example of how to use the TE-KOA API programmatically:

```python
from tekoa.io.data_loader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# Load data from a file path
file_path = 'path/to/your_data.xlsx'
with open(file_path, 'rb') as f:
    loader = DataLoader(uploaded_file=f)
    data, data_dict = loader.load_data()

# Analyze missing data
missing_report = loader.get_missing_data_report()
print("Missing Data Report:")
print(missing_report[missing_report['Missing Values'] > 0])

# Impute missing values
imputed_data = loader.impute_missing_values(method='knn', n_neighbors=5)

# Visualize a variable
plt.figure(figsize=(10, 6))
plt.hist(imputed_data['WOMAC_pain'], bins=20)
plt.title('Distribution of WOMAC Pain Scores')
plt.xlabel('WOMAC Pain Score')
plt.ylabel('Frequency')
plt.show()
```

For more detailed examples and usage patterns, refer to the [Dashboard Guide](dashboard) and the code documentation in the [GitHub repository](https://github.com/artinmajdi/tekoa).
