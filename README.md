<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="Personalized.TE.KO.Care.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# PERSONALIZED.TE.KO.CARE

<em>Empowering Personalized Care for Knee Osteoarthritis Patients</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/artinmajdi/Personalized.TE.KO.Care?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/artinmajdi/Personalized.TE.KO.Care?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/artinmajdi/Personalized.TE.KO.Care?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/artinmajdi/Personalized.TE.KO.Care?style=flat&color=0080ff" alt="repo-language-count">
<a href="https://tekoac.streamlit.app/"><img src="https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit Live Dashboard"></a>

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat&logo=JSON&logoColor=white" alt="JSON">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/TOML-9C4121.svg?style=flat&logo=TOML&logoColor=white" alt="TOML">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Pytest-0A9EDC.svg?style=flat&logo=Pytest&logoColor=white" alt="Pytest">
<br>
<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=Docker&logoColor=white" alt="Docker">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white" alt="SciPy">
<img src="https://img.shields.io/badge/Plotly-3F4F75.svg?style=flat&logo=Plotly&logoColor=white" alt="Plotly">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/uv-DE5FE9.svg?style=flat&logo=uv&logoColor=white" alt="uv">
<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=flat&logo=Pydantic&logoColor=white" alt="Pydantic">

</div>
<br>

---

# TEKOA (Transcranial Electrical Stimulation for Knee Osteoarthritis)

**➡️ Live Dashboard: [https://tekoac.streamlit.app/](https://tekoac.streamlit.app/) ⬅️**

A data science and machine learning framework for nursing research focused on analyzing the effects of transcranial electrical stimulation on knee osteoarthritis patients. This project provides tools for data loading, preprocessing, visualization, and analysis of clinical trial data.

## Project Overview

The TE-KOA project is designed to facilitate the analysis of clinical research data related to knee osteoarthritis treatments. It includes functionality for:

- Loading and preprocessing clinical trial data from Excel files (via user upload in the dashboard)
- Handling missing data through various imputation methods
- Analyzing treatment groups and their outcomes
- Visualizing data through an interactive dashboard
- Saving processed data for further analysis

## Project Structure

```
tekoa/
├── dataset/                  # Example dataset directory (Note: dashboard now relies on user uploads)
├── docs/                     # Documentation files
├── scripts/                  # Utility scripts
├── setup_config/             # Configuration files for setup
│   └── docker/               # Docker configuration files
├── tekoa/                   # Main package directory
│   ├── __init__.py           # Package initialization
│   ├── cli.py                # Command-line interface
│   ├── main.py               # Main entry point
│   ├── configurations/       # Configuration settings
│   │   ├── __init__.py
│   │   ├── params.py         # Parameter definitions
│   │   └── settings.py       # Application settings
│   ├── io/                   # Input/output operations
│   │   ├── __init__.py
│   │   ├── analyze_dictionary.py  # Data dictionary analysis
│   │   ├── analyze_excel_file.py  # Excel file analysis
│   │   └── data_loader.py    # Data loading and preprocessing
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   └── watchdog.py       # File monitoring utilities
│   └── visualization/        # Data visualization components
│       ├── __init__.py
│       └── app.py            # Streamlit dashboard application
│       └── app_refactored_claude_components/ # UI and logic components for the dashboard
├── tests/                    # Test directory
├── LICENSE.md                # License information
├── MANIFEST.in               # Package manifest
├── README.md                 # This file
├── pyproject.toml            # Project configuration
├── pytest.ini                # PyTest configuration
├── requirements.txt          # Package dependencies
└── setup.py                  # Setup script
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Setting up a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Installing the Package

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Running the Dashboard Locally

To run the dashboard on your local machine and interact by uploading your own Excel dataset:

```bash
# Ensure your virtual environment is activated
# Navigate to the root directory of the project

# Run the Streamlit dashboard
python -m streamlit run tekoa/visualization/app.py
```

Or use the CLI command (if the package is installed with `pip install -e .`):

```bash
tekoa-dashboard
```

Access the dashboard in your browser, typically at `http://localhost:8501`.

### Using the Data Loader Programmatically

While the dashboard focuses on user uploads, the `DataLoader` can still be used programmatically if you have a dataset path:

```python
from tekoa.io.data_loader import DataLoader

# Example: Assuming you have an Excel file at 'path/to/your_data.xlsx'
# The file should contain 'Sheet1' for data and 'dictionary' for the data dictionary.
try:
    loader = DataLoader(uploaded_file='path/to/your_data.xlsx') # Or use data_dir for directory-based loading if implemented
    data, data_dict = loader.load_data()
    print("Data loaded successfully!")

    # Analyze missing data
    missing_report = loader.get_missing_data_report()
    print("Missing Data Report:")
    print(missing_report[missing_report['Missing Values'] > 0])

    # Impute missing values (example)
    # imputed_data = loader.impute_missing_values(method='knn')
    # print("Data imputed.")

except FileNotFoundError:
    print("Error: The specified data file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Note: The DataLoader primarily expects 'uploaded_file' for Streamlit integration.
# For programmatic use with local files, ensure the 'uploaded_file' parameter
# in DataLoader's __init__ can handle file paths directly, or adapt as needed.
```

## Dashboard Features

The TE-KOA dashboard provides an interactive interface for exploring and analyzing uploaded clinical trial data:

1. **Upload Data**: Users can upload their own Excel files (expected to have a data sheet, typically 'Sheet1', and a 'dictionary' sheet).
2. **Overview**: Basic information about the dataset, including data types and missing values.
3. **Data Explorer**: Tools for exploring the raw data and viewing statistics.
4. **Data Dictionary**: Access to the data dictionary for understanding variable definitions.
5. **Visualizations**: Various visualization options including histograms, box plots, scatter plots, and correlation heatmaps.
6. **Missing Data & Imputation**: Tools for analyzing and imputing missing data using different methods.
7. **Variable Screening**: Identify problematic variables (e.g., near-zero variance, high collinearity).
8. **Dimensionality Reduction**: Apply techniques like PCA.
9. **Data Quality**: Assess and enhance data quality.
10. **Treatment Groups**: Analysis of treatment groups and their outcomes.
11. **Pipeline & Export**: View applied processing steps and export the processed data.

## TODO / Future Work

Based on the project meeting on 4/30/2025, the following tasks are planned:

- **Data Focus**: Ensure all analyses utilize the full dataset (n=208).
- **Phenotype Derivation (Baseline Characteristics)**:
  - Implement logic to derive patient phenotypes based on the 28 baseline characteristics highlighted by Chiyoung (dark blue in dataset/dictionary).
  - These include:
    - 13 sociodemographic, clinical, and baseline pain variables.
    - 13 quantitative sensory testing (QST) measures.
    - 2 psychological measures (CESD for depression, PCS for pain catastrophizing).
  - For phenotyping, use WOMAC subscales (pain, physical function, stiffness) as baseline characteristics, not the WOMAC total score.
- **Variable Grouping Strategies (for Phenotyping/Analysis)**:
  - **Education**: Implement and evaluate grouping strategies for the education variable. Options include:
    - `< high school` vs. `≥ high school`
    - `< two-year college degree` vs. `≥ four-year college degree`
    - The final grouping will be determined based on data proportions and statistical sense.
  - **Marital Status**: Implement and evaluate grouping strategies for marital status. Options include:
    - Grouping `1 = married` and `6 = living with partner` as `married/partnered`.
    - Categorizing others as `unmarried/unpartnered`.
    - The final grouping will be determined based on data proportions and statistical sense.
- **Outcome Variable Refinement**:
  - Confirm and implement the use of NRS (Numeric Rating Scale) and WOMAC (total score and subscales: pain, physical function, stiffness) as the primary outcomes.
  - Note: Outcomes are measured at baseline, Day 5, Day 10, and monthly follow-ups (up to 3 months).
- **Clinical Review Cycle**: Establish a process to notify Chiyoung upon completion of analysis updates for clinical validation of findings through the dashboard.

## Contributing

Contributions to the TE-KOA project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
