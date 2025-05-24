"""
Script to analyze the TE-KOA-C dataset and print its structure.
"""

import pandas as pd
from pathlib import Path
from tekoa import logger

def analyze_excel_file(file_path):
    """
    Analyze an Excel file and print information about its structure.

    Args:
        file_path: Path to the Excel file
    """
    print(f"Analyzing Excel file: {file_path}")

    # Read all sheets from the Excel file
    excel_file = pd.ExcelFile(file_path)
    print(f"Sheet names: {excel_file.sheet_names}")

    # Process the data sheet
    if "Sheet1" in excel_file.sheet_names:
        df_data = pd.read_excel(excel_file, sheet_name="Sheet1")
        print("\nData Sheet (Sheet1) Overview:")
        print(f"Number of rows: {len(df_data)}")
        print(f"Number of columns: {len(df_data.columns)}")
        print("\nColumn names:")
        for i, col in enumerate(df_data.columns):
            print(f"{i+1}. {col}")

        # Print sample data (first 5 rows)
        print("\nSample data (first 5 rows):")
        print(df_data.head())

        # Print data types
        print("\nData types:")
        print(df_data.dtypes)

        # Basic statistics
        print("\nBasic numeric statistics:")
        numeric_cols = df_data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(df_data[numeric_cols].describe())

    # Process the dictionary sheet
    if "dictionary" in excel_file.sheet_names:
        df_dict = pd.read_excel(excel_file, sheet_name="dictionary")
        print("\nDictionary Sheet Overview:")
        print(f"Number of rows: {len(df_dict)}")
        print(f"Number of columns: {len(df_dict.columns)}")
        print("\nColumn names:")
        for i, col in enumerate(df_dict.columns):
            print(f"{i+1}. {col}")

        # Print the dictionary content
        print("\nDictionary content:")
        for i, row in df_dict.iterrows():
            print(f"Row {i+1}:")
            for col in df_dict.columns:
                print(f"  {col}: {row[col]}")
            print("---")
            if i >= 9:  # Limit to first 10 rows for brevity
                print("... (showing first 10 rows only)")
                break

if __name__ == "__main__":
    # Path to the Excel file
    file_path = Path.cwd() / "dataset" / "TE-KOA-C - sheet_R01_20250410_mostupdated_only RCT data.xlsx"
    analyze_excel_file(file_path)
