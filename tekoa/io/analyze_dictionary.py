"""
Script to analyze the dictionary sheet of the TE-KOA-C dataset.
"""

import pandas as pd
from pathlib import Path
from tekoa import logger

def analyze_dictionary(file_path):
    """
    Analyze the dictionary sheet of the Excel file to better understand variable meanings.

    Args:
        file_path: Path to the Excel file
    """
    print(f"Analyzing dictionary sheet in: {file_path}")

    # Read the dictionary sheet
    df_dict = pd.read_excel(file_path, sheet_name="dictionary")

    # Rename columns for clarity
    df_dict.columns = ['Variable', 'Description', 'Hypothesis']

    # Group variables by categories based on their prefixes
    variable_categories = {}

    for _, row in df_dict.iterrows():
        variable = row['Variable']
        description = row['Description']

        if pd.isna(variable) or variable == "":
            continue

        # Extract prefix from variable name
        parts = variable.split('.')
        if len(parts) > 1:
            prefix = parts[0]
        else:
            prefix = "General"

        if prefix not in variable_categories:
            variable_categories[prefix] = []

        variable_categories[prefix].append({
            'variable': variable,
            'description': description
        })

    # Print organized categories
    print("\n=== VARIABLE CATEGORIES ===")
    for category, variables in variable_categories.items():
        print(f"\n== {category} Variables ({len(variables)}) ==")
        for var in variables[:10]:  # Limit to first 10 per category for brevity
            print(f"  {var['variable']}: {var['description']}")
        if len(variables) > 10:
            print(f"  ... and {len(variables) - 10} more {category} variables")

    # Identify key assessment scales and measurements
    print("\n=== KEY ASSESSMENT SCALES ===")
    key_scales = [
        "WOMAC", "NRS", "HPTH", "HPTO", "PPT", "CPM", "FMI", "PCS"
    ]

    for scale in key_scales:
        scale_vars = [v for v in df_dict['Variable'] if isinstance(v, str) and v.startswith(scale)]
        if scale_vars:
            print(f"\n== {scale} Scale ==")
            # Find description for the scale
            for var in scale_vars:
                if var == scale or var.count('.') == 0:
                    desc_row = df_dict[df_dict['Variable'] == var]
                    if not desc_row.empty:
                        print(f"Description: {desc_row['Description'].values[0]}")
                        break

            # Count measurements at different time points
            time_points = {}
            for var in scale_vars:
                if var.endswith('.0') or var.endswith('.5') or var.endswith('.10'):
                    time_point = var.split('.')[-1]
                    time_points[time_point] = time_points.get(time_point, 0) + 1

            if time_points:
                print("Measurements at time points:")
                for tp, count in time_points.items():
                    print(f"  Day {tp}: {count} measurements")

if __name__ == "__main__":
    # Path to the Excel file
    file_path = Path.cwd() / "dataset" / "TE-KOA-C - sheet_R01_20250410_mostupdated_only RCT data.xlsx"
    analyze_dictionary(file_path)
