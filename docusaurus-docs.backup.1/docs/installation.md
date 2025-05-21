---
sidebar_position: 2
---

# Installation Guide

This guide will walk you through the process of installing and setting up the TE-KOA project on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or higher
- pip (Python package installer)

## Setting up a Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other Python packages:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

## Installing the Package

Once your virtual environment is activated, you can install the TE-KOA package:

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/artinmajdi/tekoa.git
cd tekoa

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Verifying Installation

To verify that the installation was successful, you can run the Streamlit dashboard:

```bash
# Run the Streamlit dashboard
python -m streamlit run tekoa/visualization/app.py
```

Or use the CLI command (if the package is installed with `pip install -e .`):

```bash
tekoa-dashboard
```

You should be able to access the dashboard in your browser, typically at `http://localhost:8501`.

## Next Steps

Once you have successfully installed the TE-KOA project, you can:

- [Configure the application](configuration) based on your needs
- [Run the application using Docker](docker-usage) for containerized deployment
- [Explore the dashboard](dashboard) to analyze your data
