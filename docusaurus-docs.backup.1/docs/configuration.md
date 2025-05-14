---
sidebar_position: 3
---

# Configuration Guide

This guide outlines the configuration options for the TE-KOA application, focusing on environment setup and application settings.

## Environment Variables

The TE-KOA application uses environment variables for configuration. These can be set in your operating system or through a `.env` file in the project root.

| Variable | Required | Description |
|----------|----------|-------------|
| `STREAMLIT_SERVER_PORT` | No | Custom port for the Streamlit server (default: 8501) |
| `STREAMLIT_SERVER_HEADLESS` | No | Run Streamlit in headless mode (default: false) |
| `LOG_LEVEL` | No | Logging verbosity (default: INFO) |

## Application Configuration

The application's behavior can be configured through the settings in `te_koa/configurations/settings.py`. This file contains various parameters that control the application's functionality.

### Data Loading Configuration

The `DataLoader` class in `te_koa/io/data_loader.py` can be configured to load data from different sources:

```python
# Load data from an uploaded file (e.g., from Streamlit's file_uploader)
loader = DataLoader(uploaded_file=uploaded_file)

# Load data from a directory
loader = DataLoader(data_dir='path/to/data')
```

As noted in the project memory, the `DataLoader` class has been modified to prioritize loading data from an `uploaded_file` object if provided during initialization.

### Dashboard Configuration

The Streamlit dashboard can be configured through the `streamlit/config.toml` file. You can create this file in the `.streamlit` directory of your project root:

```toml
[server]
port = 8501
headless = false

[theme]
primaryColor = "#E694FF"
backgroundColor = "#00172B"
secondaryBackgroundColor = "#0083B8"
textColor = "#FFFFFF"
font = "sans-serif"
```

## Deployment Configuration

For deployment to platforms like Streamlit Cloud, you may need to configure additional settings:

1. Create a `requirements.txt` file with all necessary dependencies
2. Set up environment variables in the deployment platform
3. Configure any necessary secrets for the application

For more details on deploying the application, see the [Docker Usage Guide](docker-usage).

## Troubleshooting

If you encounter configuration issues:

1. Verify that your environment variables are set correctly
2. Check that the application has the necessary permissions to access data files
3. Ensure that all dependencies are installed correctly
4. Check the application logs for error messages

For persistent issues, you can increase the log level to get more detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
