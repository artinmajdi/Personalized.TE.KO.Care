---
sidebar_position: 4
---

# Docker Usage Guide

This guide provides instructions for deploying the TE-KOA application using Docker containers. Docker provides a consistent environment for running the application regardless of your local setup.

## Prerequisites

Before starting, ensure you have:

- Docker and Docker Compose installed on your system
- Sufficient disk space for Docker images and volumes
- The project code downloaded to your local machine

## Quick Start

To run the TE-KOA application using Docker:

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/artinmajdi/tekoa.git
cd tekoa

# Build and start the Docker container
docker compose up -d
```

After running these commands, the Streamlit interface will be available at http://localhost:8501.

## Docker Compose Configuration

The project includes a `docker-compose.yml` file that defines the services needed to run the application:

```yaml
version: '3'

services:
  tekoa:
    build:
      context: .
      dockerfile: setup_config/docker/Dockerfile
    ports:
      - "8501:8501"
    volumes:
      - ./:/app
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_HEADLESS=true
    command: streamlit run tekoa/visualization/app.py
```

## Environment Variables

You can customize the Docker deployment by setting environment variables:

```bash
# Run with custom environment variables
STREAMLIT_SERVER_PORT=8502 docker compose up -d
```

## Volume Mounts

The Docker configuration includes volume mounts to persist data between container restarts:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./` | `/app` | Application code and data |

This allows you to make changes to the code on your host machine and see them reflected in the container.

## Building a Custom Docker Image

If you need to customize the Docker image:

1. Modify the `setup_config/docker/Dockerfile` to include your customizations
2. Rebuild the Docker image:

```bash
docker compose build --no-cache
```

## Running the Dashboard in Docker

To run the Streamlit dashboard in Docker:

```bash
# Start the container
docker compose up -d

# View logs
docker compose logs -f
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Permission errors | Ensure proper permissions on mounted volumes: `chmod -R 755 ./` |
| Missing environment variables | Verify environment variables are set correctly |
| Port conflicts | Check if port 8501 is already in use: `lsof -i :8501` |
| Container fails to start | Check logs with `docker compose logs` |
| Out of memory errors | Increase Docker resources or optimize application |

### Diagnostic Commands

```bash
# View detailed logs
docker compose logs -f

# Rebuild containers from scratch
docker compose build --no-cache

# Check Docker system resources
docker system df

# Remove unused Docker resources
docker system prune
```

## Advanced Docker Usage

### Running with a Different Dataset

You can mount a different dataset directory when running the container:

```bash
# Mount a custom dataset directory
docker run -p 8501:8501 -v /path/to/custom/dataset:/app/dataset tekoa
```

### Customizing the Container

You can customize the container by overriding the command:

```bash
# Run with a custom command
docker compose run tekoa python -m tekoa.cli
```

This allows you to run different components of the application within the same container.
