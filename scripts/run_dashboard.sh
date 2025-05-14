#!/bin/bash

# Define colors for better user experience
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m' # Added RED for errors
NC='\033[0m' # No Color

# Get the root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${BLUE}=== Project Template - Dashboard Runner ===${NC}"
echo "This script helps you run the project dashboard."
echo ""

# Check if .env file exists
if [ ! -f "$ROOT_DIR/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Some features may not work properly.${NC}"
    echo -e "Consider running './scripts/setup_env.sh' first to set up your environment."
    echo ""
fi

# Prompt user to choose between Docker and directly running the app
echo "Please choose how you would like to run the dashboard:"
echo -e "${BLUE}1)${NC} Docker (recommended for production use)"
echo -e "${BLUE}2)${NC} Local env (recommended for development)"
echo ""

read -p "Enter your choice (1/2): " choice

case $choice in
    1)
        echo -e "${GREEN}Starting Docker container...${NC}"
        "$SCRIPT_DIR/run_docker.sh" start
        ;;
    2)
        echo -e "${GREEN}Starting application directly...${NC}"

        # Check if virtual environment exists and activate if found
        if [ -d "$ROOT_DIR/.venv" ]; then
            echo -e "${BLUE}Using virtual environment...${NC}"
            source "$ROOT_DIR/.venv/bin/activate" || source "$ROOT_DIR/.venv/Scripts/activate"
        else
            # Check if we're in a conda environment
            if command -v conda &> /dev/null && [ -n "$CONDA_DEFAULT_ENV" ]; then
                echo -e "${BLUE}Using conda environment: $CONDA_DEFAULT_ENV${NC}"
            else
                echo -e "${YELLOW}Warning: No virtual environment detected, using system Python.${NC}"
                echo -e "Consider running './scripts/install.sh' first to set up your environment."
            fi
        fi

        # Determine the application entry point
        APP_PATH="" # Initialize APP_PATH
        VIS_DIR_RELATIVE="te_koa/visualization" # Relative path for messages
        VIS_DIR_ABSOLUTE="$ROOT_DIR/$VIS_DIR_RELATIVE"
        
        echo -e "${BLUE}Looking for application files (app*.py) in '$VIS_DIR_RELATIVE'...${NC}"
        
        app_candidates_full_paths=()
        if [ -d "$VIS_DIR_ABSOLUTE" ]; then # Check if directory exists before find
            # Use find and process substitution to populate the array safely
            while IFS= read -r line; do
                app_candidates_full_paths+=("$line")
            done < <(find "$VIS_DIR_ABSOLUTE" -maxdepth 1 -name "app*.py" -type f 2>/dev/null)
        else
            echo -e "${YELLOW}Directory '$VIS_DIR_ABSOLUTE' not found.${NC}"
        fi

        num_app_candidates=${#app_candidates_full_paths[@]}

        if [ "$num_app_candidates" -eq 1 ]; then
            # Single candidate found
            APP_PATH="${app_candidates_full_paths[0]#"$ROOT_DIR/"}" # Get path relative to ROOT_DIR
            echo -e "${GREEN}Found application: $APP_PATH${NC}"
        elif [ "$num_app_candidates" -gt 1 ]; then
            # Multiple candidates found, prompt user
            echo -e "${BLUE}Multiple application entry points found in '$VIS_DIR_RELATIVE'. Please choose one:${NC}"
            
            # Prepare options for select (relative paths)
            options_relative_paths=()
            for full_path in "${app_candidates_full_paths[@]}"; do
                options_relative_paths+=("${full_path#"$ROOT_DIR/"}")
            done

            PS3="Enter your choice (number): " # Set prompt for select
            select selected_app_relative_path in "${options_relative_paths[@]}"; do
                if [[ -n "$selected_app_relative_path" ]]; then
                    APP_PATH="$selected_app_relative_path"
                    echo -e "${GREEN}You selected: $APP_PATH${NC}"
                    break
                else
                    echo -e "${YELLOW}Invalid selection. Please enter a number from the list.${NC}"
                fi
            done
            PS3="" # Reset PS3 prompt to default
        else
            # This 'else' means 0 candidates were found in VIS_DIR_RELATIVE or VIS_DIR_RELATIVE didn't exist.
            # Message for 0 candidates only if the directory actually existed
            if [ -d "$VIS_DIR_ABSOLUTE" ]; then
                 echo -e "${YELLOW}No application files (app*.py) found in '$VIS_DIR_RELATIVE'.${NC}"
            fi
            # APP_PATH remains empty, will trigger fallback logic below
        fi

        # Fallback logic: If APP_PATH is still empty (no suitable app in visualization dir or dir not found)
        if [ -z "$APP_PATH" ]; then
            echo -e "${BLUE}Attempting to find application in other common locations...${NC}"
            if [ -f "$ROOT_DIR/te_koa/dashboard/app.py" ]; then
                APP_PATH="te_koa/dashboard/app.py"
                echo -e "${GREEN}Found application: $APP_PATH${NC}"
            elif [ -f "$ROOT_DIR/te_koa/app.py" ]; then
                APP_PATH="te_koa/app.py"
                echo -e "${GREEN}Found application: $APP_PATH${NC}"
            elif [ -f "$ROOT_DIR/te_koa/main.py" ]; then
                APP_PATH="te_koa/main.py"
                echo -e "${GREEN}Found application: $APP_PATH${NC}"
            else
                # No app found by any automatic method, prompt user
                echo -e "${YELLOW}No application entry point found automatically by any method.${NC}"
                echo -e "Please specify the path relative to the project root ($ROOT_DIR)."
                # Loop to ensure user provides a valid file path
                while true; do
                    read -p "Enter the relative path to your application file: " custom_path
                    if [ -z "$custom_path" ]; then
                        echo -e "${YELLOW}Path cannot be empty. Please try again.${NC}"
                    elif [ -f "$ROOT_DIR/$custom_path" ]; then
                        APP_PATH="$custom_path"
                        echo -e "${GREEN}Using user-specified application: $APP_PATH${NC}"
                        break
                    else
                        echo -e "${RED}Error: File not found at '$ROOT_DIR/$custom_path'. Please check the path and try again.${NC}"
                    fi
                done
            fi
        fi
        
        # Final check: Ensure APP_PATH is set (should be handled by user prompt loop if all else fails)
        if [ -z "$APP_PATH" ]; then
            echo -e "${RED}Critical Error: Application entry point could not be determined even after user prompt. Exiting.${NC}"
            exit 1
        fi

        # Check if streamlit is installed
        if command -v streamlit &> /dev/null; then
            # Run with Streamlit if it's installed
            cd "$ROOT_DIR"
            echo -e "${GREEN}Starting Streamlit application: $APP_PATH${NC}"
            streamlit run "$APP_PATH" --server.runOnSave=true
        else
            # Otherwise run as a Python script
            cd "$ROOT_DIR"
            echo -e "${GREEN}Starting Python application: $APP_PATH${NC}"
            python "$APP_PATH"
        fi
        ;;
    *)
        echo -e "${YELLOW}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac
