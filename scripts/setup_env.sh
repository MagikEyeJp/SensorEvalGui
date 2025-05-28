#!/usr/bin/env bash
# scripts/setup_env.sh
# Create a virtual environment and install project dependencies.

set -e

PYTHON=${PYTHON:-python3}

# Create virtual environment in .venv
$PYTHON -m venv .venv

# Activate the environment for the rest of the script
source .venv/bin/activate

# Upgrade pip itself
pip install --upgrade pip

# Install the project in editable mode with dev extras (includes pytest)
pip install -e '.[dev]'

echo "Environment setup complete. Activate with 'source .venv/bin/activate'"

