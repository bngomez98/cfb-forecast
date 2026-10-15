#!/bin/bash
set -e

# Work from the repository root, two levels above this script.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "Installing cfb-forecast..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools

# Install the package and dependencies
echo "Installing cfb-forecast..."
pip install -e .

echo "Installation complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To run cfb-forecast, run: cfb-forecast --help"
