#!/bin/bash
# This is a shell script to run a Python script in PyCharm

# Activate virtual environment (if used)
# source venv/bin/activate

# Run the Python script
for file in *.py; do
    echo "Running $file..."
    python3 "$file"
done