wsl --install
#!/bin/bash

# Navigate to the directory containing the main and branch scripts
cd "$(dirname "$0")"  # Ensures the script runs from the MLHW3 directory

# Optional: load any necessary modules on CADE machines
# Uncomment and adjust as needed for your CADE environment
# module load python/3.x

# Make sure Python scripts are executable
chmod +x *.py

# Run the main script and save output in the output directory
echo "Running main script..."
python3 Main_HW3.py > output/main_output.txt

# Run branch scripts if needed and save their outputs
echo "Running branch script 1..."
python3 StandardPerceptron.py > output/StandardPerceptron.txt

echo "Running branch script 2..."
python3 VotedPerceptron.py > output/VotedPerceptron.txt

echo "Running branch script 3..."
python3 AveragedPerceptron.py > output/AveragedPerceptron.txt
# Print completion message
echo "All scripts have been executed. Output is saved in the output/ directory."