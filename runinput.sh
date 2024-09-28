#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <python_file.py> <number_of_runs> <crit_value>"
    exit 1
fi

# Assign arguments to variables
PYTHON_FILE=$1
NUM_RUNS=$2
CRIT_VAL=$3
STEP=0.025

# Check if the provided file exists and is a Python file
if [[ ! -f "$PYTHON_FILE" || "${PYTHON_FILE##*.}" != "py" ]]; then
    echo "Error: '$PYTHON_FILE' is not a valid Python file."
    exit 1
fi

# Initialize conda for the current shell session
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the specified conda environment
conda activate dynamifl

# Run the Python script n times
for ((i=1; i<=NUM_RUNS; i++)); do
    echo "$PYTHON_FILE (Run #$i)"
    python3 "$PYTHON_FILE" "$CRIT_VAL"
done
