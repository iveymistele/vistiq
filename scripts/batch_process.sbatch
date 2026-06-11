#!/usr/bin/env bash
#SBATCH --job-name=segment
#SBATCH --output=batch-%A-%a.log
#SBATCH --error=batch-%A-%a.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -e

# Check if file list is provided
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <py_script> <file_list> [output_dir] [extra python args...]"
    echo "  py_script:  The python script to execute"
    echo "  file_list:  Text file with one image file path per line (optionally with output dir as second column)"
    echo "              When run as job array, each task processes the file at line #ARRAY_TASK_ID"
    echo "  output_dir: Optional output directory (default: current directory)"
    echo "              If file list has second column with output dir, it will be used with priority:"
    echo "              - If second column is absolute path, it overrides CLI output_dir"
    echo "              - If second column is relative path, it is joined with CLI output_dir (or PWD)"
    echo "  extra args: Passed through to the python script after -i and -o, e.g. --n-samples 10"
    exit 1
fi

py_script="$1"
file_list="$2"
shift 2
if [[ $# -gt 0 && "$1" != --* ]]; then
    cli_output_dir="$1"
    shift
else
    cli_output_dir="$PWD"
fi
extra_args=("$@")

# Check if file list exists and is a file (not a directory)
if [[ ! -e "$file_list" ]]; then
    echo "Error: File list does not exist: $file_list"
    exit 1
elif [[ -d "$file_list" ]]; then
    echo "Error: First argument must be a file (file list), not a directory: $file_list"
    echo "       Did you mean to pass the file list as the first argument?"
    echo "       Usage: $0 <file_list> [output_dir]"
    exit 1
elif [[ ! -f "$file_list" ]]; then
    echo "Error: File list is not a regular file: $file_list"
    exit 1
fi

# Get array task ID (support both SLURM and PBS)
if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    TASK_ID=$SLURM_ARRAY_TASK_ID
else
    echo "Warning: No array task ID found. Processing first file in list."
    TASK_ID=1
fi

# Extract the file path for this task ID (1-indexed)
input_file=$(sed -n "${TASK_ID}p" "$file_list")

# Check if we got a valid file
if [[ -z "$input_file" ]]; then
    echo "Error: No file found at line $TASK_ID in $file_list"
    exit 1
fi

# Parse quoted paths from file list (handles paths with spaces)
# Format: 'path1' 'path2' or path1 path2
# Use awk with single quote as field separator to extract content between quotes
# If line starts with quote, use quoted parsing; otherwise use regular field splitting
if [[ "$input_file" =~ ^\' ]]; then
    # Line starts with quote - extract between quotes
    # Fields when split on ': empty, path1, separator, path2, empty
    input_file_path=$(echo "$input_file" | awk -F"'" '{print $2}')
    file_output_dir=$(echo "$input_file" | awk -F"'" '{print $4}')
else
    # No quotes - use regular field splitting
    input_file_path=$(echo "$input_file" | awk '{print $1}')
    file_output_dir=$(echo "$input_file" | awk '{print $2}')
fi

# Use the parsed input file path
input_file="$input_file_path"

# Determine output directory
# Priority: 1) Third CLI argument (if not a flag), 2) Second column in file list, 3) PWD
if [[ -n "$file_output_dir" ]]; then
    # File column exists
    if [[ "$file_output_dir" == /* ]]; then
        # Absolute path in file - use it directly
        output_dir="$file_output_dir"
    else
        # Relative path in file - join with cli_output_dir (which defaults to $PWD)
        output_dir="$cli_output_dir/$file_output_dir"
    fi
else
    # No file column - use cli_output_dir (which defaults to $PWD)
    output_dir="$cli_output_dir"
fi

# Input file path already extracted above, no need to extract again

# Check if the input file exists
if [[ ! -f "$input_file" ]]; then
    echo "Error: Input file does not exist: $input_file"
    exit 1
fi

echo "Task ID: $TASK_ID"
echo "Processing file: $input_file"
echo "Output directory: $output_dir"
if [[ ${#extra_args[@]} -gt 0 ]]; then
    echo "Extra python args: ${extra_args[*]}"
fi

module purge
module load miniforge
#source activate /standard/vol191/siegristlab/software/vistiq-env # replace with your own environment path
source activate ~/vistiq-env-gpu # replace with your own environment path

# Run Python script on the selected file
python "$py_script" -i "$input_file" -o "$output_dir" "${extra_args[@]}"