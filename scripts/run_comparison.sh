#!/bin/bash

# Script to run the final comparison using best configurations from BO

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# These should match the settings used/output by the BO runs
RESULTS_DIR="./results"                   # Directory containing 'best_configs'
FINAL_OUTPUT_DIR="${RESULTS_DIR}/final_models" # Where to save comparison results
SEEDS="42 43 44"                          # Space-separated list of seeds
USE_CHECKPOINTS_FLAG="--use_checkpoints"  # Set to "--use_checkpoints" or "" (empty string)
# Specify PE types or leave empty to use all found in best_configs
PE_TYPES_TO_COMPARE="" # Example: "ape rpe rope_mixed"
# --- End Configuration ---

DATE_TAG=$(date +"%Y%m%d_%H%M")
LOG_FILE="${FINAL_OUTPUT_DIR}/comparison_run_${DATE_TAG}.log"

# Create necessary directory
mkdir -p "${FINAL_OUTPUT_DIR}/logs" # Log within the final output dir

# Start logging
exec > >(tee -a ${LOG_FILE}) 2>&1

echo "==========================================================="
echo " Starting Final Comparison"
echo " Started at: $(date)"
echo " Using results from: ${RESULTS_DIR}"
echo " Saving comparison output to: ${FINAL_OUTPUT_DIR}"
echo " Original Seeds Configured: ${SEEDS}" # Show original config
echo " Use Checkpoints: ${USE_CHECKPOINTS_FLAG:-"No"}"
echo " Comparing PE Types: ${PE_TYPES_TO_COMPARE:-"All Default"}"
# --- Add logic to adjust seeds based on checkpoint flag ---
SEEDS_TO_RUN="${SEEDS}" # Default to multiple seeds
if [[ -n "${USE_CHECKPOINTS_FLAG}" ]]; then
    # If using checkpoints, only use the first seed
    SEEDS_TO_RUN=$(echo ${SEEDS} | awk '{print $1}')
    echo " Info: Using checkpoint mode, evaluating only with first seed: ${SEEDS_TO_RUN}"
fi
echo " Seeds being used for this run: ${SEEDS_TO_RUN}" # Show seeds actually used
echo "==========================================================="


# Function to display time elapsed
time_elapsed() {
    local start_time=$1
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(( (elapsed % 3600) / 60 ))
    local seconds=$((elapsed % 60))
    echo "${hours}h ${minutes}m ${seconds}s"
}

START_TIME=$(date +%s)

# Construct arguments for the python script
CMD_ARGS=(
    "--results_dir" "${RESULTS_DIR}"
    "--output_dir" "${FINAL_OUTPUT_DIR}"
    # Use the potentially modified seed list
    "--seeds" ${SEEDS_TO_RUN}
)

# Add optional flags/arguments
if [[ -n "${USE_CHECKPOINTS_FLAG}" ]]; then
    CMD_ARGS+=("${USE_CHECKPOINTS_FLAG}")
fi

if [[ -n "${PE_TYPES_TO_COMPARE}" ]]; then
    # Ensure PE types are passed correctly as separate arguments
    read -a pe_array <<< "${PE_TYPES_TO_COMPARE}" # Read into array
    CMD_ARGS+=("--pe_types" "${pe_array[@]}")
fi

echo "Running command:"
# Use printf for safer printing of command arguments
printf "python scripts/run_final_comparison.py"
printf " %q" "${CMD_ARGS[@]}" # %q quotes arguments safely
echo "" # Newline
echo "-----------------------------------------------------------"

# Execute the script
python scripts/run_final_comparison.py "${CMD_ARGS[@]}"
COMPARE_STATUS=$? # Capture exit status immediately

# Check status (set -e handles errors, but this gives a clear message)
if [ ${COMPARE_STATUS} -ne 0 ]; then
    echo "ERROR: Final comparison script failed with status code: ${COMPARE_STATUS}"
    exit ${COMPARE_STATUS}
else
    echo "Final comparison script completed successfully!"
fi

ELAPSED_TIME=$(time_elapsed ${START_TIME})
echo "-----------------------------------------------------------"
echo "Final comparison execution time: ${ELAPSED_TIME}"
echo "Results saved in: ${FINAL_OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "==========================================================="

# Display summary table if it exists
SUMMARY_TABLE="${FINAL_OUTPUT_DIR}/summary_table.txt"
if [ -f "${SUMMARY_TABLE}" ]; then
    echo "Summary Table:"
    cat "${SUMMARY_TABLE}"
    echo "==========================================================="
fi

exit 0
