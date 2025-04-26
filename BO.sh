#!/bin/bash

# Script to run Bayesian Optimization, (final comparison is currently commented out),
# and archive results.
# Re-running this script will automatically resume the Bayesian Optimization
# studies from where they left off, provided the OUTPUT_DIR is the same.

# Default parameters
N_TRIALS=${1:-50}  # Total number of trials desired for BO (default: 50)
SEEDS=${2:-"42 43 44"}  # Random seeds for final comparison (default: 42 43 44)
OUTPUT_DIR="./results" # IMPORTANT: Keep this consistent to resume studies
DATE_TAG=$(date +"%Y%m%d_%H%M")
# Use a consistent log file name or append timestamp for different runs
# LOG_FILE="${OUTPUT_DIR}/experiment_run.log" # Option 1: Append to one log
LOG_FILE="${OUTPUT_DIR}/logs/experiment_${DATE_TAG}.log" # Option 2: New log per run
ZIP_FILE="vit_pe_results_${DATE_TAG}.zip" # Create a new zip for each run

# Create necessary directories
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/bo_results" # Ensure base BO results dir exists
mkdir -p "${OUTPUT_DIR}/best_configs" # Ensure best configs dir exists

# Start logging
exec > >(tee -a ${LOG_FILE}) 2>&1

echo "==========================================================="
echo "  ViT Positional Encoding Comparison Experiment"
echo "  Run started/resumed at: $(date)"
echo "  Target total trials per encoding: ${N_TRIALS}"
echo "  Final comparison seeds: ${SEEDS}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "==========================================================="

# Check Python environment
echo "Checking Python environment..."
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Please install Python 3.7+"
    exit 1
fi

# Check if required modules are installed
echo "Checking required packages..."
REQUIRED_PACKAGES="torch numpy matplotlib pandas optuna tqdm"
for package in ${REQUIRED_PACKAGES}; do
    python -c "import ${package}" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: Required package '${package}' not found. Please install it with pip."
        exit 1
    fi
done

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

# Run Bayesian Optimization
# Optuna study will resume automatically if database exists in OUTPUT_DIR
echo "Starting/Resuming Bayesian Optimization (Target: ${N_TRIALS} total trials per encoding)..."
START_TIME=$(date +%s)

# Call run_bo.py - it will handle individual PE types and resuming internally.
# Removed --pe_types argument to let run_bo.py use its default (all types)
# or specify the types you want to run/resume here if needed.
python scripts/run_bo.py \
  --n_trials=${N_TRIALS} \
  --output_dir=${OUTPUT_DIR} \
  --pe_types rope_mixed

BO_STATUS=$?

if [ ${BO_STATUS} -ne 0 ]; then
    echo "WARNING: Bayesian Optimization script exited with errors (status code: ${BO_STATUS})"
    echo "Check logs. Some studies might not have completed."
else
    echo "Bayesian Optimization script finished!"
fi

BO_ELAPSED=$(time_elapsed ${START_TIME})
echo "Bayesian Optimization execution time (this run): ${BO_ELAPSED}"

# ------------------------------------------------------------------
# # Run final comparison (commented out)
# echo "Starting final comparison..."
# START_TIME=$(date +%s)
#
# # Convert space-separated seeds to proper format for Python script
# SEED_ARGS=""
# for seed in ${SEEDS}; do
#     SEED_ARGS="${SEED_ARGS} ${seed}"
# done
#
# python scripts/run_final_comparison.py --seeds ${SEED_ARGS} --results_dir=${OUTPUT_DIR} --output_dir=${OUTPUT_DIR}/final_models --use_checkpoints
# COMPARE_STATUS=$?
#
# if [ ${COMPARE_STATUS} -ne 0 ]; then
#     echo "ERROR: Final comparison failed with status code: ${COMPARE_STATUS}"
#     echo "Check logs for details. Results may be incomplete."
# else
#     echo "Final comparison completed successfully!"
# fi
#
# COMPARE_ELAPSED=$(time_elapsed ${START_TIME})
# echo "Final comparison completed in: ${COMPARE_ELAPSED}"
# # End of final comparison block
# ------------------------------------------------------------------

# Archive results from this run
echo "Archiving results..."
# Note: This zips the entire OUTPUT_DIR each time. Consider more specific archiving if needed.
zip -r ${ZIP_FILE} ${OUTPUT_DIR} -x "*.pyc" "**/__pycache__/*" "**/.git/*" "*.db-journal" # Exclude journal files
ZIP_STATUS=$?

if [ ${ZIP_STATUS} -ne 0 ]; then
    echo "ERROR: Failed to create archive. Status code: ${ZIP_STATUS}"
else
    echo "Results archived successfully to: ${ZIP_FILE}"
    echo "Archive size: $(du -h ${ZIP_FILE} | cut -f1)"
fi

# Print summary
echo "==========================================================="
echo "  Experiment Run Summary"
echo "  Completed at: $(date)"
echo "  BO execution time (this run): ${BO_ELAPSED}"
# echo "  Final comparison time: ${COMPARE_ELAPSED}"  # commented out
echo "  Results directory: ${OUTPUT_DIR}"
echo "  Log file: ${LOG_FILE}"
echo "  Archive: ${ZIP_FILE}"
echo "==========================================================="

# Top performing methods (commented out since final comparison is disabled)
# if [ -f "${OUTPUT_DIR}/final_models/summary_table.txt" ]; then
#     echo "Top performing positional encoding methods:"
#     head -n 3 "${OUTPUT_DIR}/final_models/summary_table.txt"
#     echo "==========================================================="
# fi

echo "Script finished!"
exit 0
