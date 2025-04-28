"""
Script to load a completed Optuna study and run the post-optimization analysis
and visualization steps originally found in bo_main.py.
"""

import os
import sys
from pathlib import Path # Import Path
import json
import optuna
import argparse
import numpy as np
import logging
from datetime import datetime
import pandas as pd
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import shutil
import re # Import regex for fallback source code parsing

# --- Dynamically Add Project Root to Python Path ---
# Try to find the project root directory based on known subdirectories
current_dir = Path(os.path.abspath('.'))
project_root = None
# Look for known directories ('config', 'scripts', 'results') by going up the tree
search_dir = current_dir
print(f"Debug: Starting search for project root from: {search_dir}")
for i in range(3): # Search up to 3 levels up
    print(f"Debug: Checking directory: {search_dir}")
    # Adjust check based on your mandatory project folders
    if (search_dir / 'config').is_dir() and \
       (search_dir / 'scripts').is_dir() and \
       (search_dir / 'utils').is_dir():
        project_root = search_dir
        print(f"Debug: Found project root at level {i}: {project_root}")
        break
    if search_dir.parent == search_dir: # Reached filesystem root
         print("Debug: Reached filesystem root during search.")
         break
    search_dir = search_dir.parent

project_root_str = None
if project_root:
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str) # Insert at beginning for import priority
        print(f"Info: Dynamically added project root '{project_root_str}' to sys.path")
    else:
        print(f"Info: Project root '{project_root_str}' already in sys.path")
else:
    # Fallback: Assume the script is run from the project root
    project_root_str = str(current_dir)
    print(f"Warning: Could not automatically determine project root based on subdirs.")
    print(f"Assuming current directory '{project_root_str}' is project root.")
    if project_root_str not in sys.path:
         sys.path.insert(0, project_root_str)
         print(f"Info: Added current directory '{project_root_str}' to sys.path")

print(f"Debug: Final project root path being used: {project_root_str}")
print(f"Debug: Current sys.path: {sys.path}")
# --- End Project Root Path Addition ---


# Project imports - Check if these paths are correct
# These should now work if the project root was added correctly
try:
    from config.default_config import load_config, save_config # Added save_config
    # Import the analysis/plotting functions directly if they are complex
    # or copy their definitions here if simpler.
    # Assuming they are defined within the original bo_main.py for now:
    # IMPORTANT: Ensure 'bo_main.py' exists in the 'scripts' directory
    # and contains the required analysis functions.
    # We will define update_global_best locally instead of importing it.
    from scripts.bo_main import ( # Import from the original bo_main script name
        generate_experiment_summary,
        analyze_parameter_sensitivity,
        analyze_early_stopping_patterns
        # Removed update_global_best from import
    )
    from utils.visualization import (
        plot_optimization_history,
        plot_parameter_importances,
        plot_param_vs_performance,
        plot_slice
    )
    print("Debug: Successfully imported project modules.")
except ImportError as e:
     print(f"Error importing project modules or analysis functions: {e}")
     # Provide more specific guidance based on the error
     if "scripts.bo_main" in str(e):
          print("Ensure 'bo_main.py' exists in the 'scripts' directory and contains the required functions:")
          print("  generate_experiment_summary, analyze_parameter_sensitivity, analyze_early_stopping_patterns") # Removed update_global_best
     elif "utils.visualization" in str(e):
          print("Ensure 'visualization.py' exists in the 'utils' directory.")
     elif "config.default_config" in str(e):
           print("Ensure 'default_config.py' exists in the 'config' directory.")
     else:
          print("Check that the required source files exist in their respective directories (scripts/, utils/, config/).")
     print("Also check that the project root was correctly identified and added to sys.path.")
     sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during imports: {type(e).__name__} - {e}")
     sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Define update_global_best function locally ---
# This logic was likely part of the objective function or a helper in bo_main.py
def update_global_best(best_val_acc, best_params, best_trial_obj, best_config_dir, trial_checkpoint_dir):
    """Updates the global best records."""
    global_best_val_file = best_config_dir / "best_val_acc.txt"
    best_config_file = best_config_dir / "config.json"
    best_model_info_file = best_config_dir / "best_model_info.json"
    global_best_checkpoint_file = best_config_dir / "best_checkpoint.pth"
    trial_best_checkpoint_file = trial_checkpoint_dir / "best_checkpoint.pth"

    logger.info(f"Updating global best records in {best_config_dir} for trial {best_trial_obj.number}")

    # Update best validation accuracy record
    try:
        with open(global_best_val_file, "w") as f:
            f.write(f"{best_val_acc}")
        logger.info(f"  - Updated global best validation accuracy record to {best_val_acc:.6f}")
    except Exception as e:
        logger.error(f"  - Failed to update global best validation record: {e}")

    # Save the best configuration (ensure best_params is the nested dict)
    try:
        save_config(best_params, best_config_file) # save_config expects nested dict
        logger.info(f"  - Saved best configuration globally to {best_config_file}")
    except Exception as e:
        logger.error(f"  - Failed to save best configuration globally: {e}")

    # Copy the best checkpoint from the trial directory
    try:
        if trial_best_checkpoint_file.exists():
            shutil.copy(trial_best_checkpoint_file, global_best_checkpoint_file)
            logger.info(f"  - Copied best checkpoint from trial {best_trial_obj.number} to {global_best_checkpoint_file}")

            # Save meta-information about this best model
            # Try getting history summary from the trial object if available
            # Note: User attributes might not be populated when loading study this way
            # It's safer to try and load the trial_summary.json if needed
            trial_history_summary = {}
            trial_summary_path = trial_checkpoint_dir.parent / "trial_summary.json"
            if trial_summary_path.exists():
                 try:
                      with open(trial_summary_path, 'r') as tsf:
                           trial_summary_content = json.load(tsf)
                           trial_history_summary = trial_summary_content.get("history_summary", {})
                 except Exception as ts_e:
                      logger.warning(f"    - Could not load trial_summary.json for meta info: {ts_e}")


            meta_info = {
                "trial_number": best_trial_obj.number,
                "best_val_acc": best_val_acc,
                "best_epoch": trial_history_summary.get("best_epoch", "N/A"), # Get from history if possible
                "update_timestamp": datetime.now().isoformat(),
                "early_stopped": trial_history_summary.get("early_stopped", "N/A"),
                "parameters": best_params # Include params for reference
            }
            with open(best_model_info_file, "w") as f:
                # Use default=str for non-serializable types like Path objects if params contain them
                json.dump(meta_info, f, indent=4, default=str)
            logger.info(f"  - Saved best model meta-information to {best_model_info_file}")
        else:
            logger.warning(f"  - Best checkpoint file not found in trial directory: {trial_best_checkpoint_file}")
            if global_best_checkpoint_file.exists():
                 logger.warning(f"  - Keeping previous global best checkpoint file.")

    except Exception as e:
        logger.error(f"  - Failed to copy best checkpoint or save meta-information: {e}", exc_info=True)
# --- End of update_global_best definition ---


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze completed Bayesian Optimization study')

    parser.add_argument('--pe_type', type=str, required=True,
                        choices=['ape', 'rope_axial', 'rope_mixed', 'rpe', 'polynomial_rpe'],
                        help='Type of positional encoding study to analyze')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Base directory where results are stored')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Specific name of the Optuna study (optional, defaults to pe_type_optimization)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage string (optional, defaults to SQLite path)')

    return parser.parse_args()

def run_analysis(args):
    """Loads the study and runs the analysis functions."""
    pe_type = args.pe_type
    base_output_dir = Path(args.output_dir).resolve() # Use absolute path

    # Determine study name
    # Use provided name or default convention (ensure consistency with bo_main.py)
    study_name = args.study_name or f"{pe_type}_optimization"
    # Try finding the timestamped version if the simple one fails
    # TODO: Add logic here if your bo_main.py uses timestamped names by default
    possible_study_names = [study_name]


    # Determine storage path
    storage_dir = base_output_dir / "bo_results" / pe_type
    storage_path = storage_dir / "optuna.db"
    storage = args.storage or f"sqlite:///{storage_path}"

    logger.info(f"Attempting to load study for PE Type: {pe_type}")
    logger.info(f"Storage Path: {storage}")
    logger.info(f"Base Output Directory: {base_output_dir}")

    if not storage_path.exists():
         logger.error(f"Database file not found at expected location: {storage_path}")
         logger.error("Please ensure the --output_dir argument points to the correct base results directory.")
         return

    study = None
    loaded_study_name = None

    # Try loading the study using possible names
    for name_attempt in possible_study_names:
         try:
             study = optuna.load_study(study_name=name_attempt, storage=storage)
             loaded_study_name = name_attempt
             logger.info(f"Successfully loaded study '{loaded_study_name}' from {storage}")
             break # Stop trying names once loaded
         except KeyError: # Study name not found in this DB
             logger.warning(f"Study name '{name_attempt}' not found in the database at {storage_path}.")
             continue # Try next name if available
         except Exception as e:
             logger.error(f"Failed to load study '{name_attempt}' from {storage}: {e}")
             # Continue to try other names just in case

    if not study:
        logger.error(f"Could not load study for {pe_type}. Tried name(s): {possible_study_names}. Please check study name and storage path.")
        return

    # --- Run Post-Optimization Analysis ---
    logger.info(f"Starting analysis for study '{loaded_study_name}'...")

    # Define the directory where analysis outputs should be saved
    analysis_output_dir = storage_dir # Save analysis plots/jsons in the PE type's BO results dir

    best_trial = None
    try:
        # Use get_trials to filter for COMPLETED state before finding best
        completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if not completed_trials:
             logger.warning(f"No completed trials found for study {loaded_study_name}. Cannot determine best trial.")
        else:
             # Find best trial among completed ones based on study direction
             if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                  # Ensure value is not None before comparison
                  valid_trials = [t for t in completed_trials if t.value is not None]
                  if not valid_trials:
                       logger.warning("No completed trials with valid objective values found.")
                  else:
                       best_trial = max(valid_trials, key=lambda t: t.value)
             elif study.direction == optuna.study.StudyDirection.MINIMIZE:
                  valid_trials = [t for t in completed_trials if t.value is not None]
                  if not valid_trials:
                       logger.warning("No completed trials with valid objective values found.")
                  else:
                       best_trial = min(valid_trials, key=lambda t: t.value)
             else: # Handle potential multi-objective cases if needed, though unlikely here
                  logger.warning(f"Unsupported study direction: {study.direction}. Cannot determine single best trial.")


             if best_trial and best_trial.value is not None:
                  logger.info(f"Best completed trial determined: {best_trial.number} with value {best_trial.value:.6f}")

                  # Ensure the global best config reflects the final best trial found
                  best_config_dir = base_output_dir / "best_configs" / pe_type
                  best_trial_dir = analysis_output_dir / f"trial_{best_trial.number}"
                  trial_checkpoint_dir = best_trial_dir / "checkpoints"

                  # Create best_config_dir if it doesn't exist
                  best_config_dir.mkdir(parents=True, exist_ok=True)

                  if best_trial_dir.exists():
                       best_trial_config_path = best_trial_dir / "config.json"
                       if best_trial_config_path.exists():
                            # Load the config associated with the best trial number
                            best_config_params = load_config(best_trial_config_path)
                            # Call update_global_best (defined locally now)
                            update_global_best(best_trial.value, best_config_params, best_trial, best_config_dir, trial_checkpoint_dir)
                            logger.info(f"Verified/Updated global best configuration files based on best trial {best_trial.number}.")
                       else:
                            logger.warning(f"Config file not found for best trial {best_trial.number} at {best_trial_config_path}. Cannot update global best config.")
                  else:
                       logger.warning(f"Directory not found for best trial {best_trial.number} at {best_trial_dir}. Cannot update global best config.")
             elif best_trial and best_trial.value is None:
                  logger.warning(f"Best trial candidate (Trial {best_trial.number}) has a None value. Cannot proceed with best trial logic.")
             # else: best_trial is None because no valid completed trials were found


    except Exception as e:
         logger.error(f"Error accessing best trial info or updating global best: {e}", exc_info=True) # Log traceback


    # Generate optimization visualizations
    logger.info("Generating optimization plots...")
    try:
        plot_optimization_history(study, analysis_output_dir / "optimization_history.png")
    except Exception as e:
        logger.warning(f"Could not generate optimization_history plot: {e}")

    try:
        plot_parameter_importances(study, analysis_output_dir / "parameter_importances.png")
    except Exception as e:
        logger.warning(f"Could not generate parameter_importances plot: {e}")

    key_params = [
        'batch_size', 'lr', 'drop_rate', 'label_smoothing', 'mixup_alpha',
        'weight_decay', 'embed_dim', 'depth', 'num_heads', 'early_stopping_patience'
    ]
    logger.info("Generating parameter vs performance plots...")
    for param in key_params:
        try:
            plot_param_vs_performance(study, param, analysis_output_dir / f"{param}_performance.png")
        except Exception as e:
            logger.warning(f"Could not generate plot for {param}: {e}")

    continuous_params = ['drop_rate', 'lr', 'weight_decay', 'mixup_alpha', 'label_smoothing', 'early_stopping_delta']
    logger.info("Generating slice plots...")
    for param in continuous_params:
        try:
            plot_slice(study, param, analysis_output_dir / f"{param}_slice.png")
        except Exception as e:
            logger.warning(f"Could not generate slice plot for {param}: {e}")

    # Generate experiment summary
    logger.info("Generating experiment summary...")
    try:
        generate_experiment_summary(study, analysis_output_dir)
    except Exception as e:
        logger.error(f"Failed to generate experiment summary: {e}", exc_info=True)

    # Generate parameter sensitivity analysis
    logger.info("Generating parameter sensitivity analysis...")
    try:
        analyze_parameter_sensitivity(study, analysis_output_dir)
    except Exception as e:
        logger.error(f"Failed to generate parameter sensitivity analysis: {e}", exc_info=True)

    # Analyze early stopping and overfitting patterns
    logger.info("Analyzing early stopping patterns...")
    try:
        analyze_early_stopping_patterns(study, analysis_output_dir)
    except Exception as e:
        logger.error(f"Failed to analyze early stopping patterns: {e}", exc_info=True)

    logger.info(f"Analysis complete for {pe_type}. Results saved in {analysis_output_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
