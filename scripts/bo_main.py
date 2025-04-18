"""
Bayesian Optimization main script for ViT positional encoding comparison.
Uses Optuna to optimize hyperparameters for each positional encoding type.
"""

import os
import sys
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import optuna
import argparse
import numpy as np
from pathlib import Path
import subprocess
import logging
from datetime import datetime
import pandas as pd
from collections import defaultdict
import torch

# Project imports
from config.search_spaces import SearchSpaces
from config.default_config import save_config
from utils.visualization import (
    plot_optimization_history,
    plot_parameter_importances,
    plot_param_vs_performance,
    plot_slice
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run Bayesian Optimization for ViT positional encodings')
    
    parser.add_argument('--pe_type', type=str, default='ape',
                        choices=['ape', 'rope_axial', 'rope_mixed', 'rpe', 'polynomial_rpe'],
                        help='Type of positional encoding to optimize')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of optimization trials to run')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Name for the Optuna study (defaults to pe_type)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage string (SQLite or MySQL)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout for optimization in seconds')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def objective(trial, pe_type, output_dir):
    """Objective function for Optuna optimization"""
    # Get hyperparameters for this trial
    params = SearchSpaces.get_trial_params(trial, pe_type)
    
    # Apply parameter constraints
    constraint_updates = SearchSpaces.add_parameter_constraints(trial, pe_type)
    for param, value in constraint_updates.items():
        params[param] = value
    
    # Make sure device is properly set
    if 'device' not in params or params['device'] is None:
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create trial directory
    trial_dir = Path(output_dir) / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the output directory in the config to match trial_dir
    params['output_dir'] = str(trial_dir)
    
    # Save trial configuration
    config_path = trial_dir / "config.json"
    save_config(params, config_path)
    
    # Prepare command to run main.py with this configuration
    cmd = [
        sys.executable, 
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"),
        f"--config_file={config_path}",
    ]
    
    # Run the training process
    logger.info(f"Starting trial {trial.number} with parameters: {params}")
    
    try:
        # Run without capturing output to show real-time progress
        process = subprocess.run(cmd, check=True)
        
        # Log files produced
        logger.info(f"Trial {trial.number} completed. Files generated: {[f.name for f in trial_dir.glob('*')]}")
        
        # Parse output to get validation accuracy
        results_file = trial_dir / "results.txt"
        if results_file.exists():
            with open(results_file, "r") as f:
                results_text = f.read()
                # Extract validation accuracy
                val_acc_line = [line for line in results_text.split("\n") 
                                if "Final validation accuracy" in line]
                if val_acc_line:
                    val_acc = float(val_acc_line[0].split(":")[-1].strip())
                    
                    # Report intermediate values to Optuna
                    history_file = trial_dir / "training_history.json"
                    if history_file.exists():
                        with open(history_file, "r") as hf:
                            history = json.load(hf)
                            # Report validation accuracy at different epochs
                            for epoch, acc in enumerate(history["val_acc"]):
                                trial.report(acc, epoch)
                    
                    # Create a trial summary JSON with all relevant metrics
                    trial_summary = {
                        "trial_number": trial.number,
                        "parameters": params,
                        "validation_accuracy": val_acc,
                        "best_epoch": history.get("val_acc", []).index(max(history.get("val_acc", [0]))) if history else None,
                        "training_time": sum(history.get("epoch_times", [])) if history else None,
                        "final_train_acc": history.get("train_acc", [-1])[-1] if history else None,
                        "final_val_acc": history.get("val_acc", [-1])[-1] if history else None,
                        "final_train_loss": history.get("train_loss", [-1])[-1] if history else None,
                        "final_val_loss": history.get("val_loss", [-1])[-1] if history else None
                    }
                    
                    # Save trial summary
                    with open(trial_dir / "trial_summary.json", "w") as f:
                        json.dump(trial_summary, f, indent=4)
                    
                    logger.info(f"Trial {trial.number} achieved validation accuracy: {val_acc:.4f}")
                    return val_acc
        
        # If validation accuracy can't be found, return a poor score
        logger.error(f"Could not find validation accuracy for trial {trial.number}")
        return 0.0
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        return 0.0

def run_optimization(args):
    """Run the optimization process"""
    pe_type = args.pe_type
    n_trials = args.n_trials
    
    # Set up study name
    study_name = args.study_name or f"{pe_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set up storage
    storage = args.storage or f"sqlite:///results/bo_results/{pe_type}/optuna.db"
    
    # Create results directory
    output_dir = Path(args.output_dir) / "bo_results" / pe_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create Optuna study with pruner and sampler for better exploration
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=10, interval_steps=3
    )
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True
    )
    
    # Run optimization
    logger.info(f"Starting optimization for {pe_type} with {n_trials} trials")
    study.optimize(
        lambda trial: objective(trial, pe_type, output_dir),
        n_trials=n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=True
    )
    
    # Save study results
    best_trial = study.best_trial
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best validation accuracy: {best_trial.value:.4f}")
    logger.info(f"Best parameters: {best_trial.params}")
    
    # Save best configuration to best_configs directory
    best_config_dir = Path(args.output_dir) / "best_configs" / pe_type
    best_config_dir.mkdir(parents=True, exist_ok=True)
    
    # Load full configuration of best trial
    best_trial_dir = output_dir / f"trial_{best_trial.number}"
    best_trial_config_path = best_trial_dir / "config.json"
    
    if best_trial_config_path.exists():
        with open(best_trial_config_path, "r") as f:
            best_config = json.load(f)
        
        # Save to best_configs directory
        save_config(best_config, best_config_dir / "config.json")
    
    # Generate optimization visualizations using centralized visualization functions
    plot_optimization_history(study, output_dir / "optimization_history.png")
    plot_parameter_importances(study, output_dir / "parameter_importances.png")
    
    # Generate parameter sensitivity visualizations for key hyperparameters
    key_params = [
        'batch_size', 'lr', 'drop_rate', 'label_smoothing', 'mixup_alpha', 
        'weight_decay', 'embed_dim', 'depth', 'num_heads'
    ]
    
    for param in key_params:
        try:
            plot_param_vs_performance(study, param, output_dir / f"{param}_performance.png")
        except Exception as e:
            logger.warning(f"Could not generate plot for {param}: {e}")
    
    # Generate slice plots for continuous parameters
    continuous_params = ['drop_rate', 'lr', 'weight_decay', 'mixup_alpha', 'label_smoothing']
    for param in continuous_params:
        try:
            plot_slice(study, param, output_dir / f"{param}_slice.png")
        except Exception as e:
            logger.warning(f"Could not generate slice plot for {param}: {e}")
    
    # Generate experiment summary
    generate_experiment_summary(study, output_dir)
    
    # Generate parameter sensitivity analysis
    analyze_parameter_sensitivity(study, output_dir)
    
    return study

def generate_experiment_summary(study, output_dir):
    """Generate comprehensive experiment summary"""
    summary = {
        "study_name": study.study_name,
        "direction": "maximize",
        "n_trials": len(study.trials),
        "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params
        },
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameter_statistics": {},
        "trials": []
    }
    
    # Collect data for all trials
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            summary["trials"].append({
                "number": trial.number,
                "params": trial.params,
                "value": trial.value
            })
    
    # Save summary to file
    with open(output_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    # Create parameter-specific data for post-hoc analysis
    param_data = defaultdict(list)
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            for param_name, param_value in trial.params.items():
                param_data[param_name].append({
                    "value": param_value,
                    "performance": trial.value
                })
    
    # Save parameter-specific data
    for param_name, data in param_data.items():
        with open(output_dir / f"{param_name}_analysis.json", "w") as f:
            json.dump(data, f, indent=4)

def analyze_parameter_sensitivity(study, output_dir):
    """Analyze sensitivity of model performance to different parameters"""
    # Only consider completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) < 2:
        return  # Need at least 2 trials for analysis
    
    # Get all parameter names
    param_names = set()
    for trial in completed_trials:
        param_names.update(trial.params.keys())
    
    # Analyze each parameter
    sensitivity_analysis = {}
    
    for param_name in param_names:
        # Group trials by parameter value
        grouped_data = defaultdict(list)
        
        for trial in completed_trials:
            if param_name in trial.params:
                value = trial.params[param_name]
                performance = trial.value
                # Convert value to string if it's not a simple type
                if not isinstance(value, (int, float, str, bool)):
                    value = str(value)
                grouped_data[value].append(performance)
        
        # Calculate statistics
        param_stats = {
            "unique_values": len(grouped_data),
            "value_performance": {},
            "overall_impact": None
        }
        
        # Calculate mean performance for each value
        for value, performances in grouped_data.items():
            param_stats["value_performance"][value] = {
                "mean": np.mean(performances),
                "std": np.std(performances),
                "min": min(performances),
                "max": max(performances),
                "count": len(performances)
            }
        
        # Calculate overall parameter impact (variation in mean performance)
        if len(grouped_data) > 1:
            mean_performances = [stats["mean"] for stats in param_stats["value_performance"].values()]
            param_stats["overall_impact"] = {
                "range": max(mean_performances) - min(mean_performances),
                "std": np.std(mean_performances)
            }
        
        sensitivity_analysis[param_name] = param_stats
    
    # Sort parameters by impact
    impact_ranking = []
    for param_name, stats in sensitivity_analysis.items():
        if stats["overall_impact"]:
            impact_ranking.append({
                "parameter": param_name,
                "impact_range": stats["overall_impact"]["range"],
                "impact_std": stats["overall_impact"]["std"]
            })
    
    # Sort by impact range in descending order
    impact_ranking.sort(key=lambda x: x["impact_range"], reverse=True)
    
    # Add ranking to sensitivity analysis
    sensitivity_analysis["impact_ranking"] = impact_ranking
    
    # Save to file
    with open(output_dir / "parameter_sensitivity.json", "w") as f:
        json.dump(sensitivity_analysis, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    study = run_optimization(args)
    
    # Print best parameters
    print("\nBest trial parameters:")
    print(json.dumps(study.best_trial.params, indent=2))