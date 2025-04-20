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
import matplotlib.pyplot as plt


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
    
    # Set checkpoint directory for saving best model checkpoints
    checkpoint_dir = trial_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    params['checkpoint_dir'] = str(checkpoint_dir)
    
    # Save trial configuration
    config_path = trial_dir / "config.json"
    save_config(params, config_path)
    
    # Check for global best validation accuracy from previous trials
    best_config_dir = Path(args.output_dir) / "best_configs" / pe_type
    best_config_dir.mkdir(parents=True, exist_ok=True)
    
    global_best_val_file = best_config_dir / "best_val_acc.txt"
    global_best_val_acc = 0.0
    
    if global_best_val_file.exists():
        try:
            with open(global_best_val_file, "r") as f:
                global_best_val_acc = float(f.read().strip())
                logger.info(f"Current global best validation accuracy: {global_best_val_acc:.4f}")
        except (ValueError, IOError) as e:
            logger.warning(f"Could not read global best validation accuracy: {e}")
    
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
                    history = None
                    history_file = trial_dir / "training_history.json"
                    if history_file.exists():
                        try:
                            with open(history_file, "r") as hf:
                                history = json.load(hf)
                                
                                # Report validation accuracy at different epochs
                                for epoch, acc in enumerate(history.get("val_acc", [])):
                                    trial.report(acc, epoch)
                                    
                                    # Check for pruning - allows Optuna to terminate trials early
                                    if trial.should_prune():
                                        logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                                        raise optuna.exceptions.TrialPruned()
                        except Exception as e:
                            logger.warning(f"Error loading history file for trial {trial.number}: {e}")
                    
                    # MORE ROBUST: Create a trial summary JSON with all relevant metrics
                    try:
                        trial_summary = {
                            "trial_number": trial.number,
                            "parameters": params,
                            "validation_accuracy": val_acc,
                        }
                        
                        # Add history-related metrics with careful error handling
                        if history and isinstance(history, dict):
                            try:
                                trial_summary["best_epoch"] = history.get("best_epoch", 0)
                            except Exception as e:
                                logger.warning(f"Error getting best_epoch for trial {trial.number}: {e}")
                                trial_summary["best_epoch"] = None
                                
                            try:
                                trial_summary["early_stopped"] = history.get("early_stopped", False)
                            except Exception as e:
                                logger.warning(f"Error getting early_stopped for trial {trial.number}: {e}")
                                trial_summary["early_stopped"] = None
                            
                            try:
                                epoch_times = history.get("epoch_times", [])
                                trial_summary["training_time"] = sum(epoch_times) if epoch_times else None
                            except Exception as e:
                                logger.warning(f"Error calculating training_time for trial {trial.number}: {e}")
                                trial_summary["training_time"] = None
                            
                            try:
                                train_acc = history.get("train_acc", [])
                                trial_summary["final_train_acc"] = train_acc[-1] if train_acc else None
                            except Exception as e:
                                logger.warning(f"Error getting final_train_acc for trial {trial.number}: {e}")
                                trial_summary["final_train_acc"] = None
                            
                            try:
                                val_acc_list = history.get("val_acc", [])
                                trial_summary["final_val_acc"] = val_acc_list[-1] if val_acc_list else None
                            except Exception as e:
                                logger.warning(f"Error getting final_val_acc for trial {trial.number}: {e}")
                                trial_summary["final_val_acc"] = None
                            
                            try:
                                trial_summary["best_val_acc"] = history.get("best_val_acc", val_acc)
                            except Exception as e:
                                logger.warning(f"Error getting best_val_acc for trial {trial.number}: {e}")
                                trial_summary["best_val_acc"] = val_acc
                            
                            try:
                                train_loss = history.get("train_loss", [])
                                trial_summary["final_train_loss"] = train_loss[-1] if train_loss else None
                            except Exception as e:
                                logger.warning(f"Error getting final_train_loss for trial {trial.number}: {e}")
                                trial_summary["final_train_loss"] = None
                            
                            try:
                                val_loss = history.get("val_loss", [])
                                trial_summary["final_val_loss"] = val_loss[-1] if val_loss else None
                            except Exception as e:
                                logger.warning(f"Error getting final_val_loss for trial {trial.number}: {e}")
                                trial_summary["final_val_loss"] = None
                            
                            try:
                                train_val_gap = history.get("train_val_gap", [])
                                trial_summary["avg_train_val_gap"] = float(np.mean(train_val_gap)) if train_val_gap else None
                            except Exception as e:
                                logger.warning(f"Error calculating avg_train_val_gap for trial {trial.number}: {e}")
                                trial_summary["avg_train_val_gap"] = None
                        else:
                            logger.warning(f"History object is missing or invalid for trial {trial.number}")
                            # Set all history-dependent fields to None
                            for field in ["best_epoch", "early_stopped", "training_time", "final_train_acc", 
                                        "final_val_acc", "best_val_acc", "final_train_loss", "final_val_loss", 
                                        "avg_train_val_gap"]:
                                trial_summary[field] = None
                                
                            # Default best_val_acc to current val_acc if history object doesn't exist
                            trial_summary["best_val_acc"] = val_acc
                        
                        # Save trial summary
                        try:
                            with open(trial_dir / "trial_summary.json", "w") as f:
                                json.dump(trial_summary, f, indent=4)
                            logger.info(f"Trial summary saved for trial {trial.number}")
                        except Exception as e:
                            logger.error(f"Failed to save trial summary for trial {trial.number}: {e}")
                            
                    except Exception as e:
                        logger.error(f"Error creating trial summary for trial {trial.number}: {e}")
                        # Fall back to a minimal trial summary
                        trial_summary = {
                            "trial_number": trial.number,
                            "parameters": params,
                            "validation_accuracy": val_acc,
                            "best_val_acc": val_acc
                        }
                    
                    # Get best_val_acc for this trial safely
                    best_val_acc = trial_summary.get("best_val_acc", val_acc)
                    logger.info(f"Trial {trial.number} achieved best validation accuracy: {best_val_acc:.4f}")
                    
                    if "best_epoch" in trial_summary and trial_summary["best_epoch"] is not None:
                        logger.info(f"Best epoch: {trial_summary['best_epoch']+1}")
                    
                    # MORE ROBUST: Check if this trial is the new global best
                    if best_val_acc > global_best_val_acc:
                        logger.info(f"New global best found! Previous: {global_best_val_acc:.4f}, New: {best_val_acc:.4f}")
                        global_best_val_acc = best_val_acc
                        
                        try:
                            # Update the global best validation accuracy record
                            with open(global_best_val_file, "w") as f:
                                f.write(f"{global_best_val_acc}")
                            logger.info(f"Updated global best validation accuracy record")
                        except Exception as e:
                            logger.error(f"Failed to update global best validation record: {e}")
                        
                        try:
                            # Copy the best configuration to the centralized location
                            save_config(params, best_config_dir / "config.json")
                            logger.info(f"Saved best configuration to global location")
                        except Exception as e:
                            logger.error(f"Failed to save best configuration: {e}")
                        
                        # Copy the best checkpoint
                        best_checkpoint_path = checkpoint_dir / "best_checkpoint.pth"
                        try:
                            if best_checkpoint_path.exists():
                                import shutil
                                target_path = best_config_dir / "best_checkpoint.pth"
                                shutil.copy(best_checkpoint_path, target_path)
                                logger.info(f"Copied best checkpoint to {target_path}")
                                
                                # Create safe meta-information with defaults for missing values
                                meta_info = {
                                    "trial_number": trial.number,
                                    "best_val_acc": best_val_acc,
                                    "best_epoch": trial_summary.get("best_epoch", 0),
                                    "update_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "early_stopped": trial_summary.get("early_stopped", False)
                                }
                                
                                # Save additional meta-information about this best model
                                with open(best_config_dir / "best_model_info.json", "w") as f:
                                    json.dump(meta_info, f, indent=4)
                                logger.info(f"Saved best model meta-information")
                            else:
                                logger.warning(f"Best checkpoint file not found at {best_checkpoint_path}")
                        except Exception as e:
                            logger.error(f"Failed to copy best checkpoint or save meta-information: {e}")
                    
                    # Check if early stopping was triggered
                    if trial_summary.get("early_stopped", False):
                        if history and "val_acc" in history:
                            logger.info(f"Trial {trial.number} was early stopped after epoch {len(history['val_acc'])}")
                        else:
                            logger.info(f"Trial {trial.number} was early stopped")
                        
                        if "avg_train_val_gap" in trial_summary and trial_summary["avg_train_val_gap"] is not None:
                            logger.info(f"Average train-val gap: {trial_summary['avg_train_val_gap']:.4f}")
                    
                    return best_val_acc
        
        # If validation accuracy can't be found, return a poor score
        logger.error(f"Could not find validation accuracy for trial {trial.number}")
        return 0.0
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        return 0.0
    except optuna.exceptions.TrialPruned:
        logger.info(f"Trial {trial.number} was pruned by Optuna")
        raise  # Re-raise the pruning exception for Optuna to handle

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
        
        # Also copy best checkpoint if available
        best_checkpoint_path = best_trial_dir / "checkpoints" / "best_checkpoint.pth"
        if best_checkpoint_path.exists():
            import shutil
            shutil.copy(best_checkpoint_path, best_config_dir / "best_checkpoint.pth")
    
    # Generate optimization visualizations using centralized visualization functions
    plot_optimization_history(study, output_dir / "optimization_history.png")
    plot_parameter_importances(study, output_dir / "parameter_importances.png")
    
    # Generate parameter sensitivity visualizations for key hyperparameters
    key_params = [
        'batch_size', 'lr', 'drop_rate', 'label_smoothing', 'mixup_alpha', 
        'weight_decay', 'embed_dim', 'depth', 'num_heads', 'early_stopping_patience'
    ]
    
    for param in key_params:
        try:
            plot_param_vs_performance(study, param, output_dir / f"{param}_performance.png")
        except Exception as e:
            logger.warning(f"Could not generate plot for {param}: {e}")
    
    # Generate slice plots for continuous parameters
    continuous_params = ['drop_rate', 'lr', 'weight_decay', 'mixup_alpha', 'label_smoothing', 'early_stopping_delta']
    for param in continuous_params:
        try:
            plot_slice(study, param, output_dir / f"{param}_slice.png")
        except Exception as e:
            logger.warning(f"Could not generate slice plot for {param}: {e}")
    
    # Generate experiment summary
    generate_experiment_summary(study, output_dir)
    
    # Generate parameter sensitivity analysis
    analyze_parameter_sensitivity(study, output_dir)
    
    # Analyze early stopping and overfitting patterns
    analyze_early_stopping_patterns(study, output_dir)
    
    return study

def generate_experiment_summary(study, output_dir):
    """Generate comprehensive experiment summary"""
    summary = {
        "study_name": study.study_name,
        "direction": "maximize",
        "n_trials": len(study.trials),
        "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
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
            # Get additional metrics from trial summary if available
            trial_dir = Path(output_dir) / f"trial_{trial.number}"
            trial_summary_path = trial_dir / "trial_summary.json"
            
            additional_metrics = {}
            if trial_summary_path.exists():
                with open(trial_summary_path, "r") as f:
                    trial_summary = json.load(f)
                    additional_metrics = {
                        "best_epoch": trial_summary.get("best_epoch"),
                        "early_stopped": trial_summary.get("early_stopped", False),
                        "avg_train_val_gap": trial_summary.get("avg_train_val_gap")
                    }
            
            summary["trials"].append({
                "number": trial.number,
                "params": trial.params,
                "value": trial.value,
                **additional_metrics
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
                    "performance": trial.value,
                    "trial_number": trial.number
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

def analyze_early_stopping_patterns(study, output_dir):
    """Analyze patterns in early stopping and overfitting"""
    # Collect data from trial summaries
    early_stopping_data = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_dir = Path(output_dir) / f"trial_{trial.number}"
            trial_summary_path = trial_dir / "trial_summary.json"
            
            if trial_summary_path.exists():
                with open(trial_summary_path, "r") as f:
                    summary = json.load(f)
                    
                    early_stopping_data.append({
                        "trial_number": trial.number,
                        "early_stopped": summary.get("early_stopped", False),
                        "best_epoch": summary.get("best_epoch", 0),
                        "best_val_acc": summary.get("best_val_acc", 0),
                        "avg_train_val_gap": summary.get("avg_train_val_gap", 0),
                        "patience": trial.params.get("early_stopping_patience", 10) if hasattr(trial, "params") else None,
                        "delta": trial.params.get("early_stopping_delta", 0.001) if hasattr(trial, "params") else None,
                    })
    
    # Skip analysis if not enough data
    if len(early_stopping_data) < 2:
        return
    
    # Create early stopping analysis
    early_stopping_analysis = {
        "n_early_stopped": sum(1 for d in early_stopping_data if d["early_stopped"]),
        "average_best_epoch": np.mean([d["best_epoch"] for d in early_stopping_data]),
        "early_stopping_rate": sum(1 for d in early_stopping_data if d["early_stopped"]) / len(early_stopping_data),
        "average_train_val_gap": np.mean([d["avg_train_val_gap"] for d in early_stopping_data if d["avg_train_val_gap"] is not None]),
    }
    
    # Group by patience values to see effect
    patience_groups = defaultdict(list)
    for d in early_stopping_data:
        if d["patience"] is not None:
            patience_groups[d["patience"]].append(d["best_val_acc"])
    
    patience_analysis = {}
    for patience, accs in patience_groups.items():
        patience_analysis[patience] = {
            "mean_accuracy": np.mean(accs),
            "count": len(accs)
        }
    
    early_stopping_analysis["patience_analysis"] = patience_analysis
    
    # Group by train-val gap to analyze overfitting correlation
    df = pd.DataFrame(early_stopping_data)
    if not df.empty and "avg_train_val_gap" in df.columns and "best_val_acc" in df.columns:
        corr = df["avg_train_val_gap"].corr(df["best_val_acc"])
        early_stopping_analysis["gap_acc_correlation"] = corr
    
    # Save analysis
    with open(output_dir / "early_stopping_analysis.json", "w") as f:
        json.dump(early_stopping_analysis, f, indent=4)
    
    # Create visualizations for early stopping patterns
    # Plot best epoch histogram
    plt.figure(figsize=(10, 6))
    plt.hist([d["best_epoch"] for d in early_stopping_data], bins=10, alpha=0.7)
    plt.xlabel("Best Epoch")
    plt.ylabel("Count")
    plt.title("Distribution of Best Epochs Across Trials")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_dir / "best_epoch_distribution.png")
    plt.close()
    
    # Plot train-val gap vs performance
    if not df.empty and "avg_train_val_gap" in df.columns and "best_val_acc" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df["avg_train_val_gap"], df["best_val_acc"], alpha=0.7)
        plt.xlabel("Average Train-Val Accuracy Gap")
        plt.ylabel("Best Validation Accuracy")
        plt.title("Overfitting vs. Performance")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(output_dir / "overfitting_vs_performance.png")
        plt.close()

if __name__ == "__main__":
    args = parse_args()
    study = run_optimization(args)
    
    # Print best parameters
    print("\nBest trial parameters:")
    print(json.dumps(study.best_trial.params, indent=2))