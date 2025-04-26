"""
Run final comparison between different positional encoding methods using their best configurations.
This script:
1. Loads best configurations for each positional encoding type.
2. Evaluates models by loading checkpoints OR retrains them using the best configs.
3. Compares performance across all encoding types.
4. Generates comparative visualizations.
"""

import os
import sys
from pathlib import Path
import json
import argparse
import numpy as np
import logging
from datetime import datetime
import pandas as pd
from collections import defaultdict
import torch
import torch.nn
import torch.optim

import matplotlib.pyplot as plt

# --- Add Project Root to Path ---
project_root = None
current_path = Path(os.path.abspath('.')).resolve()
print(f"Debug: Initial current_path: {current_path}")
for i in range(3):
    print(f"Debug: Checking directory: {current_path}")
    # Adjust check based on your mandatory project folders
    if (current_path / 'config').is_dir() and \
       (current_path / 'scripts').is_dir() and \
       (current_path / 'utils').is_dir():
        project_root = str(current_path)
        print(f"Debug: Found project root at level {i}: {project_root}")
        break
    if current_path.parent == current_path:
         print("Debug: Reached filesystem root during search.")
         break
    current_path = current_path.parent
if project_root is None:
     project_root = str(Path(os.path.abspath('.')).resolve())
     print(f"Warning: Could not automatically determine project root based on subdirs.")
     print(f"Assuming current directory '{project_root}' is project root or parent.")
     # If running from scripts/, go one level up
     if os.path.basename(project_root) == 'scripts':
          project_root = str(Path(project_root).parent)
          print(f"Adjusted project root to parent: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Info: Added project root '{project_root}' to sys.path")
else:
    print(f"Info: Project root '{project_root}' already in sys.path")
print(f"Debug: Final project root path being used: {project_root}")
print(f"Debug: Current sys.path: {sys.path}")
# --- End Path Addition ---

# --- Project Imports ---
try:
    from config.default_config import load_config, save_config
    from utils.data import get_cifar10_dataloaders
    from models.model_factory import create_vit_model # Ensure this exists
    from training.trainer import train_model, evaluate
    from utils.visualization import (
        plot_training_history,
        plot_accuracy_comparison,
        plot_convergence_comparison,
        plot_parameter_efficiency,
        plot_configuration_heatmap,
        plot_overfitting_analysis,
        # plot_early_stopping_comparison, # Import attempted separately below
        PE_TYPE_COLORS,
        set_plot_style
    )
    # Attempt to import the early stopping plot function separately for robust error handling
    try:
        from utils.visualization import plot_early_stopping_comparison
        can_plot_early_stopping = True
        print("Debug: Successfully imported plot_early_stopping_comparison.")
    except ImportError:
        print("Debug: Failed to import plot_early_stopping_comparison from utils.visualization.")
        can_plot_early_stopping = False

    print("Debug: Successfully imported project modules.")
except ImportError as e:
     print(f"Error importing project modules: {e}")
     print("Please ensure all required files (model_factory.py, etc.) exist in the correct directories.")
     print(f"Project root used for imports: {project_root}")
     sys.exit(1)
# --- End Imports ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run final comparison between different positional encoding methods')
    # Arguments remain the same...
    parser.add_argument('--pe_types', type=str, nargs='+',
                        default=['ape', 'rope_axial', 'rope_mixed', 'rpe', 'polynomial_rpe'],
                        help='Positional encoding types to compare')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds to use for multiple runs')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train for final evaluation (if not using checkpoints)')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing "best_configs" folder')
    parser.add_argument('--output_dir', type=str, default='./results/final_models',
                        help='Directory to save final models and comparison results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--use_checkpoints', action='store_true',
                        help='Use best checkpoints from optimization instead of retraining')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience for final evaluation (if retraining)')
    return parser.parse_args()

def load_best_config(pe_type, results_dir):
    """Load best configuration for the given positional encoding type"""
    config_path = Path(results_dir) / "best_configs" / pe_type / "config.json"
    if not config_path.exists():
        logger.error(f"Best configuration not found for {pe_type} at {config_path}")
        return None
    try:
        config = load_config(config_path)
        if not isinstance(config, dict):
             logger.error(f"Loaded configuration for {pe_type} is not a dictionary.")
             return None
        # Convert nested config to flat dict
        flat_config = {}
        for section, section_config in config.items():
            if isinstance(section_config, dict):
                 flat_config.update(section_config)
            else:
                 flat_config[section] = section_config
        return flat_config
    except Exception as e:
        logger.error(f"Error loading or processing config file {config_path}: {e}")
        return None

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception as e:
        logger.error(f"Error counting total parameters: {e}")
        return 0

def load_checkpoint(model, checkpoint_path, device):
    """Load model state_dict from checkpoint, allowing unsafe loading."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    try:
        # Load checkpoint with weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        state_dict_to_load = None
        if 'model_state_dict' in checkpoint:
             state_dict_to_load = checkpoint['model_state_dict']
             logger.info("Loading 'model_state_dict' from checkpoint.")
        elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
             state_dict_to_load = checkpoint
             logger.info("Checkpoint file seems to be a raw state_dict. Loading directly.")
        else:
             logger.error("Checkpoint file format not recognized.")
             raise ValueError("Unrecognized checkpoint format")

        # Load the state dict
        model_dict = model.state_dict()
        # Filter out unnecessary keys (like mismatched heads if fine-tuning)
        filtered_dict = {k: v for k, v in state_dict_to_load.items() if k in model_dict and v.shape == model_dict[k].shape}
        missing_keys, unexpected_keys = model.load_state_dict(filtered_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        logger.info("Successfully loaded model state from checkpoint.")

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}", exc_info=True)
        raise e

    return model

def train_and_evaluate(config, pe_type, seed, output_dir, args):
    """Train and evaluate a model with the given configuration and seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"Set seed to {seed}")

    run_dir = Path(output_dir) / pe_type / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    try:
        serializable_config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        save_config(serializable_config, run_dir / "config.json")
    except Exception as e:
        logger.error(f"Failed to save config for {pe_type} seed {seed}: {e}")

    logger.info("Loading data...")
    try:
        train_loader, val_loader, test_loader = get_cifar10_dataloaders(
            batch_size=config.get('batch_size', 128),
            num_workers=config.get('num_workers', 4),
            aug_params={k: config.get(k) for k in ['random_crop_padding', 'random_erasing_prob', 'color_jitter_brightness', 'color_jitter_contrast']}
        )
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load data for {pe_type} seed {seed}: {e}", exc_info=True)
        raise e

    logger.info(f"Creating {pe_type} model with seed {seed}")
    try:
        model_kwargs = {}
        if pe_type == 'polynomial_rpe':
            model_kwargs['polynomial_degree'] = config.get('polynomial_degree', 3)
        model = create_vit_model(
            pe_type=pe_type,
            img_size=config.get('img_size', 32),
            patch_size=config.get('patch_size', 4),
            embed_dim=config.get('embed_dim', 192),
            depth=config.get('depth', 9),
            num_heads=config.get('num_heads', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            drop_rate=config.get('drop_rate', 0.1),
            **model_kwargs
        )
        model = model.to(args.device)
        logger.info("Model created successfully.")
    except Exception as e:
        logger.error(f"Failed to create model {pe_type} with seed {seed}: {e}", exc_info=True)
        raise e

    checkpoint_loaded_successfully = False
    if args.use_checkpoints:
        best_checkpoint_path = Path(args.results_dir) / "best_configs" / pe_type / "best_checkpoint.pth"
        if best_checkpoint_path.exists():
            try:
                model = load_checkpoint(model, best_checkpoint_path, args.device)
                logger.info(f"Loaded best checkpoint for {pe_type} from {best_checkpoint_path}")
                checkpoint_loaded_successfully = True
            except Exception as e:
                 logger.error(f"Failed loading checkpoint {best_checkpoint_path}, cannot proceed for this run: {e}")
                 # If checkpoint loading fails when requested, this run is invalid
                 return None
        else:
            logger.warning(f"Checkpoint file specified but not found: {best_checkpoint_path}. Cannot evaluate from checkpoint.")
            # If checkpoint not found, we cannot just evaluate, we must retrain or fail.
            logger.error(f"Cannot proceed with {pe_type} seed {seed} as checkpoint is missing and retraining is disabled.")
            return None # Indicate failure for this run


    evaluate_only = args.use_checkpoints and checkpoint_loaded_successfully
    retrain_model = not evaluate_only

    # Count total parameters (PE parameter counting removed)
    total_params = count_parameters(model)
    logger.info(f"Model has {total_params:,} total trainable parameters")

    history = None
    if retrain_model:
        logger.info("Setting up optimizer and scheduler for retraining...")
        try:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.get('lr', 0.0005),
                weight_decay=config.get('weight_decay', 0.05)
            )
            def warmup_cosine_schedule(optimizer, warmup_epochs, total_epochs, lr_decay_factor=0.8, min_lr=1e-6, base_lr=0.0005):
                def lr_lambda(epoch):
                    if epoch < warmup_epochs: return float(epoch) / float(max(1, warmup_epochs))
                    else:
                        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                        progress = progress * lr_decay_factor
                        cosine_arg = np.pi * min(progress, 1.0)
                        min_lr_factor = min_lr / base_lr if base_lr > 0 else 0
                        return max(min_lr_factor, 0.5 * (1.0 + np.cos(cosine_arg)))
                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            scheduler = warmup_cosine_schedule(
                optimizer,
                warmup_epochs=config.get('warmup_epochs', 5),
                total_epochs=args.num_epochs,
                lr_decay_factor=config.get('lr_decay_factor', 0.8),
                base_lr=config.get('lr', 0.0005)
            )
            logger.info("Optimizer and scheduler set up.")
        except Exception as e: logger.error(f"Failed setup: {e}", exc_info=True); raise e

        logger.info(f"Retraining {pe_type} model with seed {seed} for {args.num_epochs} epochs...")
        try:
            model, history = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                num_epochs=args.num_epochs, device=args.device,
                mixup_alpha=config.get('mixup_alpha', 0.2),
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_delta=config.get('early_stopping_delta', 0.001),
                checkpoint_dir=str(checkpoint_dir)
            )
            logger.info(f"Retraining completed for {pe_type} seed {seed}.")
        except Exception as e: logger.error(f"Retraining failed: {e}", exc_info=True); history = None

    if evaluate_only or history is None:
        logger.info(f"Creating dummy history for {pe_type} seed {seed} (used checkpoint or training failed).")
        history = { 'best_epoch': 0, 'best_val_acc': 0.0, 'early_stopped': False, 'total_training_time': 0.0 } # Minimal dummy

    logger.info(f"Evaluating {pe_type} model with seed {seed} on test set...")
    try:
        eval_criterion = torch.nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(model, test_loader, eval_criterion, args.device)
        logger.info(f"Test evaluation completed. Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    except Exception as e: logger.error(f"Test evaluation failed: {e}", exc_info=True); test_loss, test_acc = -1.0, -1.0

    try:
        torch.save(model.state_dict(), run_dir / 'model_final.pth')
        logger.info(f"Saved final model state to {run_dir / 'model_final.pth'}")
    except Exception as e: logger.error(f"Failed to save final model state: {e}")

    results = {
        'pe_type': pe_type, 'seed': seed, 'total_parameters': total_params,
        # PE parameters removed
        'history': history, 'test_accuracy': test_acc, 'test_loss': test_loss,
        'config': config, 'used_checkpoint': evaluate_only, # Reflects if checkpoint was successfully loaded and used
        'evaluation_timestamp': datetime.now().isoformat()
    }

    try:
        with open(run_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)
        logger.info(f"Saved run results to {run_dir / 'results.json'}")
    except Exception as e: logger.error(f"Failed to save results JSON: {e}")

    if retrain_model and history and history.get('val_acc'): # Only plot if retrained and history exists
        try:
            plot_training_history(history, test_metrics=(test_loss, test_acc),
                                  output_path=run_dir / 'training_history.png',
                                  pe_type=f"{pe_type} (Seed {seed})")
            logger.info(f"Saved training history plot to {run_dir / 'training_history.png'}")
        except Exception as e: logger.error(f"Failed to plot training history: {e}")

    return results

# Removed collect_results function

def generate_summary_tables(results, output_dir, use_checkpoints_flag):
    """Generate summary tables from all collected results"""
    if not results: logger.warning("No results provided to generate summary tables."); return None

    summary_data = defaultdict(lambda: defaultdict(list))

    # --- Define safe_float helper function with proper indentation ---
    def safe_float(value):
        """Helper to safely convert to float, return None on failure"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    # --- End safe_float definition ---

    for result in results:
        pe_type = result.get('pe_type')
        if not pe_type: continue

        test_acc = safe_float(result.get('test_accuracy'))
        total_params = safe_float(result.get('total_parameters'))
        # PE parameters removed

        if test_acc is not None and test_acc >= 0: summary_data[pe_type]['test_accuracy'].append(test_acc)
        else: logger.warning(f"Invalid test_accuracy for {pe_type} seed {result.get('seed')}. Excluding.")
        if total_params is not None: summary_data[pe_type]['total_parameters'].append(total_params)

        # Only collect history stats if retraining occurred
        if not use_checkpoints_flag:
             history = result.get('history')
             if isinstance(history, dict):
                  early_stopped = history.get('early_stopped')
                  best_epoch = safe_float(history.get('best_epoch'))
                  if early_stopped is not None: summary_data[pe_type]['early_stopped'].append(bool(early_stopped))
                  if best_epoch is not None: summary_data[pe_type]['best_epoch'].append(best_epoch)

    summary_table = []
    for pe_type, data in summary_data.items():
        # Ensure lists are not empty before calculating stats
        mean_acc = np.mean(data['test_accuracy']) if data['test_accuracy'] else np.nan
        std_acc = np.std(data['test_accuracy']) if data['test_accuracy'] else np.nan
        mean_total_params = np.mean(data['total_parameters']) if data['total_parameters'] else np.nan
        # PE parameters removed

        entry = {
            'PE Type': pe_type,
            'Test Accuracy (Mean)': mean_acc,
            'Test Accuracy (Std)': std_acc,
            'Parameters (Total)': mean_total_params,
            # PE parameters removed
            'Num Seeds Run': len(data.get('test_accuracy',[]))
        }

        # Add early stopping stats only if retraining occurred
        if not use_checkpoints_flag:
             # Check if data exists before calculating mean
             if data.get('early_stopped'):
                  entry['Early Stopped (%)'] = np.mean(data['early_stopped']) * 100 if data['early_stopped'] else np.nan
             if data.get('best_epoch'):
                  # Add 1 to epoch number for display (since it's 0-indexed)
                  entry['Avg Best Epoch'] = (np.mean(data['best_epoch']) + 1) if data['best_epoch'] else np.nan

        summary_table.append(entry)

    if not summary_table: logger.warning("Summary table empty."); return None
    summary_df = pd.DataFrame(summary_table)
    summary_df = summary_df.sort_values('Test Accuracy (Mean)', ascending=False, na_position='last')

    try:
        summary_csv_path = Path(output_dir) / 'summary_table.csv'
        summary_txt_path = Path(output_dir) / 'summary_table.txt'
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
        logger.info(f"Summary table saved to {summary_csv_path}")
        with open(summary_txt_path, 'w') as f:
            f.write(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"))
        logger.info(f"Summary table saved to {summary_txt_path}")
    except Exception as e: logger.error(f"Failed to save summary tables: {e}")

    return summary_df

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Run Training/Evaluation ---
    # Removed pre-collection and skipping logic - run evaluate/train every time
    all_results = []
    for pe_type in args.pe_types:
        logger.info(f"Processing positional encoding type: {pe_type}")
        config = load_best_config(pe_type, args.results_dir)
        if not config:
            logger.warning(f"Skipping {pe_type} due to missing/invalid configuration")
            continue

        for seed in args.seeds:
            logger.info(f"Running {pe_type} with seed {seed}")
            try:
                # Pass args directly - train_and_evaluate doesn't modify it
                result = train_and_evaluate(config, pe_type, seed, output_dir, args)
                if result: # Check if train_and_evaluate succeeded
                     all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pe_type} with seed {seed}: {e}", exc_info=True)

    if not all_results:
         logger.error("No results generated. Cannot create summaries or plots.")
         return

    # --- Generate Summaries and Plots ---
    logger.info("Generating final summary table...")
    # Pass args.use_checkpoints to control summary table columns
    summary_df = generate_summary_tables(all_results, output_dir, args.use_checkpoints)
    # Removed duplicate print statement for summary table

    logger.info("Generating comparison visualizations...")
    # Wrap plot calls in try-except
    try: plot_accuracy_comparison(all_results, output_dir / 'accuracy_comparison.png')
    except Exception as e: logger.error(f"Failed plot_accuracy_comparison: {e}")

    try: plot_convergence_comparison(all_results, output_dir / 'convergence_comparison.png')
    except Exception as e: logger.error(f"Failed plot_convergence_comparison: {e}")

    try: plot_parameter_efficiency(all_results, output_dir / 'parameter_efficiency.png')
    except Exception as e: logger.error(f"Failed plot_parameter_efficiency: {e}")

    try: plot_configuration_heatmap(all_results, output_dir / 'configuration_heatmap.png')
    except Exception as e: logger.error(f"Failed plot_configuration_heatmap: {e}")

    try: plot_overfitting_analysis(all_results, output_dir / 'overfitting_analysis.png')
    except Exception as e: logger.error(f"Failed plot_overfitting_analysis: {e}")

    # Use the flag determined during import for the early stopping plot
    if can_plot_early_stopping:
        try:
            plot_early_stopping_comparison(all_results, output_dir / 'early_stopping_comparison.png')
        except Exception as e:
            logger.error(f"Failed plot_early_stopping_comparison: {e}")
    else:
         logger.error("Skipping plot_early_stopping_comparison as it could not be imported.")


    logger.info(f"Final comparison processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
