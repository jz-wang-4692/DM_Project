"""
Run final comparison between different positional encoding methods using their best configurations.
This script:
1. Loads best configurations for each positional encoding type
2. Trains final models with identical random seeds
3. Evaluates and compares performance across all encoding types
4. Generates comparative visualizations
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
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
from config.default_config import load_config, save_config
from utils.data import get_cifar10_dataloaders
from models.model_factory import create_vit_model
from training.trainer import train_model, evaluate
from utils.visualization import (
    plot_training_history,
    plot_accuracy_comparison,
    plot_convergence_comparison,
    plot_parameter_efficiency,
    plot_configuration_heatmap,
    plot_overfitting_analysis
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run final comparison between different positional encoding methods')
    
    parser.add_argument('--pe_types', type=str, nargs='+', 
                        default=['ape', 'rope_axial', 'rope_mixed', 'rpe', 'polynomial_rpe'],
                        help='Positional encoding types to compare')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds to use for multiple runs')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train for final evaluation')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory with optimization results')
    parser.add_argument('--output_dir', type=str, default='./results/final_models',
                        help='Directory to save final models and results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    return parser.parse_args()

def load_best_config(pe_type, results_dir):
    """Load best configuration for the given positional encoding type"""
    config_path = Path(results_dir) / "best_configs" / pe_type / "config.json"
    
    if not config_path.exists():
        logger.error(f"Best configuration not found for {pe_type}")
        return None
    
    return load_config(config_path)

def get_pe_parameters(model, pe_type):
    """Count parameters specific to the positional encoding method"""
    pe_params = 0
    
    if pe_type == 'ape':
        # APE uses a single positional embedding table (excluding cls token embedding)
        if hasattr(model, 'pos_embed'):
            # Standard format is (1, num_patches + 1, embed_dim) where +1 is for cls token
            # Subtract the cls token parameters
            pe_params = model.pos_embed.numel() - model.pos_embed.shape[-1]
    
    elif pe_type == 'rpe':
        # For standard RPE: Find parameters in RelativePositionalAttention blocks
        for name, module in model.named_modules():
            if 'attn' in name and hasattr(module, 'relative_position_bias_table'):
                pe_params += module.relative_position_bias_table.numel()
    
    elif pe_type == 'polynomial_rpe':
        # For Polynomial RPE: Find polynomial coefficients
        for name, module in model.named_modules():
            if 'attn' in name and hasattr(module, 'poly_coeffs'):
                pe_params += module.poly_coeffs.numel()
    
    elif pe_type in ['rope_axial', 'rope_mixed']:
        # RoPE often doesn't have additional parameters as it uses fixed rotations
        # If there are any learnable parameters in RoPE implementation, they would be counted here
        for name, module in model.named_modules():
            if 'rope' in name.lower() and any(p.requires_grad for p in module.parameters()):
                pe_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    return pe_params

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate(config, pe_type, seed, output_dir, args):
    """Train and evaluate a model with the given configuration and seed"""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create output directory
    run_dir = Path(output_dir) / pe_type / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(config, run_dir / "config.json")
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config['batch_size'],
        num_workers=4,  # Fixed for final evaluation
        aug_params={
            'random_crop_padding': config.get('random_crop_padding', 4),
            'random_erasing_prob': config.get('random_erasing_prob', 0.2),
            'color_jitter_brightness': config.get('color_jitter_brightness', 0.1),
            'color_jitter_contrast': config.get('color_jitter_contrast', 0.1)
        }
    )
    
    # Create model
    logger.info(f"Creating {pe_type} model with seed {seed}")
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
    
    # Count parameters
    total_params = count_parameters(model)
    pe_params = get_pe_parameters(model, pe_type)
    
    logger.info(f"Model has {total_params:,} total trainable parameters")
    logger.info(f"Positional encoding '{pe_type}' uses {pe_params:,} parameters ({pe_params/total_params*100:.2f}% of total)")
    
    # Set up loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 0.0005),
        weight_decay=config.get('weight_decay', 0.05)
    )
    
    # Set up learning rate scheduler with warmup
    def warmup_cosine_schedule(optimizer, warmup_epochs, total_epochs, lr_decay_factor=0.8, min_lr=1e-6):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return float(epoch) / float(max(1, warmup_epochs))
            else:
                # Cosine annealing with slower decay
                progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                # Multiply by factor to slow down the decay rate
                progress = progress * lr_decay_factor
                return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = warmup_cosine_schedule(
        optimizer, 
        warmup_epochs=config.get('warmup_epochs', 5), 
        total_epochs=args.num_epochs,
        lr_decay_factor=config.get('lr_decay_factor', 0.8)
    )
    
    # Train model
    logger.info(f"Training {pe_type} model with seed {seed}")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=args.device,
        mixup_alpha=config.get('mixup_alpha', 0.2)
    )
    
    # Evaluate on test set
    logger.info(f"Evaluating {pe_type} model with seed {seed}")
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
    logger.info(f"Test accuracy: {test_acc:.4f}")
    
    # Save model
    torch.save(model.state_dict(), run_dir / 'model.pth')
    
    # Save training history and results
    results = {
        'pe_type': pe_type,
        'seed': seed,
        'total_parameters': total_params,
        'pe_parameters': pe_params,
        'pe_percentage': pe_params/total_params*100,
        'history': history,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'config': config
    }
    
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Use centralized visualization to plot training history
    plot_training_history(
        history, 
        test_metrics=(test_loss, test_acc),
        output_path=run_dir / 'training_history.png', 
        pe_type=f"{pe_type} (Seed {seed})"
    )
    
    return results

def collect_results(output_dir, pe_types, seeds):
    """Collect all results for analysis"""
    all_results = []
    
    for pe_type in pe_types:
        for seed in seeds:
            result_file = Path(output_dir) / pe_type / f"seed_{seed}" / "results.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    all_results.append(result)
    
    return all_results

def generate_summary_tables(results, output_dir):
    """Generate summary tables from all results"""
    # Summary table with mean and std for each PE type
    summary_data = defaultdict(lambda: defaultdict(list))
    
    # Collect data by PE type
    for result in results:
        pe_type = result['pe_type']
        summary_data[pe_type]['test_accuracy'].append(result['test_accuracy'])
        summary_data[pe_type]['total_parameters'].append(result['total_parameters'])
        summary_data[pe_type]['pe_parameters'].append(result['pe_parameters'])
    
    # Calculate statistics
    summary_table = []
    for pe_type, data in summary_data.items():
        summary_table.append({
            'PE Type': pe_type,
            'Test Accuracy (Mean)': np.mean(data['test_accuracy']),
            'Test Accuracy (Std)': np.std(data['test_accuracy']),
            'Parameters (Total)': np.mean(data['total_parameters']),
            'Parameters (PE)': np.mean(data['pe_parameters']),
            'PE Parameters (%)': np.mean(data['pe_parameters']) / np.mean(data['total_parameters']) * 100
        })
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_table)
    
    # Sort by mean test accuracy
    summary_df = summary_df.sort_values('Test Accuracy (Mean)', ascending=False)
    
    # Save to CSV
    summary_df.to_csv(Path(output_dir) / 'summary_table.csv', index=False)
    
    # Also save as readable text
    with open(Path(output_dir) / 'summary_table.txt', 'w') as f:
        f.write(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    return summary_df

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results to collect
    all_results = []
    
    # Train and evaluate models for each PE type and seed
    for pe_type in args.pe_types:
        logger.info(f"Processing positional encoding type: {pe_type}")
        
        # Load best configuration
        config = load_best_config(pe_type, args.results_dir)
        if not config:
            logger.warning(f"Skipping {pe_type} due to missing configuration")
            continue
        
        # Run with multiple seeds
        for seed in args.seeds:
            logger.info(f"Running {pe_type} with seed {seed}")
            try:
                result = train_and_evaluate(config, pe_type, seed, output_dir, args)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error training {pe_type} with seed {seed}: {e}")
    
    # If we have existing results, load them
    existing_results = collect_results(output_dir, args.pe_types, args.seeds)
    all_results.extend([r for r in existing_results if r not in all_results])
    
    # Generate summary table
    logger.info("Generating summary table")
    summary_df = generate_summary_tables(all_results, output_dir)
    print("\nSummary of Results:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Generate comparison visualizations using centralized visualization functions
    logger.info("Generating comparison visualizations")
    plot_accuracy_comparison(all_results, output_dir / 'accuracy_comparison.png')
    plot_convergence_comparison(all_results, output_dir / 'convergence_comparison.png')
    plot_parameter_efficiency(all_results, output_dir / 'parameter_efficiency.png')
    plot_configuration_heatmap(all_results, output_dir / 'configuration_heatmap.png')
    
    # Add new overfitting analysis plot from centralized visualization
    plot_overfitting_analysis(all_results, output_dir / 'overfitting_analysis.png')
    
    logger.info(f"All results and visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()