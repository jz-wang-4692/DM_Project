"""
Default configuration settings for ViT positional encoding experiments.
This module provides organized configuration parameters and functions to load/save them.
"""

import os
import json
import yaml
from pathlib import Path
import torch

# Default configuration organized by logical sections
DEFAULT_CONFIG = {
    # Model parameters
    "model": {
        "pe_type": "ape",  # Choices: 'ape', 'rope_axial', 'rope_mixed', 'rpe', 'polynomial_rpe'
        "img_size": 32,
        "patch_size": 4,
        "embed_dim": 192,
        "depth": 9,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "drop_rate": 0.1,
        "polynomial_degree": 3,  # Only used for polynomial_rpe
    },
    
    # Training parameters
    "training": {
        "batch_size": 128,
        "epochs": 100,
        "lr": 0.0005,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "mixup_alpha": 0.2,
        "lr_decay_factor": 0.8,
        "label_smoothing": 0.1,
    },
    
    # Data augmentation parameters
    "augmentation": {
        "random_crop_padding": 4,
        "random_erasing_prob": 0.2,
        "color_jitter_brightness": 0.1,
        "color_jitter_contrast": 0.1,
    },
    
    # System parameters
    "system": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4,
        "output_dir": "./output",
        "seed": 42,
    }
}

def save_config(config, filepath):
    """Save configuration to JSON or YAML file"""
    filepath = Path(filepath)
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
    elif filepath.suffix in ['.yaml', '.yml']:
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    return filepath

def load_config(filepath):
    """Load configuration from JSON or YAML file"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            config = json.load(f)
    elif filepath.suffix in ['.yaml', '.yml']:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    return config

def get_config(filepath=None, args=None):
    """
    Get configuration from default, file, and command-line arguments.
    Priority: command-line args > config file > defaults
    
    Args:
        filepath: Path to configuration file (optional)
        args: Parsed command-line arguments (optional)
        
    Returns:
        config: Complete configuration dictionary
    """
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Override with config file if provided
    if filepath:
        file_config = load_config(filepath)
        
        # Deep update of nested dictionaries
        for section, section_config in file_config.items():
            if section in config:
                config[section].update(section_config)
            else:
                config[section] = section_config
    
    # Override with command-line args if provided
    if args:
        args_dict = vars(args)
        
        # Map flat args to nested config structure
        for key, value in args_dict.items():
            if key == 'config_file':  # Skip config_file argument
                continue
                
            # Find the right section for each parameter
            if key in config["model"]:
                config["model"][key] = value
            elif key in config["training"]:
                config["training"][key] = value
            elif key in config["augmentation"]:
                config["augmentation"][key] = value
            elif key in config["system"]:
                config["system"][key] = value
    
    return config

def get_flat_config(config):
    """Convert nested config dict to flat dict for easier access in main.py"""
    flat_config = {}
    for section, section_config in config.items():
        for key, value in section_config.items():
            flat_config[key] = value
    return flat_config

def create_argparser():
    """Create an argument parser with all config options as arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate ViT with different positional encodings on CIFAR-10')
    
    # Add config file argument
    parser.add_argument('--config_file', type=str, help='Path to configuration file (JSON or YAML)')
    
    # Add all config options as arguments
    for section, section_config in DEFAULT_CONFIG.items():
        for key, value in section_config.items():
            # Determine argument type
            arg_type = type(value)
            
            # Handle special cases
            if key == 'pe_type':
                parser.add_argument(f'--{key}', type=str, choices=['ape', 'rope_axial', 'rope_mixed', 'rpe', 'polynomial_rpe'])
            elif arg_type == bool:
                parser.add_argument(f'--{key}', action='store_true' if not value else 'store_false')
            else:
                parser.add_argument(f'--{key}', type=arg_type)
    
    return parser