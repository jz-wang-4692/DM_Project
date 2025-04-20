import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Project imports
from models.model_factory import create_vit_model
from utils.data import get_cifar10_dataloaders
from training.trainer import train_model, evaluate
from config.default_config import create_argparser, get_config, get_flat_config, save_config

import json
from datetime import datetime

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_pe_parameters(model, pe_type):
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

def save_config_and_results(config, history, model_info, output_dir):
    """Save configuration, training history, and model information to JSON files"""
    # Save configuration
    config_file = output_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save training history
    history_file = output_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Save model information
    model_info_file = output_dir / 'model_info.json'
    with open(model_info_file, 'w') as f:
        json.dump(model_info, f, indent=4)

def main():
    # Parse command-line arguments
    parser = create_argparser()
    args = parser.parse_args()
    
    # Get configuration (combining defaults, config file, and command-line args)
    nested_config = get_config(args.config_file, args)
    
    # Convert to flat dictionary for easier access
    config = get_flat_config(nested_config)
    
    # Save the full configuration at the start of training
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(nested_config, output_dir / 'full_config.json')
    
    # Set device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        aug_params={
            'random_crop_padding': config['random_crop_padding'],
            'random_erasing_prob': config['random_erasing_prob'],
            'color_jitter_brightness': config['color_jitter_brightness'],
            'color_jitter_contrast': config['color_jitter_contrast']
        }
    )
    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images, Test: {len(test_loader.dataset)} images")
    
    # Create model with appropriate positional encoding
    print(f"Creating model with {config['pe_type']} positional encoding...")
    model_kwargs = {}
    if config['pe_type'] == 'polynomial_rpe':
        model_kwargs['polynomial_degree'] = config['polynomial_degree']
    
    model = create_vit_model(
        pe_type=config['pe_type'],
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        drop_rate=config['drop_rate'],
        **model_kwargs
    )
    model = model.to(device)

    # Count parameters
    total_params = count_parameters(model)
    pe_params = count_pe_parameters(model, config['pe_type'])

    print(f"Model has {total_params:,} total trainable parameters")
    print(f"Positional encoding '{config['pe_type']}' uses {pe_params:,} parameters ({pe_params/total_params*100:.2f}% of total)")

    # Create model info dictionary
    model_info = {
        "pe_type": config['pe_type'],
        "total_parameters": total_params,
        "pe_parameters": pe_params,
        "pe_percentage": pe_params/total_params*100,
        "img_size": config['img_size'],
        "patch_size": config['patch_size'],
        "embed_dim": config['embed_dim'],
        "depth": config['depth'],
        "num_heads": config['num_heads'],
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
        
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
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
        warmup_epochs=config['warmup_epochs'], 
        total_epochs=config['epochs'],
        lr_decay_factor=config['lr_decay_factor']
    )
    
    # Train model
    print(f"Training for {config['epochs']} epochs...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['epochs'],
        device=device,
        mixup_alpha=config['mixup_alpha'],
        early_stopping_patience=config.get('early_stopping_patience', 15),
        early_stopping_delta=config.get('early_stopping_delta', 0.001),
        checkpoint_dir=config.get('checkpoint_dir', None)
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Add test results to model_info
    model_info["test_accuracy"] = test_acc
    model_info["test_loss"] = test_loss
    
    # Save model
    torch.save(model.state_dict(), output_dir / 'model.pth')
    
    # Save training history
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{config["pe_type"]} Accuracy')
    
    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{config["pe_type"]} Loss')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png')
    
    # Save results summary
    with open(output_dir / 'results.txt', 'w') as f:
        f.write(f"Model: Vision Transformer with {config['pe_type']}\n")
        f.write(f"Embedding dimension: {config['embed_dim']}\n")
        f.write(f"Depth: {config['depth']}\n")
        f.write(f"Number of heads: {config['num_heads']}\n")
        f.write(f"Training epochs: {config['epochs']}\n")
        f.write(f"Final training accuracy: {history['train_acc'][-1]:.4f}\n")
        f.write(f"Final validation accuracy: {history['val_acc'][-1]:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
    
    # Save full configuration and results
    save_config_and_results(nested_config, history, model_info, output_dir)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()