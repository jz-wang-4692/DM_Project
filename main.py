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
        json.dump(vars(config), f, indent=4)
    
    # Save training history
    history_file = output_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Save model information
    model_info_file = output_dir / 'model_info.json'
    with open(model_info_file, 'w') as f:
        json.dump(model_info, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate ViT with different positional encodings on CIFAR-10')
    
    # Model parameters
    parser.add_argument('--pe_type', type=str, default='ape',
                        choices=['ape', 'rope_axial', 'rope_mixed', 'rpe', 'polynomial_rpe'],
                        help='Type of positional encoding')
    parser.add_argument('--img_size', type=int, default=32, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=192, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=9, help='Transformer depth')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP hidden dim ratio')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--polynomial_degree', type=int, default=3, help='Degree for polynomial RPE')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')

    # Add MixUp parameter
    parser.add_argument('--mixup_alpha', type=float, default=0.2, 
                    help='Alpha parameter for mixup augmentation (0 to disable)')
    
    # Add learning rate decay
    parser.add_argument('--lr_decay_factor', type=float, default=0.8, 
                    help='Factor to slow down the learning rate decay (lower = slower decay)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.pe_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images, Test: {len(test_loader.dataset)} images")
    
    # Create model with appropriate positional encoding
    print(f"Creating model with {args.pe_type} positional encoding...")
    model_kwargs = {}
    if args.pe_type == 'polynomial_rpe':
        model_kwargs['polynomial_degree'] = args.polynomial_degree

    # if args.pe_type in ['rpe', 'polynomial_rpe']:
    #     model_kwargs['img_size'] = args.img_size
    #     model_kwargs['patch_size'] = args.patch_size
    
    model = create_vit_model(
        pe_type=args.pe_type,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
        **model_kwargs
    )
    model = model.to(device)

    # After creating the model:
    total_params = count_parameters(model)
    pe_params = count_pe_parameters(model, args.pe_type)

    print(f"Model has {total_params:,} total trainable parameters")
    print(f"Positional encoding '{args.pe_type}' uses {pe_params:,} parameters ({pe_params/total_params*100:.2f}% of total)")

    # Create model info dictionary
    model_info = {
        "pe_type": args.pe_type,
        "total_parameters": total_params,
        "pe_parameters": pe_params,
        "pe_percentage": pe_params/total_params*100,
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
        
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Add Label Smoothing
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
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
                # Multiply by factor (default 0.8) to slow down the decay rate
                progress = progress * lr_decay_factor
                return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = warmup_cosine_schedule(
        optimizer, 
        warmup_epochs=args.warmup_epochs, 
        total_epochs=args.epochs,
        lr_decay_factor=args.lr_decay_factor
    )
    
    # Train model
    print(f"Training for {args.epochs} epochs...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        mixup_alpha=args.mixup_alpha
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.4f}")
    
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
    plt.title(f'{args.pe_type} Accuracy')
    
    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{args.pe_type} Loss')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png')
    
    # Save results summary
    with open(output_dir / 'results.txt', 'w') as f:
        f.write(f"Model: Vision Transformer with {args.pe_type}\n")
        f.write(f"Embedding dimension: {args.embed_dim}\n")
        f.write(f"Depth: {args.depth}\n")
        f.write(f"Number of heads: {args.num_heads}\n")
        f.write(f"Training epochs: {args.epochs}\n")
        f.write(f"Final training accuracy: {history['train_acc'][-1]:.4f}\n")
        f.write(f"Final validation accuracy: {history['val_acc'][-1]:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()