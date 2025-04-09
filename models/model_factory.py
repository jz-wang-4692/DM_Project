"""
Model factory for creating Vision Transformer variants with different positional encodings.
This module provides a unified interface for creating ViT models with:
1. Absolute Positional Encoding (APE) - standard ViT
2. RoPE-Axial - from Heo et al. 2024
3. RoPE-Mixed - from Heo et al. 2024
4. Regular Relative Positional Encoding (RPE) - with 2L-1 learnable parameters
5. Polynomial RPE - using L1 distances with learnable coefficients
"""

import sys
import torch
import torch.nn as nn
import timm
from pathlib import Path

# Add RoPE-ViT path to sys.path for imports
ROPE_VIT_PATH = Path(__file__).parent.parent / "rope-vit"
sys.path.append(str(ROPE_VIT_PATH))

# Import RoPE implementations from the cloned repo
from rope_vit.models.vit_rope import (
    rope_mixed_deit_small_patch16_LS,
    rope_axial_deit_small_patch16_LS
)

# Import custom positional encoding implementations
from models.positional_encoding.RPE import RelativePositionalAttention
from models.positional_encoding.Poly_RPE import PolynomialPositionalAttention


def create_vit_model(
    pe_type,
    img_size=32,
    patch_size=4,
    in_channels=3,
    num_classes=10,
    embed_dim=192,
    depth=6,
    num_heads=8,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.1,
    attn_drop_rate=0.0,
    **kwargs
):
    """
    Factory function to create ViT models with different positional encodings.
    
    Args:
        pe_type (str): Type of positional encoding. One of:
                     ['ape', 'rope_axial', 'rope_mixed', 'rpe', 'polynomial_rpe']
        img_size (int): Input image size
        patch_size (int): Patch size
        in_channels (int): Number of input channels
        num_classes (int): Number of classes
        embed_dim (int): Embedding dimension
        depth (int): Transformer depth
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP ratio
        qkv_bias (bool): Whether to use bias in qkv projection
        drop_rate (float): Dropout rate
        attn_drop_rate (float): Attention dropout rate
        **kwargs: Additional arguments for specific PE types
    
    Returns:
        nn.Module: Vision Transformer with the specified positional encoding
    """
    
    if pe_type == 'ape':
        # Standard ViT with Absolute Positional Encoding from timm
        model = timm.create_model(
            'vit_small_patch16_224',
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            pretrained=False
        )
        
        # For CIFAR-10, adjust patch embedding if needed
        if img_size != 224:
            model.patch_embed.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )
        
        return model
    
    elif pe_type == 'rope_axial':
        # ViT with RoPE-Axial from Heo et al.
        model = rope_axial_deit_small_patch16_LS(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            num_classes=num_classes,
            pretrained=False
        )
        return model
    
    elif pe_type == 'rope_mixed':
        # ViT with RoPE-Mixed from Heo et al.
        model = rope_mixed_deit_small_patch16_LS(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            num_classes=num_classes,
            pretrained=False
        )
        return model
    
    elif pe_type == 'rpe':
        # Create a ViT with custom RPE attention
        model = timm.create_model(
            'vit_small_patch16_224',
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            pretrained=False
        )
        
        # For CIFAR-10, adjust patch embedding if needed
        if img_size != 224:
            model.patch_embed.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )
        
        # Replace attention with RPE attention
        for i, block in enumerate(model.blocks):
            block.attn = RelativePositionalAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=None,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate
            )
        
        return model
    
    elif pe_type == 'polynomial_rpe':
        # Create a ViT with Polynomial RPE attention
        polynomial_degree = kwargs.get('polynomial_degree', 3)
        
        model = timm.create_model(
            'vit_small_patch16_224',
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            pretrained=False
        )
        
        # For CIFAR-10, adjust patch embedding if needed
        if img_size != 224:
            model.patch_embed.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )
        
        # Replace attention with Polynomial RPE attention
        for i, block in enumerate(model.blocks):
            block.attn = PolynomialPositionalAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=None,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                polynomial_degree=polynomial_degree
            )
        
        return model
    
    else:
        raise ValueError(f"Unknown positional encoding type: {pe_type}")