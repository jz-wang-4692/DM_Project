import sys
import torch
import torch.nn as nn

# Add the RoPE repo to path
sys.path.append('./rope-vit')  # Adjust path as needed

# Import from the RoPE repository
from models.vit_rope import (
    rope_mixed_deit_small_patch16_LS,
    rope_axial_deit_small_patch16_LS
)

def get_rope_mixed_vit(img_size=32, patch_size=4, num_classes=10):
    """Returns a ViT with RoPE-Mixed from Heo et al. repo"""
    model = rope_mixed_deit_small_patch16_LS(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        pretrained=False
    )
    return model

def get_rope_axial_vit(img_size=32, patch_size=4, num_classes=10):
    """Returns a ViT with RoPE-Axial from Heo et al. repo"""
    model = rope_axial_deit_small_patch16_LS(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        pretrained=False
    )
    return model