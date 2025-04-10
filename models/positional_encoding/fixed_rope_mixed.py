"""
Monkey-patching approach to fix gradient issues in RoPE Mixed implementation.
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add the RoPE-ViT repo path to the system path
ROPE_VIT_PATH = Path(__file__).parent.parent.parent / "rope-vit"
sys.path.append(str(ROPE_VIT_PATH))

# Fixed version of reshape_for_broadcast that avoids in-place operations
def fixed_reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    shape = [1] * ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape[-2:] = freqs_cis.shape
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape[-3:] = freqs_cis.shape
    return freqs_cis.reshape(*shape)  # Uses new shape instead of view

# Fixed version of apply_rotary_emb that avoids in-place operations
def fixed_apply_rotary_emb(xq, xk, freqs_cis):
    # Make copies to avoid modifying the originals
    xq_copy = xq.clone()
    xk_copy = xk.clone()
    
    # Convert to complex representation (on copies)
    xq_ = torch.view_as_complex(xq_copy.float().reshape(*xq_copy.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk_copy.float().reshape(*xk_copy.shape[:-1], -1, 2))
    
    # Use our fixed reshape function
    freqs_cis_view = fixed_reshape_for_broadcast(freqs_cis, xq_)
    
    # Apply rotary embeddings
    xq_out = torch.view_as_real(xq_ * freqs_cis_view).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_view).flatten(3)
    
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

# Wrapper class for the model that uses fixed operations
class FixedRoPEMixedModel(nn.Module):
    def __init__(self, img_size=32, num_classes=10):
        super().__init__()
        # Import here to ensure our monkey-patching happens first
        from models.vit_rope import rope_mixed_deit_small_patch16_LS
        
        # Replace the original function with our fixed version
        # This is monkey-patching - we're replacing the function globally
        import models.vit_rope as vit_rope
        original_apply_rotary_emb = vit_rope.apply_rotary_emb
        vit_rope.apply_rotary_emb = fixed_apply_rotary_emb
        
        # Now create the model with the patched function
        self.model = rope_mixed_deit_small_patch16_LS(
            img_size=img_size,
            num_classes=num_classes,
            pretrained=False
        )
    
    def forward(self, x):
        return self.model(x)