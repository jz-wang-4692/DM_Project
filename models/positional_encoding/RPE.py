import torch
import torch.nn as nn
import math
import sys
from pathlib import Path

# Add paths for imports
ROPE_VIT_PATH = Path(__file__).parent.parent.parent / "rope-vit"
sys.path.append(str(ROPE_VIT_PATH))

# Add the self-attn directory specifically to handle the hyphen
sys.path.append(str(ROPE_VIT_PATH / "self-attn"))

# Now import directly from the module
from rope_self_attn import Attention

class RelativePositionalAttention(Attention):
    """Self-Attention with efficient additive Relative Positional Encoding
    
    This implementation uses a direct additive bias approach:
    softmax(Q*K^T / sqrt(d_QK) + RelPosBias)
    """
    
    def init(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., img_size=32, patch_size=4):
        super().init(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Dynamic sequence length calculation
        self.seq_len = (img_size // patch_size) ** 2
        
        # Create a learnable relative position bias table
        # Use a direct table for all relative positions and all heads
        size = 2 * self.seq_len - 1
        self.rel_pos_bias_table = nn.Parameter(torch.zeros(size, self.num_heads))
        
        # Initialize with truncated normal distribution
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=.02)
        
        # Create relative position index mapping
        pos_indices = torch.arange(self.seq_len)
        i, j = torch.meshgrid(pos_indices, pos_indices, indexing='ij')
        rel_pos_indices = i.flatten() - j.flatten() + self.seq_len - 1
        self.register_buffer('rel_pos_indices', rel_pos_indices.reshape(self.seq_len, self.seq_len))
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply relative position bias
        # Handle CLS token separately
        cls_attn = attn[:, :, 0:1, :]  # (B, H, 1, N)
        patch_attn = attn[:, :, 1:, 1:]  # (B, H, N-1, N-1)
        
        # Get relative position bias for patches
        # Shape: [seq_len, seq_len, num_heads]
        rel_pos_bias = self.rel_pos_bias_table[self.rel_pos_indices].permute(2, 0, 1)
        
        # Add bias directly to attention scores (efficient additive approach)
        # Permute rel_pos_bias to [num_heads, seq_len, seq_len] for addition
        patch_attn = patch_attn + rel_pos_bias.unsqueeze(0)  # [B, H, seq_len, seq_len]
        
        # Reconstruct full attention matrix
        attn = torch.cat([cls_attn, torch.cat([attn[:, :, 1:, 0:1], patch_attn], dim=3)], dim=2)
        
        # Apply softmax after adding the bias
        attn = attn.softmax(dim=-1)
        
        # Apply dropout
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x