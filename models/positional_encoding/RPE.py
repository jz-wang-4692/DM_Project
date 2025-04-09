import torch
import torch.nn as nn
import math
import sys

# Add the RoPE repo to path
sys.path.append('./rope-vit')

# Import the base attention implementation from the RoPE repository
from self_attn.rope_self_attn import Attention

class RelativePositionalAttention(Attention):
    """Self-Attention with regular Relative Positional Encoding (2L-1 parameters)"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # For CIFAR-10 with patch size 4, we get 8x8=64 patches
        self.seq_len = 64  # Calculate based on image size and patch size
        
        # Create a learnable relative position embedding
        # We need 2*seq_len-1 parameters for all relative positions
        self.rel_pos_embed = nn.Parameter(torch.zeros(2 * self.seq_len - 1, self.head_dim))
        # Initialize with normal distribution
        nn.init.trunc_normal_(self.rel_pos_embed, std=.02)
        
        # Create indices for relative positions
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
        # Extract CLS token interaction separately
        cls_attn = attn[:, :, 0:1, :]  # (B, H, 1, N)
        patch_attn = attn[:, :, 1:, 1:]  # (B, H, N-1, N-1)
        
        # Get relative position embeddings for patches
        rel_pos_bias = self.rel_pos_embed[self.rel_pos_indices]  # (seq_len, seq_len, head_dim)
        
        # Convert to attention bias through dot product with q
        # Reshape q_patch to (B, H, N-1, head_dim)
        q_patch = q[:, :, 1:]
        
        # Compute position-aware attention scores
        # (B, H, N-1, head_dim) × (N-1, N-1, head_dim) -> (B, H, N-1, N-1)
        rel_logits = torch.einsum('bhid,ijd->bhij', q_patch, rel_pos_bias)
        
        # Add to patch attention
        patch_attn = patch_attn + rel_logits
        
        # Reconstruct full attention
        attn = torch.cat([cls_attn, torch.cat([attn[:, :, 1:, 0:1], patch_attn], dim=3)], dim=2)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x