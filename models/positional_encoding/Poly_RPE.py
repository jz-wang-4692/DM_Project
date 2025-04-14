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

class PolynomialPositionalAttention(Attention):
    """Self-Attention with Polynomial Relative Positional Encoding based on L1 distances"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., polynomial_degree=3, img_size=32, patch_size=4):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.dim = dim
        self.num_heads = num_heads
        self.polynomial_degree = polynomial_degree

        # Dynamic grid size calculation
        self.grid_size = img_size // patch_size
        
        # Create learnable coefficients for the polynomial
        # One set per attention head for more expressivity
        self.coefficients = nn.Parameter(
            torch.randn(self.num_heads, polynomial_degree + 1)
        )
        
        # Precompute L1 distances between all patch positions
        self._precompute_distances()
    
    def _precompute_distances(self):
        # Create 2D grid coordinates with float dtype
        pos_i, pos_j = torch.meshgrid(
            torch.arange(self.grid_size, dtype=torch.float32),  # Use float32 type
            torch.arange(self.grid_size, dtype=torch.float32),  # Use float32 type
            indexing='ij'
        )
        pos_i = pos_i.flatten()  # [grid_size²]
        pos_j = pos_j.flatten()  # [grid_size²]
        
        # Compute L1 distances between all positions
        dist_i = pos_i.unsqueeze(1) - pos_i.unsqueeze(0)  # row distances
        dist_j = pos_j.unsqueeze(1) - pos_j.unsqueeze(0)  # column distances
        
        manhattan_dist = torch.abs(dist_i) + torch.abs(dist_j)  # L1 distance
        
        # Precompute powers for polynomial (ensure float type)
        powers = torch.stack([manhattan_dist ** i for i in range(self.polynomial_degree + 1)])
        self.register_buffer('dist_powers', powers)  # [degree+1, grid_size², grid_size²]
    
    def compute_polynomial(self):
        # Compute polynomial values per head
        # coefficients: [num_heads, degree+1]
        # dist_powers: [degree+1, grid_size², grid_size²]
        poly_values = torch.einsum('hd,dpq->hpq', self.coefficients, self.dist_powers)
        return poly_values  # [num_heads, grid_size², grid_size²]
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Compute polynomial values
        poly_values = self.compute_polynomial()  # [num_heads, grid_size², grid_size²]
        
        # Extract CLS token interaction separately
        cls_attn = attn[:, :, 0:1, :]  # (B, H, 1, N)
        patch_attn = attn[:, :, 1:, 1:]  # (B, H, N-1, N-1)
        
        # Add polynomial bias to patch attention
        # Broadcast across batch dimension
        patch_attn = patch_attn + poly_values.unsqueeze(0)
        
        # Reconstruct full attention
        attn = torch.cat([cls_attn, torch.cat([attn[:, :, 1:, 0:1], patch_attn], dim=3)], dim=2)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x