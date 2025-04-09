# DM_Project
This is a Columbia Data Mining class project, focusing on comparing the effect of different positional encoding methods for Vision Transformers (ViTs).

## Authors:
Chi Han, Jiazhen Wang, Ningping Wang, Lian Zhong

This project compares different positional encoding methods for Vision Transformers on CIFAR-10:
- Absolute Positional Encoding (APE)
- Rotary Positional Encoding (RoPE) variants from Heo et al., 2024
- Regular Relative Positional Encoding with 2L-1 learnable parameters
- Polynomial L1-distance based Relative Positional Encoding

## Structure
- `models/`: Contains ViT implementations with different positional encodings
- `training/`: Training loops and optimization code
- `utils/`: Data loading and evaluation utilities