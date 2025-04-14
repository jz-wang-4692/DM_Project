# DM_Project
This is a Columbia Data Mining class project, focusing on comparing the effect of different positional encoding methods for Vision Transformers (ViTs).

## Authors:
Chi Han, Jiazhen Wang, Ningping Wang, Lian Zhong

This project compares different positional encoding methods for Vision Transformers on CIFAR-10:
- Absolute Positional Encoding (APE)
- Rotary Positional Encoding (RoPE) variants from Heo et al., 2024
- Regular Relative Positional Encoding with 2L-1 learnable parameters
- Polynomial L1-distance based Relative Positional Encoding

## Acknowledgements

This project uses code from the RoPE-ViT repository:
@inproceedings{heo2024ropevit,
    title={Rotary Position Embedding for Vision Transformer},
    author={Heo, Byeongho and Park, Song and Han, Dongyoon and Yun, Sangdoo},
    year={2024},
    booktitle={European Conference on Computer Vision (ECCV)},
}

Parts of this code are based on the following repositories:
- RoPE-ViT (Apache-2.0): https://github.com/naver-ai/rope-vit
- CodeLlama (Meta): https://github.com/meta-llama/codellama

```
📦 Position Encoding Comparison
 ┣ 📂 models                  # Model implementations
 ┃ ┣ 📂 positional_encoding   # Positional encoding variants
 ┃ ┃ ┣ 📜 __init__.py         # Package initialization with imports
 ┃ ┃ ┣ 📜 Poly_RPE.py         # Polynomial RPE based on L1 distances
 ┃ ┃ ┣ 📜 RPE.py              # Standard RPE with 2L-1 learnable parameters 
 ┃ ┃ ┗ 📜 fixed_rope_mixed.py # RoPE variants, adapted, reduced patch size
 ┃ ┣ 📜 model_factory.py      # Factory for creating ViT variants with different PEs
 ┃ ┗ 📜 vit_base.py           # Base ViT implementation
 ┃
 ┣ 📂 training                # Training utilities
 ┃ ┗ 📜 trainer.py            # Training and evaluation loops
 ┃
 ┣ 📂 utils                   # Utility functions
 ┃ ┗ 📜 data.py               # CIFAR-10 data fetch, augment, split
 ┃
 ┣ 📂 rope-vit                # External RoPE implementation (from Heo et al.): https://github.com/naver-ai/rope-vit
 ┃ ┣ 📂 models                # Original RoPE implementations
 ┃ ┣ 📂 self-attn             # Self-attention mechanisms with RoPE
 ┃ ┗ 📜 ...                   # Other files from original repo
 ┃
 ┣ 📂 output                  # NOT Existing now: Training outputs (created during runtime)
 ┃ ┣ 📂 ape                   # Results for APE models
 ┃ ┣ 📂 rope_axial            # Results for RoPE-Axial models
 ┃ ┣ 📂 rope_mixed            # Results for RoPE-Mixed models
 ┃ ┣ 📂 rpe                   # Results for standard RPE models
 ┃ ┗ 📂 polynomial_rpe        # Results for Polynomial RPE models
 ┃
 ┣ 📜 main.py                 # Main script to run experiments, with visualization
 ┣ 📜 requirements.txt        # Project dependencies
 ┗ 📜 README.md               # Project documentation
 ```