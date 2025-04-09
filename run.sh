# Currently, config is loaded in main.py, will have a seperate config file for customization

# Train with APE (default)
python main.py --pe_type ape --epochs 100

# Train with RoPE-Axial
python main.py --pe_type rope_axial --epochs 100

# Train with RoPE-Mixed
python main.py --pe_type rope_mixed --epochs 100  

# Train with standard RPE
python main.py --pe_type rpe --epochs 100

# Train with Polynomial RPE
python main.py --pe_type polynomial_rpe --polynomial_degree 3 --epochs 100