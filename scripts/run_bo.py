"""
Script to run Bayesian Optimization for all positional encoding types.
"""

import argparse
import subprocess
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run Bayesian Optimization for all positional encoding types')
    
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of optimization trials per encoding type')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--storage_type', type=str, default='sqlite',
                        choices=['sqlite', 'mysql'], help='Type of database for Optuna storage')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout for each encoding type optimization in seconds')
    parser.add_argument('--pe_types', type=str, nargs='+', 
                        default=['ape', 'rope_axial', 'rope_mixed', 'rpe', 'polynomial_rpe'],
                        help='Positional encoding types to optimize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    bo_results_dir = output_dir / 'bo_results'
    best_configs_dir = output_dir / 'best_configs'
    
    for pe_type in args.pe_types:
        type_dir = bo_results_dir / pe_type
        type_dir.mkdir(parents=True, exist_ok=True)
        
        config_dir = best_configs_dir / pe_type
        config_dir.mkdir(parents=True, exist_ok=True)
    
    # Run optimization for each positional encoding type
    for pe_type in args.pe_types:
        logger.info(f"Starting optimization for {pe_type}")
        start_time = time.time()
        
        # Set up storage string based on type
        if args.storage_type == 'sqlite':
            storage = f"sqlite:///{bo_results_dir}/{pe_type}/optuna.db"
        else:
            # Assuming MySQL connection string would be provided externally
            storage = f"mysql://user:password@localhost/optuna"
        
        # Create study name
        study_name = f"{pe_type}_optimization"
        
        # Build command to run scripts/bo_main.py (updated path)
        cmd = [
            "python", "scripts/bo_main.py",  # Updated to use scripts/ directory
            f"--pe_type={pe_type}",
            f"--n_trials={args.n_trials}",
            f"--study_name={study_name}",
            f"--storage={storage}",
            f"--output_dir={args.output_dir}",
            f"--seed={args.seed}"
        ]
        
        if args.timeout:
            cmd.append(f"--timeout={args.timeout}")
        
        # Run the command
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Optimization for {pe_type} failed with error: {e}")
            continue
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Completed optimization for {pe_type} in {duration/3600:.2f} hours")
        
    logger.info("All optimizations completed")

if __name__ == "__main__":
    main()