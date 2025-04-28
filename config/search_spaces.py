"""
Search space definitions for Bayesian Optimization of ViT positional encodings.
This module defines parameter ranges and distributions for Optuna trials.
"""

import optuna
import torch

class SearchSpaces:
    @staticmethod
    def define_model_space(trial, pe_type):
        """Define search space for model architecture parameters"""
        params = {}
        
        # Common parameters for all models
        params['embed_dim'] = trial.suggest_categorical('embed_dim', [128, 192, 256, 320])
        params['depth'] = trial.suggest_int('depth', 6, 12, step=1)
        params['num_heads'] = trial.suggest_categorical('num_heads', [4, 8, 12, 16])
        
        # Only optimize patch_size if it makes sense for the model
        params['patch_size'] = trial.suggest_categorical('patch_size', [2, 4, 8])
        
        # Specific parameters for certain position encoding types
        if pe_type == 'polynomial_rpe':
            params['polynomial_degree'] = trial.suggest_int('polynomial_degree', 2, 5)
        
        return params
    
    @staticmethod
    def define_regularization_space(trial):
        """Define search space for regularization parameters to combat overfitting"""
        params = {}
    
            # First suggest mixup_alpha
        params['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.0, 0.4, step=0.05)
        
        # Then adjust label_smoothing range based on mixup_alpha
        if params['mixup_alpha'] > 0.3:
            # Use lower label_smoothing when mixup is high
            params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.0, 0.15, step=0.01)
        else:
            # Allow higher label_smoothing when mixup is lower
            params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.0, 0.2, step=0.01)
        
        # Dropout-related parameters
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5, step=0.05)
        
        # Data augmentation parameters
        params['random_erasing_prob'] = trial.suggest_float('random_erasing_prob', 0.0, 0.4, step=0.05)
        params['color_jitter_brightness'] = trial.suggest_float('color_jitter_brightness', 0.0, 0.3, step=0.05)
        params['color_jitter_contrast'] = trial.suggest_float('color_jitter_contrast', 0.0, 0.3, step=0.05)
        
        # Weight decay (L2 regularization)
        params['weight_decay'] = trial.suggest_float('weight_decay', 0.01, 0.1, log=True)
        
        # Early stopping parameters
        params['early_stopping_patience'] = trial.suggest_int('early_stopping_patience', 5, 12)
        params['early_stopping_delta'] = trial.suggest_float('early_stopping_delta',
                                                            2e-4,   # 0.02% absolute accuracy gain
                                                            1e-2,   # up to 1.0% gain
                                                            log=True) # use a log‑space than a linear equally‑spaced grid of values
        
        return params
    
    @staticmethod
    def define_optimization_space(trial):
        """Define search space for optimization parameters"""
        params = {}
        
        # Learning rate
        params['lr'] = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
        
        # Learning rate schedule parameters
        params['warmup_epochs'] = trial.suggest_int('warmup_epochs', 3, 10)
        params['lr_decay_factor'] = trial.suggest_float('lr_decay_factor', 0.6, 1.0, step=0.05)
        
        # Batch size 
        params['batch_size'] = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        
        return params
    
    @staticmethod
    def define_system_space():
        """Define fixed system parameters"""
        params = {}
        
        # System parameters
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        params['num_workers'] = 4
        params['random_crop_padding'] = 4  # Fixed for CIFAR-10
        params['output_dir'] = './output'  # Default output directory
        params['seed'] = 42  # Default random seed
        
        return params

    @staticmethod
    def calculate_progressive_patience(trial_number, min_patience=5, max_patience=15):
        """Calculate progressive patience based on trial number"""
        # Use lower patience for early trials, higher for later trials
        # This allows quickly pruning obviously bad configs while giving promising ones more time
        return min(min_patience + (trial_number // 5), max_patience)
    
    @staticmethod
    def get_trial_params(trial, pe_type):
        """Get complete trial parameters for the given positional encoding type"""
        params = {
            'pe_type': pe_type,
            'img_size': 32,  # Fixed for CIFAR-10
        }
        
        # Add parameters from different spaces
        params.update(SearchSpaces.define_model_space(trial, pe_type))
        params.update(SearchSpaces.define_regularization_space(trial))
        params.update(SearchSpaces.define_optimization_space(trial))
        params.update(SearchSpaces.define_system_space())  # Add system parameters
        
        # Add fixed parameters
        params['mlp_ratio'] = 4.0  # Fixed MLP ratio
        params['epochs'] = 100  # Increased from 50 to 100
        
        # # Adjust early stopping patience based on trial number for progressive patience
        # if 'early_stopping_patience' in params:
        #     params['early_stopping_patience'] = SearchSpaces.calculate_progressive_patience(
        #         trial.number, 
        #         min_patience=params['early_stopping_patience'], 
        #         max_patience=20
        #     )
        
        return params
    
    @staticmethod
    def add_parameter_constraints(trial, pe_type):
        """Add constraints between parameters"""
        constraints = {}
        
        # Ensure number of heads divides embedding dimension evenly
        embed_dim = trial.params.get('embed_dim', 192)
        num_heads = trial.params.get('num_heads', 8)
        
        if embed_dim % num_heads != 0:
            # Suggest closest valid number of heads
            valid_heads = [h for h in [4, 8, 12, 16] if embed_dim % h == 0]
            if valid_heads:
                constraints['num_heads'] = valid_heads[0]
        
        # Add constraint for mixup_alpha and label_smoothing combination
        # (too much of both can make training unstable)
        mixup_alpha = trial.params.get('mixup_alpha', 0.2)
        label_smoothing = trial.params.get('label_smoothing', 0.1)
        
        if mixup_alpha > 0.3 and label_smoothing > 0.15:
            # Reduce one of them to avoid over-regularization
            constraints['label_smoothing'] = 0.1
        
        return constraints