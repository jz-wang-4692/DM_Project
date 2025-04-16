"""
Search space definitions for Bayesian Optimization of ViT positional encodings.
This module defines parameter ranges and distributions for Optuna trials.
"""

import optuna

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
        
        # Dropout-related parameters
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5, step=0.05)
        params['attn_drop_rate'] = trial.suggest_float('attn_drop_rate', 0.0, 0.3, step=0.05)
        
        # Label smoothing (strong regularizer)
        params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.0, 0.2, step=0.01)
        
        # MixUp alpha parameter
        params['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.0, 0.4, step=0.05)
        
        # Data augmentation parameters
        params['random_erasing_prob'] = trial.suggest_float('random_erasing_prob', 0.0, 0.4, step=0.05)
        params['color_jitter_brightness'] = trial.suggest_float('color_jitter_brightness', 0.0, 0.3, step=0.05)
        params['color_jitter_contrast'] = trial.suggest_float('color_jitter_contrast', 0.0, 0.3, step=0.05)
        
        # Weight decay (L2 regularization)
        params['weight_decay'] = trial.suggest_float('weight_decay', 0.01, 0.1, log=True)
        
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
        
        # Add fixed parameters
        params['mlp_ratio'] = 4.0  # Fixed MLP ratio
        params['epochs'] = 50  # Fixed number of training epochs
        
        return params
    
    @staticmethod
    def add_parameter_constraints(trial, pe_type):
        """Add constraints between parameters"""
        # Ensure number of heads divides embedding dimension evenly
        embed_dim = trial.params.get('embed_dim', 192)
        num_heads = trial.params.get('num_heads', 8)
        
        if embed_dim % num_heads != 0:
            # Suggest closest valid number of heads
            valid_heads = [h for h in [4, 8, 12, 16] if embed_dim % h == 0]
            if valid_heads:
                return {'num_heads': valid_heads[0]}
        
        return {}