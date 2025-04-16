# Bayesian Optimization Guide for Vision Transformer Positional Encodings

This document provides detailed instructions and explanations for using our Bayesian Optimization (BO) framework to find optimal hyperparameters for different positional encoding methods in Vision Transformers on the CIFAR-10 dataset.

## Table of Contents

1. [Introduction](#introduction)
2. [Scripts Overview](#scripts-overview)
3. [Hyperparameter Optimization Strategy](#hyperparameter-optimization-strategy)
4. [Usage Guide](#usage-guide)
5. [Understanding Visualizations](#understanding-visualizations)
6. [Analyzing Results](#analyzing-results)
7. [Tips for Efficient Optimization](#tips-for-efficient-optimization)

## Introduction

Our framework optimizes hyperparameters for five positional encoding methods:
- Absolute Positional Encoding (APE)
- RoPE-Axial
- RoPE-Mixed
- Relative Positional Encoding (RPE)
- Polynomial L1-distance based RPE

The primary goal is to combat overfitting and maximize validation accuracy on CIFAR-10. The optimization process prioritizes regularization parameters while maintaining appropriate model capacity.

## Scripts Overview

Our Bayesian Optimization framework consists of three main scripts:

### 1. `bo_main.py`: Single Encoding Optimizer

This script handles the optimization of a single positional encoding method:
- Creates and manages an Optuna study
- Executes training trials with different hyperparameter sets
- Collects performance metrics and generates visualizations
- Identifies and saves the best configuration

### 2. `run_bo.py`: Multi-Encoding Orchestrator

This script orchestrates optimization across all encoding methods:
- Creates the necessary directory structure
- Runs `bo_main.py` for each positional encoding type
- Manages database connections and logging
- Tracks timing and completion status

### 3. `run_final_comparison.py`: Comparative Evaluator

This script compares the best configurations found:
- Trains final models using optimized configurations
- Uses multiple random seeds for statistical significance
- Generates comparative visualizations across methods
- Creates summary tables and analysis reports

## Hyperparameter Optimization Strategy

### Optimization Focus

Our framework prioritizes regularization parameters to address overfitting:

#### Dropout Parameters
- `drop_rate`: 0.1 to 0.5 (model dropout)
- `attn_drop_rate`: 0.0 to 0.3 (attention-specific dropout)

#### Data Augmentation Parameters
- `mixup_alpha`: 0.0 to 0.4 (controls MixUp augmentation intensity)
- `random_erasing_prob`: 0.0 to 0.4 (random erasing probability)
- `color_jitter_brightness/contrast`: 0.0 to 0.3 (image augmentation)

#### Loss Function Regularization
- `label_smoothing`: 0.0 to 0.2 (prevents overconfidence)
- `weight_decay`: 0.01 to 0.1 (logarithmic scale for L2 regularization)

#### Model Architecture Parameters
- `embed_dim`: [128, 192, 256, 320] (embedding dimension)
- `depth`: 6 to 12 (transformer layers)
- `num_heads`: [4, 8, 12, 16] (attention heads)
- `patch_size`: [2, 4, 8] (vision transformer patch size)

#### Optimization Parameters
- `lr`: 1e-5 to 5e-3 (logarithmic scale)
- `batch_size`: [64, 128, 256, 512]
- `warmup_epochs`: 3 to 10
- `lr_decay_factor`: 0.6 to 1.0

### Optuna Configuration

The optimization uses Optuna with the following settings:
- **Sampler**: Tree-structured Parzen Estimator (TPESampler)
- **Pruner**: MedianPruner (5 startup trials, interval of 3)
- **Storage**: SQLite database (default)
- **Objective**: Maximize validation accuracy

## Usage Guide

### Running Full Optimization

To optimize all positional encoding methods sequentially:

```bash
python scripts/run_bo.py --n_trials=50
```

This will:
1. Create the necessary database and output directories
2. Run 50 optimization trials for each encoding method
3. Save the best configuration for each method
4. Generate analysis visualizations and files

### Optimizing a Single Method

To optimize a specific positional encoding method:

```bash
python scripts/bo_main.py --pe_type=rope_axial --n_trials=50
```

### Running Final Comparison

After optimization is complete, compare the best configurations:

```bash
python scripts/run_final_comparison.py --seeds 42 43 44
```

This will train final models with multiple seeds for statistical significance.

## Understanding Visualizations

Our framework generates extensive visualizations through the centralized `utils/visualization.py` module. Here's how to interpret each visualization type:

### Optimization Progress Visualizations

#### Optimization History
- **Filename**: `optimization_history.png`
- **Purpose**: Tracks the validation accuracy of each trial 
- **Interpretation**: Upward trend shows the optimization process finding better hyperparameters
- **Key Features**: Best value line highlights the cumulative maximum

#### Parameter Importance
- **Filename**: `parameter_importances.png`
- **Purpose**: Shows which hyperparameters most affect performance
- **Interpretation**: Taller bars indicate parameters with greater impact
- **Key Features**: Parameters are ordered by importance

#### Parameter vs. Performance Plots
- **Filename**: `{param_name}_performance.png`
- **Purpose**: Shows relationship between specific parameter values and performance
- **Interpretation**: Trend lines indicate optimal parameter ranges
- **Key Features**: Scatter points represent individual trials, red trend line shows correlation

#### Parameter Slice Plots
- **Filename**: `{param_name}_slice.png`
- **Purpose**: Shows parameter effect while holding others constant
- **Interpretation**: Peaks indicate optimal parameter values
- **Key Features**: Confidence intervals indicate prediction uncertainty

### Comparative Visualizations

#### Accuracy Comparison
- **Filename**: `accuracy_comparison.png`
- **Purpose**: Compares test accuracy across encoding methods
- **Interpretation**: Higher bars indicate better methods
- **Key Features**: Error bars show variance across seeds

#### Convergence Comparison
- **Filename**: `convergence_comparison.png`
- **Purpose**: Shows learning curves for different methods
- **Interpretation**: Faster/higher curves indicate better methods
- **Key Features**: Shaded regions show variance across seeds

#### Parameter Efficiency
- **Filename**: `parameter_efficiency.png`
- **Purpose**: Plots accuracy vs. parameter count
- **Interpretation**: Points in top-left are most efficient
- **Key Features**: Two plots for total vs. PE-specific parameters

#### Configuration Heatmap
- **Filename**: `configuration_heatmap.png`
- **Purpose**: Visualizes optimal hyperparameters across methods
- **Interpretation**: Color intensity shows parameter values
- **Key Features**: Helps identify parameter patterns across methods

#### Overfitting Analysis
- **Filename**: `overfitting_analysis.png`
- **Purpose**: Analyzes training-validation accuracy gaps
- **Interpretation**: Smaller gaps indicate less overfitting
- **Key Features**: Includes scatter plot of train vs. val accuracy

## Analyzing Results

### Key Result Files

After optimization completes, analyze these key files:

1. **Parameter Sensitivity Analysis**
   - **Location**: `bo_results/{pe_type}/parameter_sensitivity.json`
   - **Content**: Detailed impact analysis of each parameter
   - **Usage**: Identify which parameters most affect performance

2. **Experiment Summary**
   - **Location**: `bo_results/{pe_type}/experiment_summary.json`
   - **Content**: Overview of entire optimization process
   - **Usage**: Review trial statistics and best configuration details

3. **Best Configuration**
   - **Location**: `best_configs/{pe_type}/config.json`
   - **Content**: Optimal hyperparameter set
   - **Usage**: Use for final model training or further experimentation

4. **Comparative Summary**
   - **Location**: `final_models/summary_table.txt`
   - **Content**: Performance metrics across all encoding methods
   - **Usage**: Identify the best performing positional encoding method

### Interpreting Optimization Results

When analyzing results, look for:

1. **Most Important Parameters**: Check parameter importance plots to see which parameters have the greatest impact on performance.

2. **Optimal Regularization Strength**: Examine slice plots for dropout, weight decay, and other regularization parameters to find optimal strength.

3. **Architecture Trade-offs**: Review parameter vs. performance plots for embed_dim, depth, and heads to understand model capacity requirements.

4. **Convergence Patterns**: Compare convergence plots to understand which methods learn faster or generalize better.

5. **Parameter Efficiency**: Use parameter efficiency plots to identify methods that achieve good performance with fewer parameters.

## Tips for Efficient Optimization

1. **Start Small**: Begin with fewer trials (10-20) to verify your setup before running full optimization.

2. **Focus Regularization**: If time is limited, prioritize optimizing regularization parameters first (dropout, weight decay, label smoothing).

3. **Sequential Optimization**: Run positional encoding methods sequentially rather than simultaneously to maximize GPU utilization.

4. **Check Early Trials**: Monitor the first 5-10 trials to ensure they complete successfully before leaving long runs unattended.

5. **Resume Interrupted Runs**: Use the same study name to resume optimization if it's interrupted.

6. **Analyze Mid-Optimization**: You can analyze partial results even before optimization completes by examining the SQLite database.

7. **Pruning Sensitivity**: If optimization seems too aggressive in pruning, adjust the pruner parameters in `bo_main.py`.

8. **Multiple Seeds**: For final evaluation, always use multiple seeds (at least 3) to verify statistical significance.
