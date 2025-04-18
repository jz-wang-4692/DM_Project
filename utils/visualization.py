"""
Centralized visualization utilities for positional encoding comparison.
Provides standardized plotting functions for training metrics, optimization results,
and comparative analysis between different positional encoding methods.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
import optuna


# Set consistent styling for all plots
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = sns.color_palette("muted")
PE_TYPE_COLORS = {
    'ape': COLORS[0],
    'rope_axial': COLORS[1],
    'rope_mixed': COLORS[2],
    'rpe': COLORS[3],
    'polynomial_rpe': COLORS[4]
}


def set_plot_style(ax, title=None, xlabel=None, ylabel=None, legend=True):
    """Apply consistent styling to a plot"""
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    if legend:
        ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


# Training visualization functions
def plot_training_history(history, test_metrics=None, output_path=None, pe_type=None):
    """
    Plot training/validation accuracy and loss curves
    
    Args:
        history: Dictionary with training history (train_acc, val_acc, train_loss, val_loss)
        test_metrics: Optional tuple of (test_loss, test_acc) to include in the plot
        output_path: Path to save the plot (if None, just displays the plot)
        pe_type: Positional encoding type (for plot title)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training and validation accuracy
    ax1.plot(history['train_acc'], label='Train', color='blue', linewidth=2)
    ax1.plot(history['val_acc'], label='Validation', color='green', linewidth=2)
    
    if test_metrics and len(test_metrics) == 2:
        test_loss, test_acc = test_metrics
        ax1.axhline(y=test_acc, color='red', linestyle='--', 
                  label=f'Test ({test_acc:.4f})', linewidth=2)
    
    title = f'{pe_type} Accuracy' if pe_type else 'Training and Validation Accuracy'
    set_plot_style(ax1, title=title, xlabel='Epoch', ylabel='Accuracy')
    
    # Plot training and validation loss
    ax2.plot(history['train_loss'], label='Train', color='blue', linewidth=2)
    ax2.plot(history['val_loss'], label='Validation', color='green', linewidth=2)
    
    if test_metrics and len(test_metrics) == 2:
        test_loss, test_acc = test_metrics
        ax2.axhline(y=test_loss, color='red', linestyle='--', 
                  label=f'Test ({test_loss:.4f})', linewidth=2)
    
    title = f'{pe_type} Loss' if pe_type else 'Training and Validation Loss'
    set_plot_style(ax2, title=title, xlabel='Epoch', ylabel='Loss')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Optimization visualization functions
def plot_optimization_history(study, output_path=None):
    """Plot optimization history from an Optuna study"""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Get trials data
    trials = study.trials
    complete_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not complete_trials:
        return
    
    trials_df = pd.DataFrame({
        'number': [t.number for t in complete_trials],
        'value': [t.value for t in complete_trials],
    })
    
    # Plot trials
    ax.plot(trials_df['number'], trials_df['value'], marker='o', linestyle='-', 
            color='blue', alpha=0.7)
    
    # Plot running best value
    running_best = pd.Series(trials_df['value']).cummax()
    ax.plot(trials_df['number'], running_best, marker='', linestyle='-', 
            color='red', alpha=0.9, linewidth=2, label='Best Value')
    
    set_plot_style(ax, title='Optimization History', 
                  xlabel='Trial Number', ylabel='Objective Value')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_parameter_importances(study, output_path=None):
    """Plot parameter importance from an Optuna study"""
    try:
        # Get parameter importances
        param_importances = optuna.importance.get_param_importances(study)
        
        # Convert to DataFrame for plotting
        importance_df = pd.DataFrame({
            'Parameter': list(param_importances.keys()),
            'Importance': list(param_importances.values())
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(importance_df['Parameter'], importance_df['Importance'], 
                      color=COLORS)
        
        set_plot_style(ax, title='Parameter Importances', 
                      xlabel='Parameter', ylabel='Importance Score', legend=False)
        
        # Rotate x labels for readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        print(f"Could not plot parameter importance: {e}")


def plot_param_vs_performance(study, param_name, output_path=None):
    """Plot relationship between parameter value and performance"""
    # Extract parameter values and corresponding performances
    values = []
    scores = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and param_name in trial.params:
            values.append(trial.params[param_name])
            scores.append(trial.value)
    
    if not values:
        return  # Skip if no data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # For categorical parameters, do a boxplot instead
    if isinstance(values[0], str) or isinstance(values[0], bool):
        # Convert to DataFrame for categorical plotting
        df = pd.DataFrame({'value': values, 'score': scores})
        sns.boxplot(x='value', y='score', data=df, ax=ax, palette='muted')
        set_plot_style(ax, title=f'Impact of {param_name} on Model Performance', 
                      xlabel=param_name, ylabel='Validation Accuracy', legend=False)
    else:
        # Scatter plot for numerical parameters
        ax.scatter(values, scores, alpha=0.7)
        
        # Add trend line
        if len(values) > 2:
            try:
                z = np.polyfit(values, scores, 1)
                p = np.poly1d(z)
                x_sorted = sorted(values)
                ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.8)
            except:
                pass  # Skip trend line if it can't be calculated
        
        set_plot_style(ax, title=f'Impact of {param_name} on Model Performance', 
                      xlabel=param_name, ylabel='Validation Accuracy', legend=False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Comparative visualization functions
def plot_accuracy_comparison(results, output_path=None):
    """
    Plot accuracy comparison between all positional encoding types
    
    Args:
        results: List of result dictionaries containing pe_type and test_accuracy
        output_path: Path to save the plot
    """
    # Organize data by PE type
    data = defaultdict(list)
    for result in results:
        pe_type = result['pe_type']
        data[pe_type].append(result['test_accuracy'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for plotting
    pe_types = list(data.keys())
    mean_accuracy = [np.mean(data[pe]) for pe in pe_types]
    std_accuracy = [np.std(data[pe]) for pe in pe_types]
    
    # Sort by mean accuracy
    sorted_indices = np.argsort(mean_accuracy)[::-1]  # Descending order
    pe_types = [pe_types[i] for i in sorted_indices]
    mean_accuracy = [mean_accuracy[i] for i in sorted_indices]
    std_accuracy = [std_accuracy[i] for i in sorted_indices]
    
    # Use consistent colors
    colors = [PE_TYPE_COLORS.get(pe, 'gray') for pe in pe_types]
    
    # Plot
    bars = ax.bar(pe_types, mean_accuracy, yerr=std_accuracy, capsize=10, 
                  alpha=0.7, color=colors)
    
    # Add value labels on top of bars
    for bar, accuracy in zip(bars, mean_accuracy):
        ax.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + 0.005, 
                f'{accuracy:.4f}', 
                ha='center', va='bottom', rotation=0)
    
    set_plot_style(ax, title='Comparison of Positional Encoding Methods (Test Accuracy)', 
                  xlabel='Positional Encoding Type', ylabel='Test Accuracy', legend=False)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_convergence_comparison(results, output_path=None):
    """
    Plot convergence curves for all positional encoding types
    
    Args:
        results: List of result dictionaries containing pe_type and history
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Group by PE type
    pe_types = set(result['pe_type'] for result in results)
    
    # Plot validation accuracy curves
    for pe_type in pe_types:
        # Collect validation accuracy curves for this PE type
        val_curves = []
        for result in results:
            if result['pe_type'] == pe_type:
                val_curves.append(result['history']['val_acc'])
        
        # Calculate mean and std
        max_length = max(len(curve) for curve in val_curves)
        aligned_curves = []
        for curve in val_curves:
            # Pad shorter curves if necessary
            if len(curve) < max_length:
                aligned_curves.append(curve + [curve[-1]] * (max_length - len(curve)))
            else:
                aligned_curves.append(curve)
        
        val_acc_mean = np.mean(aligned_curves, axis=0)
        val_acc_std = np.std(aligned_curves, axis=0)
        
        # Plot mean curve with shaded std region
        epochs = np.arange(1, max_length + 1)
        color = PE_TYPE_COLORS.get(pe_type, None)
        ax1.plot(epochs, val_acc_mean, label=pe_type, linewidth=2, color=color)
        ax1.fill_between(epochs, 
                       val_acc_mean - val_acc_std, 
                       val_acc_mean + val_acc_std, 
                       alpha=0.2, color=color)
    
    set_plot_style(ax1, title='Convergence Comparison (Validation Accuracy)', 
                  xlabel='Epoch', ylabel='Validation Accuracy')
    
    # Plot training accuracy curves
    for pe_type in pe_types:
        # Collect training accuracy curves for this PE type
        train_curves = []
        for result in results:
            if result['pe_type'] == pe_type:
                train_curves.append(result['history']['train_acc'])
        
        # Calculate mean and std
        max_length = max(len(curve) for curve in train_curves)
        aligned_curves = []
        for curve in train_curves:
            # Pad shorter curves if necessary
            if len(curve) < max_length:
                aligned_curves.append(curve + [curve[-1]] * (max_length - len(curve)))
            else:
                aligned_curves.append(curve)
        
        train_acc_mean = np.mean(aligned_curves, axis=0)
        train_acc_std = np.std(aligned_curves, axis=0)
        
        # Plot mean curve with shaded std region
        epochs = np.arange(1, max_length + 1)
        color = PE_TYPE_COLORS.get(pe_type, None)
        ax2.plot(epochs, train_acc_mean, label=pe_type, linewidth=2, color=color)
        ax2.fill_between(epochs, 
                       train_acc_mean - train_acc_std, 
                       train_acc_mean + train_acc_std, 
                       alpha=0.2, color=color)
    
    set_plot_style(ax2, title='Convergence Comparison (Training Accuracy)', 
                  xlabel='Epoch', ylabel='Training Accuracy')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_parameter_efficiency(results, output_path=None):
    """
    Plot parameter efficiency (accuracy vs parameter count)
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Organize data
    pe_types = []
    accuracies = []
    total_params = []
    pe_params = []
    
    # Group by PE type and calculate means
    pe_type_data = defaultdict(lambda: {'acc': [], 'total': [], 'pe': []})
    
    for result in results:
        pe_type = result['pe_type']
        pe_type_data[pe_type]['acc'].append(result['test_accuracy'])
        pe_type_data[pe_type]['total'].append(result['total_parameters'])
        pe_type_data[pe_type]['pe'].append(result['pe_parameters'])
    
    for pe_type, data in pe_type_data.items():
        pe_types.append(pe_type)
        accuracies.append(np.mean(data['acc']))
        total_params.append(np.mean(data['total']))
        pe_params.append(np.mean(data['pe']))
    
    # Convert to arrays
    accuracies = np.array(accuracies)
    total_params = np.array(total_params)
    pe_params = np.array(pe_params)
    
    # Plot with total parameters
    for pe, acc, params in zip(pe_types, accuracies, total_params):
        color = PE_TYPE_COLORS.get(pe, None)
        ax1.scatter(params, acc, s=120, label=pe, color=color, edgecolor='white')
        ax1.text(params*1.02, acc, pe, fontsize=10)
    
    set_plot_style(ax1, title='Accuracy vs. Total Parameters', 
                  xlabel='Total Parameters', ylabel='Test Accuracy', legend=False)
    
    # Add reference line to see if more parameters = better performance
    if len(pe_types) > 1:
        try:
            z = np.polyfit(total_params, accuracies, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(total_params)*0.9, max(total_params)*1.1, 100)
            ax1.plot(x_range, p(x_range), "k--", alpha=0.5, linewidth=1)
        except:
            pass
    
    # Plot with PE parameters
    for pe, acc, params in zip(pe_types, accuracies, pe_params):
        color = PE_TYPE_COLORS.get(pe, None)
        ax2.scatter(params, acc, s=120, label=pe, color=color, edgecolor='white')
        ax2.text(params*1.02, acc, pe, fontsize=10)
    
    set_plot_style(ax2, title='Accuracy vs. PE Parameters', 
                  xlabel='PE Parameters', ylabel='Test Accuracy', legend=True)
    
    # Add reference line
    if len(pe_types) > 1:
        try:
            z = np.polyfit(pe_params, accuracies, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(pe_params)*0.9, max(pe_params)*1.1, 100)
            ax2.plot(x_range, p(x_range), "k--", alpha=0.5, linewidth=1)
        except:
            pass
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_configuration_heatmap(results, output_path=None):
    """
    Plot heatmap of optimal hyperparameter configurations
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the plot
    """
    # Collect optimal hyperparameters for each PE type
    optimal_configs = {}
    
    # Group by PE type
    pe_type_results = defaultdict(list)
    for result in results:
        pe_type_results[result['pe_type']].append(result)
    
    # For each PE type, find the best result (highest test accuracy)
    for pe_type, type_results in pe_type_results.items():
        best_result = max(type_results, key=lambda x: x['test_accuracy'])
        optimal_configs[pe_type] = best_result['config']
    
    # Select common hyperparameters to compare
    common_params = [
        'drop_rate', 'label_smoothing', 'mixup_alpha', 'weight_decay',
        'lr', 'batch_size', 'embed_dim', 'depth', 'num_heads'
    ]
    
    # Create data for heatmap
    heatmap_data = []
    for pe_type, config in optimal_configs.items():
        row = {'PE Type': pe_type}
        for param in common_params:
            if param in config:
                row[param] = config[param]
            else:
                row[param] = None
        heatmap_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(heatmap_data)
    df.set_index('PE Type', inplace=True)
    
    # Create a custom normalized colormap for each parameter column
    # to highlight differences between methods
    fig, ax = plt.subplots(figsize=(14, len(optimal_configs) * 1.2))
    
    # Create the heatmap
    cmap = sns.cm.rocket_r
    sns.heatmap(df, annot=True, cmap=cmap, cbar=False, fmt='.3g', ax=ax)
    
    ax.set_title('Optimal Hyperparameter Configurations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_overfitting_analysis(results, output_path=None):
    """
    Plot analysis of overfitting (train-val accuracy gap) for each PE type
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the plot
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Group by PE type
    pe_types = sorted(set(result['pe_type'] for result in results))
    
    # Calculate overfitting metrics
    overfitting_gap = []
    pe_labels = []
    
    for pe_type in pe_types:
        type_results = [r for r in results if r['pe_type'] == pe_type]
        if not type_results:
            continue
            
        # Calculate mean train-val gap for final epoch
        gaps = []
        for result in type_results:
            try:
                history = result['history']
                final_train = history['train_acc'][-1]
                final_val = history['val_acc'][-1]
                gap = final_train - final_val
                gaps.append(gap)
            except (KeyError, IndexError):
                continue
        
        if gaps:
            mean_gap = np.mean(gaps)
            overfitting_gap.append(mean_gap)
            pe_labels.append(pe_type)
    
    # Sort by overfitting gap (ascending)
    sorted_indices = np.argsort(overfitting_gap)
    sorted_gaps = [overfitting_gap[i] for i in sorted_indices]
    sorted_labels = [pe_labels[i] for i in sorted_indices]
    
    # Plot overfitting gap bar chart
    colors = [PE_TYPE_COLORS.get(pe, 'gray') for pe in sorted_labels]
    bars = ax1.bar(sorted_labels, sorted_gaps, color=colors, alpha=0.7)
    
    # Add value labels
    for bar, gap in zip(bars, sorted_gaps):
        ax1.text(bar.get_x() + bar.get_width()/2., 
                gap + 0.002 if gap > 0 else gap - 0.02, 
                f'{gap:.4f}', 
                ha='center', va='bottom' if gap > 0 else 'top',
                rotation=0)
    
    set_plot_style(ax1, title='Overfitting Gap (Train - Val Accuracy)', 
                  xlabel='Positional Encoding Type', ylabel='Accuracy Gap', 
                  legend=False)
    
    # Plot train vs val accuracy scatter plot
    for pe_type in pe_types:
        type_results = [r for r in results if r['pe_type'] == pe_type]
        if not type_results:
            continue
            
        train_accs = []
        val_accs = []
        
        for result in type_results:
            try:
                history = result['history']
                train_accs.append(history['train_acc'][-1])
                val_accs.append(history['val_acc'][-1])
            except (KeyError, IndexError):
                continue
        
        if train_accs and val_accs:
            color = PE_TYPE_COLORS.get(pe_type, None)
            ax2.scatter(train_accs, val_accs, label=pe_type, color=color, 
                       s=80, alpha=0.7, edgecolor='white')
    
    # Add diagonal line (perfect fit, no overfitting)
    min_val = min([min(ax2.get_xlim()[0], ax2.get_ylim()[0]), 0.7])
    max_val = max([max(ax2.get_xlim()[1], ax2.get_ylim()[1]), 1.0])
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    set_plot_style(ax2, title='Train vs. Validation Accuracy', 
                  xlabel='Train Accuracy', ylabel='Validation Accuracy', 
                  legend=True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_slice(study, param_name, output_path=None):
    """
    Create a slice plot for a parameter showing its effect on performance
    
    Args:
        study: Optuna study
        param_name: Parameter name to plot
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gather data
    param_values = []
    objective_values = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and param_name in trial.params:
            param_values.append(trial.params[param_name])
            objective_values.append(trial.value)
    
    if not param_values:
        return
    
    # Sort data by parameter value for line plot
    sorted_indices = np.argsort(param_values)
    sorted_param_values = [param_values[i] for i in sorted_indices]
    sorted_objectives = [objective_values[i] for i in sorted_indices]
    
    # Plot the values
    ax.plot(sorted_param_values, sorted_objectives, 'o-', color='blue', alpha=0.7)
    
    set_plot_style(ax, title=f'Parameter Slice: {param_name}', 
                  xlabel=param_name, ylabel='Objective Value', legend=False)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()