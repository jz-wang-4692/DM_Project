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
import logging # Import logging for potential use (though removed from plot_early_stopping)

# Configure basic logging (optional, if other functions might use it)
# logger = logging.getLogger(__name__) # Define logger if needed elsewhere

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
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if legend and handles and labels:
        ax.legend(frameon=True, fancybox=True, shadow=True)
    elif legend:
         print(f"Debug: No handles/labels found for legend in plot '{title}'") # Debug print
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


# Training visualization functions
def plot_training_history(history, test_metrics=None, output_path=None, pe_type=None):
    """
    Plot training/validation accuracy and loss curves
    """
    # Check if history dict has the required keys and non-empty lists
    required_keys = ['train_acc', 'val_acc', 'train_loss', 'val_loss']
    if not history or not all(k in history and isinstance(history[k], list) and history[k] for k in required_keys):
        print(f"Warning: Skipping training history plot for {pe_type} due to missing or empty history data.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot training and validation accuracy
    ax1.plot(history['train_acc'], label='Train', color='blue', linewidth=2)
    ax1.plot(history['val_acc'], label='Validation', color='green', linewidth=2)

    if test_metrics and len(test_metrics) == 2:
        test_loss, test_acc = test_metrics
        # Ensure test_acc is valid before plotting line
        if test_acc is not None and test_acc >= 0:
             ax1.axhline(y=test_acc, color='red', linestyle='--',
                       label=f'Test ({test_acc:.4f})', linewidth=2)

    title = f'{pe_type} Accuracy' if pe_type else 'Training and Validation Accuracy'
    set_plot_style(ax1, title=title, xlabel='Epoch', ylabel='Accuracy')

    # Plot training and validation loss
    ax2.plot(history['train_loss'], label='Train', color='blue', linewidth=2)
    ax2.plot(history['val_loss'], label='Validation', color='green', linewidth=2)

    if test_metrics and len(test_metrics) == 2:
        test_loss, test_acc = test_metrics
        # Ensure test_loss is valid before plotting line
        if test_loss is not None and test_loss >= 0:
             ax2.axhline(y=test_loss, color='red', linestyle='--',
                       label=f'Test ({test_loss:.4f})', linewidth=2)

    title = f'{pe_type} Loss' if pe_type else 'Training and Validation Loss'
    set_plot_style(ax2, title=title, xlabel='Epoch', ylabel='Loss')

    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving plot {output_path}: {e}")
        plt.close(fig) # Ensure figure is closed
    else:
        plt.show()


# Optimization visualization functions
def plot_optimization_history(study, output_path=None):
    """Plot optimization history from an Optuna study"""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))

    if not trials:
        print("Warning: No completed trials found in study. Cannot plot optimization history.")
        plt.close(fig)
        return

    # Ensure values are not None before plotting
    trial_numbers = [t.number for t in trials if t.value is not None]
    trial_values = [t.value for t in trials if t.value is not None]

    if not trial_numbers:
         print("Warning: No completed trials with valid values found. Cannot plot optimization history.")
         plt.close(fig)
         return

    trials_df = pd.DataFrame({'number': trial_numbers, 'value': trial_values})
    trials_df = trials_df.sort_values('number') # Sort by trial number

    # Plot trials
    ax.plot(trials_df['number'], trials_df['value'], marker='o', linestyle='-',
            color='blue', alpha=0.7, label='Trial Value')

    # Plot running best value
    running_best = trials_df['value'].cummax()
    ax.plot(trials_df['number'], running_best, marker='', linestyle='-',
            color='red', alpha=0.9, linewidth=2, label='Best Value')

    set_plot_style(ax, title='Optimization History',
                  xlabel='Trial Number', ylabel='Objective Value (Validation Acc)', legend=True) # Added legend=True

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving plot {output_path}: {e}")
        plt.close(fig)
    else:
        plt.show()


def plot_parameter_importances(study, output_path=None):
    """Plot parameter importance from an Optuna study"""
    try:
        # Get parameter importances
        param_importances = optuna.importance.get_param_importances(study)

        if not param_importances:
             print("Warning: Could not calculate parameter importances (possibly no completed trials or dependencies missing). Skipping plot.")
             return

        # Convert to DataFrame for plotting
        importance_df = pd.DataFrame({
            'Parameter': list(param_importances.keys()),
            'Importance': list(param_importances.values())
        })

        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)

        # Plot
        fig, ax = plt.subplots(figsize=(12, max(6, len(importance_df) * 0.4))) # Adjust height
        bars = ax.barh(importance_df['Parameter'][::-1], importance_df['Importance'][::-1], # Horizontal bar plot
                      color=COLORS[0], alpha=0.8) # Use a single color

        set_plot_style(ax, title='Parameter Importances',
                      xlabel='Importance Score', ylabel='Parameter', legend=False)

        # Add labels to bars
        # for bar in bars:
        #      width = bar.get_width()
        #      ax.text(width + 0.01 * importance_df['Importance'].max(), bar.get_y() + bar.get_height()/2.,
        #              f'{width:.3f}', va='center')


        plt.tight_layout()

        if output_path:
            try:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Error saving plot {output_path}: {e}")
            plt.close(fig)
        else:
            plt.show()

    except ImportError as e:
         # Catch specific sklearn import error if needed
         print(f"Could not plot parameter importance, likely missing dependency: {e}")
    except Exception as e:
        print(f"Could not plot parameter importance: {type(e).__name__} - {e}")


def plot_param_vs_performance(study, param_name, output_path=None):
    """Plot relationship between parameter value and performance (including pruned trials)"""
    values = []
    scores = []
    trial_states = []

    for trial in study.trials:
        if param_name in trial.params:
            state = trial.state
            value = trial.params[param_name]
            score = None

            if state == optuna.trial.TrialState.COMPLETE:
                score = trial.value
                state_str = "complete"
            elif state == optuna.trial.TrialState.PRUNED:
                if trial.intermediate_values:
                    score = max(trial.intermediate_values.values())
                    state_str = "pruned"
                else: continue # Skip pruned without intermediate
            else: continue # Skip other states like RUNNING, FAIL

            if score is not None: # Ensure score is valid
                 values.append(value)
                 scores.append(score)
                 trial_states.append(state_str)


    if not values:
        print(f"No valid data available for parameter {param_name}. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    df = pd.DataFrame({'value': values, 'score': scores, 'state': trial_states})

    # Determine if parameter is categorical or numerical for plotting
    is_categorical = False
    if isinstance(values[0], str) or isinstance(values[0], bool):
        is_categorical = True
    elif isinstance(values[0], (int, float)):
         # Treat as categorical if few unique values relative to total points
         if len(df['value'].unique()) < min(10, len(df) * 0.5):
              is_categorical = True
              # Convert to string for plotting if numerical but treated as categorical
              df['value'] = df['value'].astype(str)


    if is_categorical:
        order = sorted(df['value'].unique(), key=lambda x: float(x) if x.replace('.','',1).isdigit() else x) # Sort numerically if possible
        sns.boxplot(x='value', y='score', data=df, ax=ax, palette='muted', order=order, showfliers=False)
        sns.stripplot(x='value', y='score', data=df, ax=ax,
                      hue='state', dodge=True, alpha=0.6, size=5, order=order,
                      palette={'complete': 'blue', 'pruned': 'red'})
        # Improve categorical axis labels
        try:
             ax.tick_params(axis='x', rotation=30)
        except: pass # Ignore errors if rotation fails
    else: # Numerical scatter plot
        complete_df = df[df['state'] == 'complete']
        pruned_df = df[df['state'] == 'pruned']

        ax.scatter(complete_df['value'], complete_df['score'],
                  alpha=0.6, label='Complete Trials', color='blue', s=30)
        ax.scatter(pruned_df['value'], pruned_df['score'],
                  alpha=0.6, label='Pruned Trials', color='red', marker='x', s=30)

        # Add trend line (using only complete trials)
        if len(complete_df) > 2:
            try:
                # Use lowess smoothing for potentially non-linear trends
                lowess = sm.nonparametric.lowess
                x_vals = complete_df['value'].values
                y_vals = complete_df['score'].values
                # Sort values for lowess
                sorted_indices = np.argsort(x_vals)
                x_sorted = x_vals[sorted_indices]
                y_sorted = y_vals[sorted_indices]
                smoothed = lowess(y_sorted, x_sorted, frac=0.6) # Adjust frac as needed
                ax.plot(smoothed[:, 0], smoothed[:, 1], "g--", alpha=0.8, label='Trend (LOWESS)', linewidth=2)
            except NameError: # If statsmodels not installed
                 print("Statsmodels not found, falling back to linear trend line.")
                 try:
                      z = np.polyfit(complete_df['value'], complete_df['score'], 1)
                      p = np.poly1d(z)
                      x_range = np.linspace(complete_df['value'].min(), complete_df['value'].max(), 100)
                      ax.plot(x_range, p(x_range), "b--", alpha=0.7, label='Trend (Linear)')
                 except Exception as e:
                      print(f"Could not generate linear trend line: {e}")
            except Exception as e:
                print(f"Could not generate trend line: {e}")

    # Add legend only if needed
    handles, labels = ax.get_legend_handles_labels()
    if handles:
         ax.legend()

    set_plot_style(ax, title=f'Parameter {param_name} vs Performance',
                  xlabel=param_name, ylabel='Objective Value', legend=False) # Legend handled above

    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving plot {output_path}: {e}")
        plt.close(fig)
    else:
        plt.show()


# Comparative visualization functions
def plot_accuracy_comparison(results, output_path=None):
    """
    Plot accuracy comparison between all positional encoding types
    """
    if not results: print("Warning: No results provided for accuracy comparison."); return

    data = defaultdict(list)
    for result in results:
        pe_type = result.get('pe_type')
        test_acc = result.get('test_accuracy')
        # Ensure accuracy is a valid number
        if pe_type and test_acc is not None:
            try:
                 data[pe_type].append(float(test_acc))
            except (ValueError, TypeError):
                 print(f"Warning: Invalid test_accuracy '{test_acc}' for {pe_type}. Skipping.")


    if not data: print("Warning: No valid accuracy data found for comparison."); return

    fig, ax = plt.subplots(figsize=(12, 6))

    pe_types = list(data.keys())
    mean_accuracy = [np.mean(data[pe]) for pe in pe_types]
    std_accuracy = [np.std(data[pe]) for pe in pe_types]

    sorted_indices = np.argsort(mean_accuracy)[::-1]
    pe_types = [pe_types[i] for i in sorted_indices]
    mean_accuracy = [mean_accuracy[i] for i in sorted_indices]
    std_accuracy = [std_accuracy[i] for i in sorted_indices]

    colors = [PE_TYPE_COLORS.get(pe, 'gray') for pe in pe_types]

    bars = ax.bar(pe_types, mean_accuracy, yerr=std_accuracy, capsize=5, # Reduced capsize
                  alpha=0.8, color=colors, edgecolor='black', linewidth=0.5) # Added edge color

    for bar, accuracy in zip(bars, mean_accuracy):
        ax.text(bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.005,
                f'{accuracy:.4f}',
                ha='center', va='bottom', rotation=0, fontsize=9)

    set_plot_style(ax, title='Comparison of Positional Encoding Methods (Test Accuracy)',
                  xlabel='Positional Encoding Type', ylabel='Test Accuracy', legend=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y))) # Format y-axis as percentage


    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving plot {output_path}: {e}")
        plt.close(fig)
    else:
        plt.show()


def plot_convergence_comparison(results, output_path=None):
    """
    Plot convergence curves for all positional encoding types.
    Only plots for runs where history data is available (i.e., not using checkpoints).
    """
    if not results: print("Warning: No results provided for convergence comparison."); return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True) # Share x-axis

    plotted_anything = False
    pe_types_plotted = set()

    # Group results by PE type
    grouped_results = defaultdict(list)
    for r in results:
        if r.get('pe_type'):
             grouped_results[r['pe_type']].append(r)

    pe_types = sorted(list(grouped_results.keys()))

    for pe_type in pe_types:
        # Collect curves ONLY if history is valid and contains data
        val_curves = []
        train_curves = []
        for result in grouped_results[pe_type]:
            history = result.get('history')
            # Check if history exists, is dict, has non-empty lists, and wasn't dummy
            if (isinstance(history, dict) and
                history.get('val_acc') and isinstance(history['val_acc'], list) and
                history.get('train_acc') and isinstance(history['train_acc'], list) and
                history.get('total_training_time', 0) > 0): # Check if training actually happened
                val_curves.append(history['val_acc'])
                train_curves.append(history['train_acc'])

        if not val_curves or not train_curves: # Skip if no valid history for this PE type
             print(f"Info: No valid training history found for {pe_type}. Skipping convergence plot.")
             continue

        plotted_anything = True
        pe_types_plotted.add(pe_type)

        # --- Plot Validation Accuracy ---
        max_len_val = max(len(c) for c in val_curves)
        aligned_val = [c + [c[-1]]*(max_len_val - len(c)) if len(c)>0 else [np.nan]*max_len_val for c in val_curves] # Handle empty lists
        val_mean = np.nanmean(aligned_val, axis=0) # Use nanmean
        val_std = np.nanstd(aligned_val, axis=0)   # Use nanstd
        epochs_val = np.arange(1, max_len_val + 1)
        color = PE_TYPE_COLORS.get(pe_type, None)
        ax1.plot(epochs_val, val_mean, label=pe_type, linewidth=2, color=color)
        ax1.fill_between(epochs_val, val_mean - val_std, val_mean + val_std, alpha=0.15, color=color)

        # --- Plot Training Accuracy ---
        max_len_train = max(len(c) for c in train_curves)
        aligned_train = [c + [c[-1]]*(max_len_train - len(c)) if len(c)>0 else [np.nan]*max_len_train for c in train_curves]
        train_mean = np.nanmean(aligned_train, axis=0)
        train_std = np.nanstd(aligned_train, axis=0)
        epochs_train = np.arange(1, max_len_train + 1)
        # Use same color as validation plot
        ax2.plot(epochs_train, train_mean, label=pe_type, linewidth=2, color=color)
        ax2.fill_between(epochs_train, train_mean - train_std, train_mean + train_std, alpha=0.15, color=color)


    if not plotted_anything:
        print("Warning: No valid history data found for any PE type. Cannot generate convergence plot.")
        plt.close(fig)
        return

    set_plot_style(ax1, title='Convergence Comparison (Validation Accuracy)',
                  xlabel=None, ylabel='Validation Accuracy', legend=True) # Legend on top plot
    set_plot_style(ax2, title='Convergence Comparison (Training Accuracy)',
                  xlabel='Epoch', ylabel='Training Accuracy', legend=False) # No legend on bottom

    # Improve layout and save
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving plot {output_path}: {e}")
        plt.close(fig)
    else:
        plt.show()


def plot_parameter_efficiency(results, output_path=None):
    """
    Plot parameter efficiency (accuracy vs TOTAL parameter count).
    Removed PE parameter plot.
    """
    if not results: print("Warning: No results provided for parameter efficiency plot."); return

    pe_types = []
    accuracies = []
    total_params = []

    pe_type_data = defaultdict(lambda: {'acc': [], 'total': []})

    for result in results:
        pe_type = result.get('pe_type')
        test_acc = result.get('test_accuracy')
        t_params = result.get('total_parameters')
        if pe_type and test_acc is not None and t_params is not None:
             try:
                  pe_type_data[pe_type]['acc'].append(float(test_acc))
                  pe_type_data[pe_type]['total'].append(float(t_params))
             except (ValueError, TypeError):
                  print(f"Warning: Invalid data for {pe_type} in parameter efficiency plot. Skipping.")


    for pe_type, data in pe_type_data.items():
         if data['acc'] and data['total']: # Ensure data exists for this type
              pe_types.append(pe_type)
              accuracies.append(np.mean(data['acc']))
              total_params.append(np.mean(data['total']))

    if not pe_types: print("Warning: No valid data found for parameter efficiency plot."); return

    accuracies = np.array(accuracies)
    total_params = np.array(total_params)

    # --- Create Plot ---
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7)) # Single plot now

    # Plot with total parameters
    for pe, acc, params in zip(pe_types, accuracies, total_params):
        color = PE_TYPE_COLORS.get(pe, None)
        ax1.scatter(params, acc, s=120, label=pe, color=color, edgecolor='black', alpha=0.8, linewidth=0.5)
        # Add text labels slightly offset
        ax1.text(params * 1.01, acc * 1.005, pe, fontsize=9, verticalalignment='bottom')

    set_plot_style(ax1, title='Accuracy vs. Total Parameters',
                  xlabel='Total Parameters (Millions)', ylabel='Test Accuracy', legend=True) # Add legend

    # Format x-axis to millions
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y))) # Format y-axis as percentage


    # Add reference line (optional)
    # if len(pe_types) > 1:
    #     try:
    #         z = np.polyfit(total_params, accuracies, 1)
    #         p = np.poly1d(z)
    #         x_range = np.linspace(min(total_params)*0.9, max(total_params)*1.1, 100)
    #         ax1.plot(x_range, p(x_range), "k--", alpha=0.5, linewidth=1)
    #     except: pass

    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving plot {output_path}: {e}")
        plt.close(fig)
    else:
        plt.show()


def plot_configuration_heatmap(results, output_path=None):
    """
    Plot heatmap of optimal hyperparameter configurations
    """
    if not results: print("Warning: No results provided for configuration heatmap."); return

    optimal_configs = {}
    pe_type_results = defaultdict(list)
    for result in results:
         # Ensure test accuracy is valid before considering
         test_acc = result.get('test_accuracy')
         pe_type = result.get('pe_type')
         if pe_type and test_acc is not None:
              try:
                   pe_type_results[pe_type].append(result)
              except (ValueError, TypeError):
                   pass # Skip results with invalid accuracy

    if not pe_type_results: print("Warning: No valid results found for configuration heatmap."); return

    for pe_type, type_results in pe_type_results.items():
        # Filter again for valid accuracy before finding max
        valid_results = [r for r in type_results if r.get('test_accuracy') is not None]
        if valid_results:
             best_result = max(valid_results, key=lambda x: float(x['test_accuracy']))
             if isinstance(best_result.get('config'), dict):
                  optimal_configs[pe_type] = best_result['config']
             else:
                  print(f"Warning: Best result for {pe_type} has missing or invalid config.")
        else:
             print(f"Warning: No valid results with accuracy found for {pe_type} to determine best config.")


    if not optimal_configs: print("Warning: No optimal configurations found to plot heatmap."); return

    # Select common hyperparameters to compare
    common_params = [
        'drop_rate', 'label_smoothing', 'mixup_alpha', 'weight_decay',
        'lr', 'batch_size', 'embed_dim', 'depth', 'num_heads', 'patch_size' # Added patch_size
    ]

    heatmap_data = []
    pe_order = sorted(optimal_configs.keys()) # Sort PE types for consistent row order

    for pe_type in pe_order:
        config = optimal_configs[pe_type]
        row = {'PE Type': pe_type}
        for param in common_params:
             # Use .get() for safer access
             row[param] = config.get(param)
        heatmap_data.append(row)

    df = pd.DataFrame(heatmap_data)
    df.set_index('PE Type', inplace=True)
    df = df.astype(float, errors='ignore') # Convert numeric columns to float if possible

    # Normalize columns for better color mapping (optional)
    # df_normalized = df.copy()
    # for col in df.columns:
    #     if pd.api.types.is_numeric_dtype(df[col]):
    #         min_val, max_val = df[col].min(), df[col].max()
    #         if max_val > min_val:
    #              df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    #         else:
    #              df_normalized[col] = 0.5 # Assign middle value if all are same

    fig, ax = plt.subplots(figsize=(max(12, len(common_params)*1.2), len(optimal_configs) * 0.8))

    # Create the heatmap using original values for annotation
    cmap = sns.color_palette("viridis", as_cmap=True) # Use viridis
    sns.heatmap(df.select_dtypes(include=np.number), # Heatmap only numeric types
                annot=True, cmap=cmap, cbar=True, fmt='.3g', ax=ax, linewidths=.5)

    ax.set_title('Optimal Hyperparameter Configurations (Best Test Accuracy per PE Type)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving plot {output_path}: {e}")
        plt.close(fig)
    else:
        plt.show()


def plot_overfitting_analysis(results, output_path=None):
    """
    Plot analysis of overfitting (train-val accuracy gap) for each PE type.
    Only plots for runs where history data is available.
    """
    if not results: print("Warning: No results provided for overfitting analysis."); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Overfitting Analysis', fontsize=16, y=1.02)


    pe_types = sorted(list(set(result.get('pe_type') for result in results if result.get('pe_type'))))
    overfitting_gap = []
    pe_labels_gap = []
    plotted_scatter = False

    for pe_type in pe_types:
        type_results = [r for r in results if r.get('pe_type') == pe_type]
        gaps = []
        train_accs_final = []
        val_accs_final = []

        for result in type_results:
            history = result.get('history')
            # Check if history exists and has necessary non-empty lists
            if (isinstance(history, dict) and
                history.get('train_acc') and isinstance(history['train_acc'], list) and
                history.get('val_acc') and isinstance(history['val_acc'], list)):
                try:
                    final_train = float(history['train_acc'][-1])
                    final_val = float(history['val_acc'][-1])
                    gap = final_train - final_val
                    gaps.append(gap)
                    train_accs_final.append(final_train)
                    val_accs_final.append(final_val)
                except (IndexError, ValueError, TypeError):
                    continue # Skip if lists are empty or values invalid

        if gaps: # If we calculated any gaps for this PE type
            mean_gap = np.mean(gaps)
            overfitting_gap.append(mean_gap)
            pe_labels_gap.append(pe_type)

        if train_accs_final and val_accs_final: # If we have data for scatter plot
            color = PE_TYPE_COLORS.get(pe_type, None)
            ax2.scatter(train_accs_final, val_accs_final, label=pe_type, color=color,
                       s=60, alpha=0.7, edgecolor='w', linewidth=0.5)
            plotted_scatter = True


    # --- Plot Overfitting Gap Bar Chart ---
    if pe_labels_gap:
        sorted_indices = np.argsort(overfitting_gap)
        sorted_gaps = [overfitting_gap[i] for i in sorted_indices]
        sorted_labels = [pe_labels_gap[i] for i in sorted_indices]
        colors = [PE_TYPE_COLORS.get(pe, 'gray') for pe in sorted_labels]
        bars = ax1.bar(sorted_labels, sorted_gaps, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        for bar, gap in zip(bars, sorted_gaps):
            ax1.text(bar.get_x() + bar.get_width()/2.,
                    gap + 0.002 if gap >= 0 else gap - 0.008, # Adjust vertical offset
                    f'{gap:.3f}',
                    ha='center', va='bottom' if gap >= 0 else 'top',
                    rotation=0, fontsize=9)

        set_plot_style(ax1, title='Final Epoch Overfitting Gap (Train - Val Acc)',
                      xlabel='Positional Encoding Type', ylabel='Accuracy Gap',
                      legend=False)
        ax1.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Add zero line
    else:
        ax1.text(0.5, 0.5, "No history data available\nfor overfitting gap plot.",
                 ha='center', va='center', transform=ax1.transAxes)
        set_plot_style(ax1, title='Final Epoch Overfitting Gap', xlabel='Positional Encoding Type', ylabel='Accuracy Gap', legend=False)


    # --- Plot Train vs Val Accuracy Scatter Plot ---
    if plotted_scatter:
        min_val = min([ax2.get_xlim()[0], ax2.get_ylim()[0], 0.6]) # Adjust min value if needed
        max_val = max([ax2.get_xlim()[1], ax2.get_ylim()[1], 1.0])
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4, linewidth=1, label='y=x')
        set_plot_style(ax2, title='Final Epoch Train vs. Validation Accuracy',
                      xlabel='Train Accuracy', ylabel='Validation Accuracy',
                      legend=True)
        ax2.set_xlim(left=min_val, right=max_val)
        ax2.set_ylim(bottom=min_val, top=max_val)
        ax2.set_aspect('equal', adjustable='box') # Make axes equal
    else:
        ax2.text(0.5, 0.5, "No history data available\nfor train vs val plot.",
                 ha='center', va='center', transform=ax2.transAxes)
        set_plot_style(ax2, title='Final Epoch Train vs. Validation Accuracy', xlabel='Train Accuracy', ylabel='Validation Accuracy', legend=False)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving plot {output_path}: {e}")
        plt.close(fig)
    else:
        plt.show()


def plot_slice(study, param_name, output_path=None):
    """
    Create a slice plot for a parameter showing its effect on performance
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    param_values = []
    objective_values = []

    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        if param_name in trial.params and trial.value is not None:
            param_values.append(trial.params[param_name])
            objective_values.append(trial.value)

    if not param_values:
        print(f"Warning: No valid data for slice plot of '{param_name}'. Skipping.")
        plt.close(fig)
        return

    # Sort data by parameter value for line plot if numerical
    is_numeric = all(isinstance(x, (int, float)) for x in param_values)
    if is_numeric:
         sorted_indices = np.argsort(param_values)
         sorted_param_values = np.array(param_values)[sorted_indices]
         sorted_objectives = np.array(objective_values)[sorted_indices]
         ax.plot(sorted_param_values, sorted_objectives, 'o-', color='blue', alpha=0.7, markersize=5)
         xlabel = param_name
    else:
         # For categorical, just plot as is (order might not be meaningful)
         ax.plot(param_values, objective_values, 'o', color='blue', alpha=0.7, markersize=5)
         xlabel = f"{param_name} (Categorical)"
         try:
              plt.xticks(rotation=30, ha='right')
         except: pass


    set_plot_style(ax, title=f'Parameter Slice: {param_name}',
                  xlabel=xlabel, ylabel='Objective Value', legend=False)

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving plot {output_path}: {e}")
        plt.close(fig)
    else:
        plt.show()


# Add this function definition to your utils/visualization.py file
def plot_early_stopping_comparison(results, output_path=None):
    """
    Plot early stopping patterns (rate, best epoch, gap) for different PEs.
    Only plots for runs where history data is available.
    """
    if not results: print("Warning: No results provided for early stopping comparison."); return

    pe_types = sorted(list(set(result.get('pe_type') for result in results if result.get('pe_type'))))
    if not pe_types:
        print("No PE types found in results for early stopping comparison.")
        return

    early_stopping_rates = []
    best_epochs = []
    train_val_gaps = []
    pe_labels_present = [] # Keep track of PEs with valid data

    for pe_type in pe_types:
        type_results = [r for r in results if r.get('pe_type') == pe_type]
        histories = [r.get('history') for r in type_results if isinstance(r.get('history'), dict)]

        # Filter for histories that indicate actual training occurred
        valid_histories = [h for h in histories if h.get('total_training_time', 0) > 0 and h.get('val_acc')]

        if not valid_histories:
            print(f"Info: No valid training history found for {pe_type}. Skipping early stopping plot.")
            continue # Skip this PE type if no valid history

        pe_labels_present.append(pe_type) # Add PE type if we have data

        # Calculate early stopping rate
        early_stopped_flags = [h.get('early_stopped', False) for h in valid_histories if h.get('early_stopped') is not None]
        early_stopping_rates.append(np.mean(early_stopped_flags) if early_stopped_flags else 0)

        # Calculate average best epoch
        best_epoch_values = [h.get('best_epoch') for h in valid_histories if h.get('best_epoch') is not None]
        # Add 1 to best_epoch because they are 0-indexed
        best_epochs.append(np.mean([e + 1 for e in best_epoch_values]) if best_epoch_values else np.nan)

        # Calculate average train-val gap at the *best* epoch if possible
        gap_values = []
        for h in valid_histories:
             tv_gap_list = h.get('train_val_gap')
             best_ep = h.get('best_epoch')
             if isinstance(tv_gap_list, list) and len(tv_gap_list) > 0:
                  if best_ep is not None and 0 <= best_ep < len(tv_gap_list):
                       gap_values.append(tv_gap_list[best_ep])
                  else:
                       # Fallback: gap at final epoch if best_epoch invalid
                       gap_values.append(tv_gap_list[-1])

        train_val_gaps.append(np.mean(gap_values) if gap_values else np.nan)

    # Check if any PE type had valid data to plot
    if not pe_labels_present:
         print("Warning: No PE types with valid history data found. Cannot generate early stopping comparison plot.")
         return

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Early Stopping Behavior Comparison (Based on Retraining Runs)', fontsize=16, y=1.02)


    # Plot early stopping rates
    colors = [PE_TYPE_COLORS.get(pe, 'gray') for pe in pe_labels_present]
    ax1.bar(pe_labels_present, early_stopping_rates, color=colors, alpha=0.7, label='_nolegend_')
    for i, v in enumerate(early_stopping_rates):
        ax1.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', va='bottom')
    set_plot_style(ax1, 'Early Stopping Rate', 'Positional Encoding', 'Rate (%)', legend=False)
    ax1.set_ylim(0, max(1.1, max(early_stopping_rates)*1.1) if early_stopping_rates else 1.1) # Adjust ylim dynamically
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))


    # Plot average best epoch
    ax2.bar(pe_labels_present, best_epochs, color=colors, alpha=0.7, label='_nolegend_')
    for i, v in enumerate(best_epochs):
        if pd.notna(v):
             ax2.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
    set_plot_style(ax2, 'Average Best Epoch', 'Positional Encoding', 'Epoch Number', legend=False)

    # Plot average train-val gap
    ax3.bar(pe_labels_present, train_val_gaps, color=colors, alpha=0.7, label='_nolegend_')
    for i, v in enumerate(train_val_gaps):
         if pd.notna(v):
              ax3.text(i, v + 0.005 if v >= 0 else v - 0.01, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top')
    set_plot_style(ax3, 'Avg Train-Val Gap (at Best Epoch)', 'Positional Encoding', 'Accuracy Gap', legend=False)
    ax3.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Add zero line


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            # Removed logger call as it's not defined here
            # print(f"Info: Saved early stopping comparison plot to {output_path}") # Optional print
        except Exception as e:
             print(f"Error saving plot {output_path}: {e}")
        plt.close(fig)
    else:
        plt.show()

# --- Add statsmodels import for LOWESS smoothing ---
try:
    import statsmodels.api as sm
except ImportError:
    print("Warning: statsmodels not installed. Trend lines in parameter vs performance plots will be linear.")
    sm = None # Set sm to None if import fails
# --- End statsmodels import ---
