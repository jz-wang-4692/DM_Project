import optuna
import sys
import os
import logging
import datetime
from collections import defaultdict

# Configure basic logging for Optuna messages
logging.basicConfig(level=logging.INFO)

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_study.py <path_to_optuna_db_file>")
        print("Example: python check_study.py results/bo_results/ape/optuna.db")
        sys.exit(1)
    
    db_path = sys.argv[1]
    storage_url = f"sqlite:///{db_path}"
    
    # --- Determine Study Name (Adjust if your naming convention differs) ---
    try:
        pe_type = os.path.basename(os.path.dirname(db_path))
        if not pe_type: # Handle case where path ends with '/'
             pe_type = os.path.basename(os.path.dirname(os.path.dirname(db_path)))
        study_name = f"{pe_type}_optimization"
        print(f"Derived study name: '{study_name}' based on path.")
    except Exception as e:
        print(f"Could not automatically determine study name from path: {e}")
        print("Please ensure the database path is correct.")
        sys.exit(1)
    # --- End Study Name Determination ---
    
    print(f"\nAttempting to load study '{study_name}' from storage '{storage_url}'...")
    
    try:
        # Use load_study to check if it exists and load it
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        print(f"\n--- Study Found ---")
        print(f"Study Name: {study.study_name}")
        
        # 1. Direction information - handle numeric value
        print(f"Direction: {study.direction}")
        
        # Initialize multi_obj_names here
        multi_obj_names = []
        
        # Get direction information
        if hasattr(study, "_directions") and study._directions:
            print("Multi-Objective Optimization Study")
            print("Optimizing multiple metrics:")
            for i, direction in enumerate(study._directions):
                if direction == optuna.study.StudyDirection.MAXIMIZE:
                    print(f"  - Objective {i+1}: MAXIMIZE")
                else:
                    print(f"  - Objective {i+1}: MINIMIZE")
        elif str(study.direction) == "1":
            print("Direction explained: MAXIMIZE (trying to find the highest value)")
        elif str(study.direction) == "0":
            print("Direction explained: MINIMIZE (trying to find the lowest value)")
        elif str(study.direction) == "2":
            # Try to determine objective names by checking trial values
            trials = study.get_trials(deepcopy=False)
            for trial in trials:
                if hasattr(trial, "values") and trial.values:
                    print("Multi-Objective Optimization Study")
                    print(f"Number of objectives: {len(trial.values)}")
                    
                    # Try to infer objective names from user attributes
                    if trial.user_attrs and "objective_names" in trial.user_attrs:
                        obj_names = trial.user_attrs["objective_names"]
                        print("Objective metrics:")
                        for i, name in enumerate(obj_names):
                            print(f"  - Objective {i+1}: {name}")
                        multi_obj_names = obj_names
                    else:
                        print("Objective values from most recent completed trial:")
                        for i, val in enumerate(trial.values):
                            print(f"  - Objective {i+1}: {val:.6f}")
                    break
            
            if not multi_obj_names:
                print("Direction explained: This is a multi-objective study (MINIMIZE_AND_MAXIMIZE)")
                print("Multi-objective studies optimize for multiple metrics simultaneously")
        else:
            print(f"Direction: Unknown ({study.direction})")
        
        # Get trial data
        trials = study.get_trials(deepcopy=False)
        num_trials_in_db = len(trials)
        print(f"Total trials recorded in DB: {num_trials_in_db}")
        
        # Count trials by state
        states = {}
        for state in optuna.trial.TrialState:
            states[state.name] = 0
        for trial in trials:
             states[trial.state.name] += 1
        
        print("\nTrial States Breakdown:")
        for state_name, count in states.items():
            if count > 0:
                print(f"  - {state_name}: {count}")
        
        # Look for total planned trials in study user attributes
        user_attrs = study.user_attrs
        total_planned = None
        
        # Check common attribute names for n_trials
        for attr in ["n_trials", "num_trials", "total_trials", "n_iter", "max_trials"]:
            if attr in user_attrs:
                total_planned = user_attrs[attr]
                print(f"\nTotal planned trials: {total_planned}")
                if total_planned > 0:
                    progress = (states.get("COMPLETE", 0) / total_planned) * 100
                    print(f"Current progress: {progress:.1f}% ({states.get('COMPLETE', 0)}/{total_planned})")
                break
        
        # Try to extract from source code files in the directory
        if total_planned is None:
            try:
                # Look for Python files in the same directory
                directory = os.path.dirname(db_path)
                py_files = [f for f in os.listdir(directory) if f.endswith('.py')]
                
                for py_file in py_files:
                    with open(os.path.join(directory, py_file), 'r') as f:
                        content = f.read()
                        # Look for common patterns for setting n_trials
                        import re
                        patterns = [
                            r'n_trials\s*=\s*(\d+)',
                            r'num_trials\s*=\s*(\d+)',
                            r'total_trials\s*=\s*(\d+)',
                            r'study\.optimize\(.*?,\s*n_trials=(\d+)',
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, content)
                            if match:
                                total_planned = int(match.group(1))
                                print(f"\nTotal planned trials (from source code): {total_planned}")
                                if total_planned > 0:
                                    progress = (states.get("COMPLETE", 0) / total_planned) * 100
                                    print(f"Current progress: {progress:.1f}% ({states.get('COMPLETE', 0)}/{total_planned})")
                                break
                        
                        if total_planned:
                            break
            except Exception as e:
                # If we can't read files, just continue
                pass
        
        # If still not found, try to infer from SQL directly
        if total_planned is None:
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                # Look for setting table with n_trials
                cursor.execute("SELECT value FROM study_system_attr WHERE key LIKE '%n_trial%' OR key LIKE '%num_trial%'")
                result = cursor.fetchone()
                if result:
                    total_planned = int(result[0])
                    print(f"\nTotal planned trials (from database): {total_planned}")
                    if total_planned > 0:
                        progress = (states.get("COMPLETE", 0) / total_planned) * 100
                        print(f"Current progress: {progress:.1f}% ({states.get('COMPLETE', 0)}/{total_planned})")
                conn.close()
            except:
                pass

        # Try to estimate based on highest trial number
        if total_planned is None and trials:
            highest_number = max(t.number for t in trials)
            if highest_number > 0:
                # Estimate total as approximately 20% more than current highest
                estimated_total = int(highest_number * 1.2)
                print(f"\nEstimated total trials: ~{estimated_total} (based on current progress)")
            else:
                print("\nTotal planned trials: Unknown (not found in study metadata)")
        else:
            print("\nTotal planned trials: Unknown (not found in study metadata)")
            
        # 2. Study Configuration
        print("\nStudy Configuration:")
        print(f"Sampler: {study.sampler.__class__.__name__}")
        
        # Get sampler info
        sampler_attrs = vars(study.sampler)
        important_attrs = ['n_startup_trials', 'gamma', 'consider_prior', 'seed', 
                          'n_ei_candidates', 'multivariate', 'warn_independent_sampling']
        
        for attr in important_attrs:
            if attr in sampler_attrs:
                print(f"  - {attr}: {sampler_attrs[attr]}")
                
        # Try to guess number of trials from pruner if available
        if hasattr(study, "pruner") and study.pruner is not None:
            print(f"Pruner: {study.pruner.__class__.__name__}")
            pruner_attrs = vars(study.pruner)
            for attr in ['n_startup_trials', 'n_warmup_steps', 'interval_steps']:
                if attr in pruner_attrs:
                    print(f"  - {attr}: {pruner_attrs[attr]}")
            
        # Extract parameter space from existing trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            print("\nParameter Space (from completed trials):")
            # Collect all parameters across trials
            all_params = defaultdict(list)
            for trial in completed_trials:
                for param_name, param_value in trial.params.items():
                    all_params[param_name].append(param_value)
            
            # Print parameter ranges
            for param_name, values in all_params.items():
                if isinstance(values[0], (int, float)):
                    print(f"  - {param_name}: range [{min(values)}, {max(values)}]")
                else:
                    unique_values = set(values)
                    print(f"  - {param_name}: categorical {unique_values}")
        
        # 3. Broader study information
        print("\nStudy Statistics:")
        if completed_trials:
            # For multi-objective studies
            is_multi_obj = hasattr(study, "_directions") or str(study.direction) == "2"
            
            if is_multi_obj:
                print("  - Multi-objective study detected")
                # Try to analyze all objectives
                multi_obj_trial = None
                for trial in completed_trials:
                    if hasattr(trial, "values") and trial.values:
                        multi_obj_trial = trial
                        break
                
                if multi_obj_trial and hasattr(multi_obj_trial, "values"):
                    num_objectives = len(multi_obj_trial.values)
                    
                    # Collect all values for each objective
                    all_values = [[] for _ in range(num_objectives)]
                    for trial in completed_trials:
                        if hasattr(trial, "values") and trial.values and len(trial.values) == num_objectives:
                            for i, val in enumerate(trial.values):
                                if val is not None:
                                    all_values[i].append(val)
                    
                    # Print stats for each objective
                    for i, values in enumerate(all_values):
                        if values:
                            obj_name = f"Objective {i+1}"
                            if multi_obj_names and i < len(multi_obj_names):
                                obj_name = multi_obj_names[i]
                            print(f"  - {obj_name} stats: min={min(values):.6f}, max={max(values):.6f}, mean={sum(values)/len(values):.6f}")
                else:
                    # Fallback to single objective analysis
                    values = [t.value for t in completed_trials if t.value is not None]
                    if values:
                        print(f"  - Primary objective stats: min={min(values):.6f}, max={max(values):.6f}, mean={sum(values)/len(values):.6f}")
            else:
                values = [t.value for t in completed_trials if t.value is not None]
                if values:
                    print(f"  - Performance stats: min={min(values):.6f}, max={max(values):.6f}, mean={sum(values)/len(values):.6f}")
                    
                    # Distribution quartiles
                    values.sort()
                    q1_idx = len(values) // 4
                    q2_idx = len(values) // 2
                    q3_idx = 3 * len(values) // 4
                    print(f"  - Quartiles: Q1={values[q1_idx]:.6f}, Q2(median)={values[q2_idx]:.6f}, Q3={values[q3_idx]:.6f}")
            
            # Calculate runtime statistics
            trials_with_complete_time = [t for t in completed_trials if t.datetime_start and t.datetime_complete]
            if trials_with_complete_time:
                durations = [(t.datetime_complete - t.datetime_start).total_seconds() for t in trials_with_complete_time]
                avg_duration = sum(durations) / len(durations)
                print(f"  - Average trial duration: {avg_duration:.2f} seconds ({datetime.timedelta(seconds=int(avg_duration))})")
                
                if len(trials_with_complete_time) > 1:
                    start_time = min(t.datetime_start for t in trials_with_complete_time)
                    end_time = max(t.datetime_complete for t in trials_with_complete_time)
                    total_duration = (end_time - start_time).total_seconds()
                    print(f"  - Total study duration so far: {datetime.timedelta(seconds=int(total_duration))}")
                    
                    # Calculate completion rate
                    trials_per_hour = (len(trials_with_complete_time) / total_duration) * 3600
                    print(f"  - Average completion rate: {trials_per_hour:.2f} trials/hour")
                    
                    # Calculate convergence velocity
                    if len(completed_trials) >= 5:
                        sorted_trials = sorted(completed_trials, key=lambda t: t.number)
                        chunk_size = max(len(sorted_trials) // 3, 1)
                        early_trials = [t for t in sorted_trials[:chunk_size] if t.value is not None]
                        late_trials = [t for t in sorted_trials[-chunk_size:] if t.value is not None]
                        
                        if early_trials and late_trials:
                            early_avg = sum(t.value for t in early_trials) / len(early_trials)
                            late_avg = sum(t.value for t in late_trials) / len(late_trials)
                            
                            print(f"  - Early trials avg: {early_avg:.6f}, Recent trials avg: {late_avg:.6f}")
                            
                            if not is_multi_obj:
                                if str(study.direction) == "1":  # MAXIMIZE
                                    improvement = late_avg - early_avg
                                    print(f"  - Convergence trend: {improvement:.6f} improvement over time")
                                elif str(study.direction) == "0":  # MINIMIZE
                                    improvement = early_avg - late_avg
                                    print(f"  - Convergence trend: {improvement:.6f} improvement over time")
        
        # Get study user attributes (metadata)
        if user_attrs:
            print("\nStudy Metadata:")
            for key, value in user_attrs.items():
                print(f"  - {key}: {value}")
        
        # Best Trial Information
        if study.best_trial:
            print(f"\nBest trial found so far:")
            print(f"  Number: {study.best_trial.number}")
            
            # For multi-objective studies
            if hasattr(study, "_directions") or str(study.direction) == "2":
                if hasattr(study.best_trial, "values") and study.best_trial.values:
                    print(f"  Values (Multiple Objectives):")
                    for i, value in enumerate(study.best_trial.values):
                        obj_name = f"Objective {i+1}"
                        if multi_obj_names and i < len(multi_obj_names):
                            obj_name = multi_obj_names[i]
                        print(f"    - {obj_name}: {value:.6f}")
                else:
                    print(f"  Value (Primary Objective): {study.best_trial.value:.6f}")
            else:
                print(f"  Value (Objective): {study.best_trial.value:.6f}")
                
            print(f"  Parameters:")
            for param_name, param_value in study.best_trial.params.items():
                print(f"    - {param_name}: {param_value}")
            
            # Show best trial datetime if available
            if study.best_trial.datetime_start:
                print(f"  Started: {study.best_trial.datetime_start}")
            if study.best_trial.datetime_complete:
                print(f"  Completed: {study.best_trial.datetime_complete}")
                
            # Check for user attributes in the best trial
            if study.best_trial.user_attrs:
                print(f"  Additional metrics:")
                for key, value in study.best_trial.user_attrs.items():
                    if key != "objective_names":  # Skip if it's already been displayed
                        print(f"    - {key}: {value}")
        else:
            print("\nNo trials completed successfully yet.")
        
        # Estimate remaining trials
        running_count = states.get("RUNNING", 0)
        if running_count > 0 and 'trials_with_complete_time' in locals() and trials_with_complete_time:
            est_remaining_time = avg_duration * running_count
            print(f"\nEstimated time to complete {running_count} running trials: {datetime.timedelta(seconds=int(est_remaining_time))}")
        
        # Calculate remaining trials if total is known
        if total_planned:
            completed_count = states.get("COMPLETE", 0)
            running_count = states.get("RUNNING", 0)
            remaining = total_planned - completed_count - running_count
            
            if remaining > 0 and 'trials_with_complete_time' in locals() and trials_with_complete_time:
                est_remaining_time = avg_duration * remaining
                print(f"Estimated additional trials to run: {remaining}")
                print(f"Estimated time for remaining trials: {datetime.timedelta(seconds=int(est_remaining_time))}")
                
                # Estimate completion time
                est_completion = datetime.datetime.now() + datetime.timedelta(seconds=est_remaining_time)
                print(f"Estimated completion date: {est_completion}")
        
        # Check for visualization capabilities
        try:
            import matplotlib
            has_matplotlib = True
        except ImportError:
            has_matplotlib = False
        
        if has_matplotlib and len(completed_trials) > 1:
            print("\nVisualization capabilities detected!")
            print("You can visualize the study using:")
            print("```python")
            print("import optuna")
            print(f"study = optuna.load_study(study_name='{study_name}', storage='{storage_url}')")
            print("optuna.visualization.matplotlib.plot_optimization_history(study)")
            print("optuna.visualization.matplotlib.plot_param_importances(study)")
            print("optuna.visualization.matplotlib.plot_parallel_coordinate(study)")
            if hasattr(study, "_directions") or str(study.direction) == "2":
                print("# For multi-objective studies:")
                print("optuna.visualization.matplotlib.plot_pareto_front(study)")
            print("```")
        elif len(completed_trials) > 1:
            print("\nFor visualizations, install matplotlib: pip install matplotlib")
            
        print("\nConclusion: Study exists and can be resumed.")
    except KeyError:
        # This error specifically means the study *name* wasn't found in the DB
        print(f"\n--- Study Not Found ---")
        print(f"Error: Study '{study_name}' does not exist in the database '{db_path}'.")
        print("Running the main script will create a new study.")
    except Exception as e:
        print(f"\n--- Error Loading Study ---")
        print(f"An unexpected error occurred: {e}")
        print("Check the database file integrity or path.")

if __name__ == "__main__":
    main()