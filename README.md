# ViT Positional Encoding Comparison with Bayesian Optimization

This project investigates the performance impact of different positional encoding (PE) methods within a standard Vision Transformer (ViT) architecture, specifically trained and evaluated on the CIFAR-10 dataset. A key focus is the use of Bayesian Optimization (BO) via Optuna to rigorously tune hyperparameters for each PE method, aiming for both high accuracy and effective regularization against overfitting.

## Authors

Chi Han, Jiazhen Wang, Ningping Wang, Lian Zhong

## Positional Encoding Methods Compared

This project implements and compares the following PE techniques:

1.  **Absolute Positional Encoding (APE):** Standard learnable embeddings added to patch embeddings.
2.  **Relative Positional Encoding (RPE):** Standard relative attention bias with learnable parameters based on relative distances.
3.  **Polynomial Relative Positional Encoding (Poly-RPE):** A variant using polynomial functions of relative L1 distances to compute attention bias, potentially reducing parameters compared to standard RPE.
4.  **Rotary Positional Encoding (RoPE - Axial):** Applies rotary embeddings along spatial axes independently, adapted from NLP techniques. Based on Heo et al., 2024.
5.  **Rotary Positional Encoding (RoPE - Mixed):** Mixes axial and 2D rotary embeddings, adapted from Heo et al., 2024.

## Acknowledgements

This project adapts and utilizes concepts or code from the following sources:

* **RoPE-ViT:** The implementation and concepts for RoPE variants are adapted from the ECCV 2024 paper and repository by Heo et al.
    ```bibtex
    @inproceedings{heo2024ropevit,
        title={Rotary Position Embedding for Vision Transformer},
        author={Heo, Byeongho and Park, Song and Han, Dongyoon and Yun, Sangdoo},
        year={2024},
        booktitle={European Conference on Computer Vision (ECCV)},
    }
    ```
    * Repository (Apache-2.0): [https://github.com/naver-ai/rope-vit](https://github.com/naver-ai/rope-vit) (Included partially in the `rope-vit` directory).
* **CodeLlama:** Inspiration or utility code potentially adapted.
    * Repository (Llama 2 Community License): [https://github.com/meta-llama/codellama](https://github.com/meta-llama/codellama)
* **Optuna:** Used for Bayesian Optimization. [https://optuna.org/](https://optuna.org/)
* **PyTorch & Timm:** Used for model implementation and training. [https://pytorch.org/](https://pytorch.org/), [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)

## Project Structure

```
.
├── BO.sh                     # Top-level script to potentially run the entire BO workflow (may call scripts/run_bo.py)
├── README.md                 # This documentation file
├── config                    # Configuration files and scripts
│   ├── best_configs          # Stores the best hyperparameter configurations found by BO for each PE type
│   │   ├── ape/config.json     # Best config for APE
│   │   ├── polynomial_rpe/config.json # Best config for Polynomial RPE
│   │   ├── rope_axial/config.json # Best config for RoPE-Axial
│   │   ├── rope_mixed/config.json # Best config for RoPE-Mixed
│   │   └── rpe/config.json      # Best config for RPE
│   ├── default_config.py     # Defines default hyperparameters, handles config loading/saving (JSON/YAML), and arg parsing
│   └── search_spaces.py      # Defines hyperparameter search ranges and distributions for Optuna Bayesian Optimization
├── main.py                   # Core script to train/evaluate a *single* ViT model instance with a given configuration (called by BO)
├── models                    # Model implementations
│   ├── model_factory.py      # Crucial factory function ('create_vit_model') to instantiate ViT models with specified PE types
│   ├── positional_encoding   # Implementations of different positional encoding methods
│   │   ├── Poly_RPE.py         # Implementation of the Polynomial Relative Positional Encoding logic
│   │   ├── RPE.py              # Implementation of the standard Relative Positional Encoding logic
│   │   ├── __init__.py         # Makes the directory a Python package
│   │   └── fixed_rope_mixed.py # Implementation for RoPE (likely Axial and Mixed variants) logic
│   └── vit_base.py           # Contains the base ViT model architecture (e.g., Attention blocks, MLP layers) that incorporates the different PE methods
├── env_setup                         # Lists Python package dependencies
│   ├── conda_environment.yml   # for conda
│   └── pip_requirements.txt    # for pip      
├── results                   # Default directory for storing all experiment outputs (created during runtime)
│   ├── best_configs          # Mirrors config/best_configs, often saved here by BO script. Also stores best checkpoints and metadata.
│   │   ├── ape/...             # Best config.json, best_checkpoint.pth, best_model_info.json for APE
│   │   ├── polynomial_rpe/...  # ...for Polynomial RPE
│   │   ├── rope_axial/...      # ...for RoPE-Axial
│   │   ├── rope_mixed/...      # ...for RoPE-Mixed
│   │   └── rpe/...             # ...for RPE
│   ├── bo_results            # Detailed results from the Bayesian Optimization runs
│   │   ├── ape/                # BO results specific to APE
│   │   │   ├── optuna.db         # Optuna study database (SQLite)
│   │   │   ├── trial_XXX/...     # Subdirectories for each trial run during BO
│   │   │   ├── *.png             # Analysis plots (optimization history, importance, etc.)
│   │   │   └── *.json            # Analysis summaries (experiment summary, sensitivity, etc.)
│   │   ├── polynomial_rpe/...  # BO results specific to Polynomial RPE
│   │   ├── rope_axial/...      # BO results specific to RoPE-Axial
│   │   ├── rope_mixed/...      # BO results specific to RoPE-Mixed
│   │   └── rpe/...             # BO results specific to RPE
│   └── final_models          # Results from the final comparison run
│       ├── ape/                # Final comparison results specific to APE
│       │   ├── seed_42/...       # Results for a specific seed
│       │   │   ├── config.json     # Config used for this run
│       │   │   ├── model_final.pth # Final model weights for this run
│       │   │   ├── results.json    # Detailed metrics for this run
│       │   │   └── training_history.png # Plot if model was retrained
│       │   ├── seed_43/...       # ... for another seed ...
│       │   └── seed_44/...       # ... for another seed ...
│       ├── polynomial_rpe/...  # Final comparison results specific to Polynomial RPE
│       ├── rope_axial/...      # Final comparison results specific to RoPE-Axial
│       ├── rope_mixed/...      # Final comparison results specific to RoPE-Mixed
│       ├── rpe/...             # Final comparison results specific to RPE
│       ├── logs/               # Log files for the comparison runs
│       ├── *.png               # Final comparison plots (accuracy, convergence, etc.)
│       ├── summary_table.csv   # Summary results in CSV format
│       └── summary_table.txt   # Summary results in plain text format
├── rope-vit                  # Cloned/adapted code from the external RoPE-ViT repository (Heo et al., 2024). Contains original implementations and potentially other related models/utilities (e.g., Swin). Consult its specific README and licenses.
│   ├── ...                   # Original files from repo...
├── scripts                   # Execution scripts for running experiments and analysis
│   ├── BO_manual.md          # (User-provided notes/manual for BO)
│   ├── analyze_bo_results.py # Utility to load a completed/interrupted Optuna study and generate analysis plots/summaries
│   ├── bo_main.py            # Main script driving the Optuna BO for a *single* PE type
│   ├── check_study.py        # Utility script to check the status of an Optuna study database
│   ├── run_bo.py             # Wrapper script to run `bo_main.py` for *multiple* PE types sequentially
│   ├── run_comparison.sh     # Bash script to execute the final comparison (`run_final_comparison.py`)
│   └── run_final_comparison.py # Python script to run final evaluation/comparison using best configs from BO
├── training                  # Training utilities
│   └── trainer.py            # Contains the core training loop (`train_model`), single epoch logic (`train_one_epoch`), evaluation (`evaluate`), and includes features like mixup, gradient clipping, early stopping, and checkpointing.
└── utils                     # Utility functions
    ├── data.py               # Handles CIFAR-10 dataset loading, transformations (augmentation), train/validation splitting, and DataLoader creation.
    └── visualization.py      # Centralized plotting functions for training history, BO results, and final comparison metrics using Matplotlib/Seaborn.
```

## Dataset

This project uses the **CIFAR-10** dataset. The `utils/data.py` script will automatically download the dataset to a `./data/` directory (not included in the repo) if it's not found locally.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jz-wang-4692/DM_Project.git
    cd DM_Project
    ```
2.  **Create Environment:** It's recommended to use a virtual environment (e.g., Conda or venv).
    ```bash
    # Using Conda (example)
    conda create -n vit_pe python=3.10
    conda activate vit_pe
    ```
3.  **Install Dependencies:** Install the required Python packages listed in files under `env_setup`.
    ```bash
    # If using Conda primarily
    conda env create -f env_setup/conda_environment.yml

    # Or if using pip primarily
    pip install -r env_setup/pip_requirements.txt
    ```
    Ensure you have PyTorch installed with the correct CUDA version. Visit [pytorch.org](https://pytorch.org/) for specific installation instructions.

## Usage

The typical workflow involves two main stages: Bayesian Optimization and Final Comparison.

### 1. Bayesian Optimization

This stage finds the optimal hyperparameters for *each* positional encoding method independently.

* **Configuration:**
    * Review and adjust default parameters in `config/default_config.py`.
    * Review and adjust the hyperparameter search space in `config/search_spaces.py`.
* **Execution:**
    * Use the wrapper script `scripts/run_bo.py` to run BO for all default PE types:
        ```bash
        # Run from the project root directory
        python scripts/run_bo.py --n_trials 50 --output_dir ./results
        ```
        (Adjust `--n_trials` as needed).
    * Alternatively, use the `BO.sh` script if it's configured to call `run_bo.py`.
    * This will execute `scripts/bo_main.py` for each PE type. Each `bo_main.py` run creates an Optuna study (`optuna.db`), runs multiple trials by calling `main.py` as a subprocess, saves detailed trial results under `results/bo_results/<pe_type>/trial_XXX/`, and generates analysis plots/summaries for that PE type in `results/bo_results/<pe_type>/`. It also saves the best configuration found to `results/best_configs/<pe_type>/`.
* **Resuming:** If the BO process is interrupted, simply re-run the same `python scripts/run_bo.py ...` command. It will load the existing Optuna studies from the `.db` files and continue where it left off.
* **Monitoring/Analysis:**
    * Use `scripts/check_study.py` to check the status of a specific study database:
        ```bash
        python scripts/check_study.py results/bo_results/ape/optuna.db
        ```
    * Use `scripts/analyze_bo_results.py` to generate the analysis plots and summaries for a completed (or interrupted) study *without* rerunning optimization:
        ```bash
        python scripts/analyze_bo_results.py --pe_type ape --output_dir ./results
        ```

### 2. Final Comparison

This stage uses the best hyperparameters found during BO to train (or load) final models and compare their performance.

* **Prerequisites:** The Bayesian Optimization stage must have completed, and the best configurations must be present in `results/best_configs/`.
* **Configuration:** Edit the configuration variables at the top of the `scripts/run_comparison.sh` script:
    * `RESULTS_DIR`: Should point to the directory containing `best_configs`.
    * `FINAL_OUTPUT_DIR`: Where to save results of this stage.
    * `SEEDS`: Space-separated list of random seeds for multiple runs.
    * `USE_CHECKPOINTS_FLAG`: Set to `"--use_checkpoints"` to load saved checkpoints, or `""` (empty string) to retrain models from scratch using the best configs.
    * `PE_TYPES_TO_COMPARE`: Optionally list specific PE types (space-separated) or leave empty to compare all found in `best_configs`.
* **Execution:**
    * Run the bash script from the project root directory:
        ```bash
        chmod +x scripts/run_comparison.sh
        ./scripts/run_comparison.sh
        ```
    * This script executes `scripts/run_final_comparison.py`. For each specified PE type and seed, it loads the best config, either loads the best checkpoint or retrains the model (`training/trainer.py`), evaluates it on the test set, and saves the results (`results.json`, final model `.pth`) under `results/final_models/<pe_type>/seed_XXX/`.
    * Finally, it aggregates results across all seeds and PE types, generates comparative plots (`*.png`) and summary tables (`summary_table.*`) in the `results/final_models/` directory using `utils/visualization.py`.

## Expected Results

* **Bayesian Optimization:** `results/bo_results/` will contain subdirectories for each PE type, each holding the Optuna database (`optuna.db`), detailed logs and outputs for every trial (`trial_XXX/`), and overall analysis plots/summaries for that PE type's optimization run. `results/best_configs/` will contain the best `config.json`, `best_checkpoint.pth`, and `best_model_info.json` identified for each PE type during BO.
* **Final Comparison:** `results/final_models/` will contain subdirectories for each PE type and seed combination evaluated, storing the final model, config used, detailed metrics (`results.json`), and potentially training plots. Crucially, it will also contain the overall comparative plots (`accuracy_comparison.png`, `convergence_comparison.png`, etc.) and the final summary tables (`summary_table.csv`, `summary_table.txt`) comparing all methods across the specified seeds.

## Contributing

*(Optional: Add guidelines if others might contribute)*

## License

*(Optional: Specify the license for your code, e.g., MIT, Apache 2.0. Remember to respect the licenses of included code like RoPE-ViT)*