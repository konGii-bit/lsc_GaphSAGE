
# Large Scale Computing - GraphSAGE Link Prediction

This repository contains code for the **Large Scale Computing** class project at AGH University, Department of Computer Science.

## Project Overview
The goal of this project is to train a GraphSAGE model for link prediction on protein interaction graphs.

---

## Installation

### Requirements
- Python 3.9 or higher

### Local Setup
1. Clone the repository.
2. Install PyTorch with CUDA 11.8 support:
   ```bash
   pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
   ```
3. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Athena Setup
Due to module incompatibilities, the installation procedure is the same as the local setup.

---

## Training

1. Upload the node file to the same directory as `train.py`.  
   If the file is located elsewhere, update the path in `config.py`.

2. Configure hyperparameters and file paths in `config.py`.

### Node file format
The input file should be a CSV with the following headers:
```
protein1 protein2 combined_score
```

3. Run the training script:
```bash
python train.py
```

---

## Saving Results

After training, results are saved in two directories:
- `results/` — Contains ROC plots and evaluation metrics in JSON format.
- `runs/` — TensorBoard logs for visualizing training progress.

---

## Viewing Results

Launch TensorBoard to visualize training metrics:
```bash
tensorboard --logdir=runs
```

By default, TensorBoard runs an interactive dashboard at:  
[http://localhost:6006](http://localhost:6006)

---

## TODO
- Clean up the codebase.
- Replace `config.py` with `.env` or YAML configuration.
- Split data into train, validation, and test subsets.
- Add hyperparameter selection via CLI arguments.
- Implement hyperparameter optimization (e.g., with Optuna).
