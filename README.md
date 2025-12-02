# Semi-Supervised Medical Image Segmentation with Local Contrastive Loss (PyTorch Version)

This repository contains a PyTorch re-implementation of the method described in the paper: *"Local Contrastive Loss with Pseudo-label based Self-training for Semi-supervised Medical Image Segmentation"*.

We converted the original TensorFlow code into a modular PyTorch framework. This project aims to improve image segmentation performance when only a few labeled images are available (semi-supervised learning).

## Project Overview

The pipeline uses a **Joint Training** strategy to learn from both labeled and unlabeled data:

1.  **Supervised Branch:** Trains a U-Net on a small set of labeled images using Dice Loss.
2.  **Contrastive Branch:** Uses unlabeled images (with pseudo-labels) to learn better features. It forces pixels of the same class to be similar and pixels of different classes to be different.

## Project Structure

The project files are organized as follows:

  * **database/**: Contains the medical datasets (ACDC, MSD Heart, CHAOS).
  * **experiments/**: Stores training logs, checkpoints, and saved models.
  * **analysis/**: Contains Jupyter Notebooks for evaluating results and visualizing data (e.g., t-SNE plots)


  * **our_pytorch_version** — main folder of our PyTorch reimplementation
  - `configs/` — YAML configuration files for each dataset and experiment (paths, splits, hyperparameters)
  - `data/` — dataset utilities (ACDC / CHAOS / MSD Heart loaders, slice extraction, augmentations)
  - `losses/` — implementation of the Dice loss and the local pixel-wise contrastive loss
  - `models/` — model architectures, including the U-Net backbone, the dual heads, and the full Contrastive U-Net
  - `trainers/` — training logic (baseline supervised training and joint training with contrastive loss)
  - `scripts/` — preprocessing scripts for ACDC and CHAOS and small helper scripts to prepare the data
  - `utils/` — helper functions for evaluation and visualisation (Dice computation, t-SNE, segmentation plots, etc.)
  - `run_full_pipeline.py` — main entry point to run the complete training pipeline from the command line
  - `requirements.txt` — list of Python dependencies needed to run the PyTorch implementation

## Installation

To run this code, you need Python installed with the following libraries:

  * pytorch
  * torchvision
  * numpy
  * matplotlib
  * tqdm
  * scikit-learn
  * nibabel

## Usage

You can run the entire training pipeline using a single script. This script will train the baseline, generate pseudo-labels, and perform the joint training.

Run the following command from the root directory:

```bash
python our_pytorch_version/run_full_pipeline.py
```

### Configuration

To change the dataset (e.g., from ACDC to MSD Heart) or the number of labeled volumes (e.g., 1 or 2), you should modify the configuration settings directly inside `run_full_pipeline.py`.

## Pipeline Steps

The script `run_full_pipeline.py` performs these three steps automatically:

1.  **Train Baseline:** The model trains only on the limited labeled data (e.g., 1 labeled volume) using supervised learning.
2.  **Generate Pseudo-labels:** The baseline model predicts masks for the unlabeled data. These predictions are used as "fake" ground truth labels.
3.  **Joint Training:** The model is re-trained using both the labeled data (Supervised Loss) and the unlabeled data (Contrastive Loss) simultaneously.

## Analysis and Visualization

We provide Jupyter Notebooks for analysis to check performance:

  * `analyse_acdc_new_version.ipynb`: Results for the ACDC dataset.
  * `analyse_heart_new_version.ipynb`: Results for the MSD Heart dataset.
  * `analyse_chaos_new_version.ipynb`: Results for the CHAOS dataset.

These notebooks allow you to:

  * Calculate and plot Dice scores.
  * Visualize segmentation masks (Input vs. Prediction vs. Ground Truth).
  * Generate t-SNE plots to see how the model separates different classes (e.g., Heart vs. Background).

## Datasets

The code is designed to work with:

  * **ACDC:** Cardiac MRI (Multi-class segmentation).
  * **MSD Heart:** Cardiac MRI (Binary segmentation: Heart vs. Background).
  * **CHAOS:** Abdominal CT/MRI.

## References

  * Chaitanya et al., "Local Contrastive Loss with Pseudo-label based Self-training for Semi-supervised Medical Image Segmentation", Medical Image Analysis, 2023.