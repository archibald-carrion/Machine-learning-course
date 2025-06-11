# MNIST CNN Cluster Execution

This project provides code and resources to train and evaluate a Convolutional Neural Network (CNN) on the MNIST dataset, designed for execution on a computing cluster.

## Features

- Loads and preprocesses the MNIST dataset (handwritten digits).
- Implements a CNN using PyTorch for classification.
- Includes scripts and notebooks for both interactive exploration and batch execution.
- Designed for cluster environments (manual dataset upload recommended).

## Folder Contents

- `mnist.py` — Main script for loading data, training, and evaluating the CNN.
- `mnist_784.csv` — The MNIST dataset in CSV format (must be manually uploaded).
- `Introducción_a_las_redes_neuronales_convolucionales.ipynb` — Jupyter notebook with a step-by-step explanation and experiments.
- `run_mnist.sh` — Example shell script for running the training on a cluster.
- `run_mnist_12479.out` — Example output file from a cluster job.
- `.gitignore` — Standard ignore file.
- `README.md` — This file.

## Dataset

- The dataset (`mnist_784.csv`) must be manually downloaded and uploaded to the cluster.
- Download from: https://api.openml.org/d/554

> **Note:** Automatic download is possible, but not recommended on clusters due to potential network restrictions.

## Usage

1. Upload `mnist_784.csv` to the project directory on your cluster.
2. Install the required Python packages (see requirements below).
    Depending on your cluster setup, you may need to use a virtual environment or conda environment. Ensure that the environment has access to the necessary libraries.
3. Run the main script:
   ```
   python mnist.py
   ```
   Or submit the job using the provided shell script:
   ```
   sh run_mnist.sh
   ```
   Depending on your cluster setup, it could use a job scheduler like SLURM or PBS, so ensure the script is compatible with your environment.