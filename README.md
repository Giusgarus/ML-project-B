# ML-project-B

## Overview

This project explores the use of **Neural Networks (NN)** and **Support Vector Machines (SVM)** for classification tasks on the **MONKS datasets**. The models will be implemented using two popular frameworks: **Keras** (for NN) and **PyTorch** (for SVM). We aim to compare the performance of these two methods across all three MONKS datasets.

---

## Project Structure

```plaintext
.
├── datasets/
│   ├── monks-1.csv
│   ├── monks-2.csv
│   ├── monks-3.csv
├── notebooks/
│   ├── nn_keras_monks.ipynb     # Neural Network implementation in Keras
│   ├── svm_pytorch_monks.ipynb  # SVM implementation in PyTorch
├── src/
│   ├── data_preprocessing.py    # Preprocessing utilities for the MONKS datasets
│   ├── models.py                # Model definitions for Keras and PyTorch
├── results/
│   ├── performance_metrics.csv  # Comparison metrics for all datasets and models
├── README.md