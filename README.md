# ML-project-B
# Neural Networks and SVMs on Monks Datasets

## Project Overview
This project explores the application of Neural Networks (NNs) and Support Vector Machines (SVMs) on the Monks datasets. The goal is to compare the performance of these two machine learning techniques using two popular frameworks: **Keras** (for NNs) and **PyTorch** (for SVMs).

The Monks datasets consist of three distinct classification problems, making them a valuable benchmark for evaluating different models.

---

## Requirements
### General Dependencies
- Python 3.11
- pip or conda for managing Python packages

### Python Libraries
Ensure the following libraries are installed:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `keras`
- `torch`
- `tensorflow`

You can install the required libraries using:
```bash
pip install -r requirement.txt
```

---

## Monks Datasets
The Monks datasets are accessible from the UCI Machine Learning Repository:
- [Monks-1](https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems)
- [Monks-2](https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems)
- [Monks-3](https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems)

### Dataset Details
- **Monks-1**: Simple classification with clear separable patterns.
- **Monks-2**: Includes noise and is more challenging.
- **Monks-3**: Combines the complexity of Monks-2 with additional patterns.

---

## Project Structure
```plaintext
├── Datasets/              # Directory for storing datasets
├── Graphs/                # Contains the plot of the best models
├── Models/                # Contains model definitions for Keras and PyTorch
├── results/               # Store training logs and performance metrics
├── README.md              # Project documentation
```

---

## Implementation

### Neural Network (Keras)
A feedforward neural network is implemented using Keras. Key features include:
- Activation functions: `tanh`, `linear`, `relu`, `leaky relu`
- Optimization techniques: SGD, momentum regularization, minibatch, Adam
- Early stopping for preventing overfitting
- All features are configurable and can be toggled on or off

### Neural Network (PyTorch)
A feedforward neural network is implemented using PyTorch. Key features include:
- Activation functions: `tanh`, `linear`, `relu`, `leaky relu`
- Optimization techniques: SGD, momentum regularization, minibatch, Adam
- Early stopping for preventing overfitting
- All features are configurable and can be toggled on or off


### Neural Network (Scikit-learn)
An SVM is implemented using Scikit-learn. Key features include:
- Activation functions: `tanh`, `linear`, `relu`, `leaky relu`
- Optimization techniques: SGD, momentum regularization, minibatch, Adam
- Early stopping for preventing overfitting
- All features are configurable and can be toggled on or off


### Support Vector Machine (Scikit-learn)
An SVM is implemented using Scikit-learn. Key features include:
- Kernel functions: `linear`, `poly`, `rbf`, `sigmoid`
- Regularization parameter: `C`
- Gamma parameter for RBF kernel
- Support for both classification and regression tasks
- Grid search for hyperparameter tuning

### K-Nearest Neighbors (Scikit-learn)
A K-Nearest Neighbors (KNN) algorithm is implemented using Scikit-learn. Key features include:
- Number of neighbors: `k`
- Distance metrics: `euclidean`, `manhattan`, `minkowski`
- Weighting options: `uniform`, `distance`
- Support for both classification and regression tasks
- Grid search for hyperparameter tuning
- Cross-validation for model evaluation
- Early stopping for preventing overfitting
- All features are configurable and can be toggled on or off

---

## Evaluation Metrics
- **Accuracy**: Measures the proportion of correctly classified instances.
- **Precision, Recall, and F1-Score**: Provide deeper insights into the model's performance.
- **Confusion Matrix**: Visual representation of predictions vs actual labels.

---

## Results
Results from the experiments will be summarized and stored in the `results/` directory. Graphs and tables will help visualize:
- Training and validation accuracy over epochs
- Performance comparison across the three datasets

---

## Future Work
- Experiment with advanced neural network architectures.
- Implement SVMs with different kernels (e.g., RBF, polynomial).
- Test the models on additional datasets.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- UCI Machine Learning Repository for providing the Monks datasets.
- The developers of Keras and PyTorch for their powerful frameworks.
