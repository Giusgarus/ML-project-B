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
TO BE DECIDED

### Neural Network (PyTorch)
TO BE DECIDED

### Support Vector Machine (Keras)
TO BE DECIDED

### Support Vector Machine (Scikit-learn)
An SVM is implemented manually in PyTorch to gain a deeper understanding of its workings. Key features include:
TO BE DECIDED

### KNN (Scikit-learn)

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
