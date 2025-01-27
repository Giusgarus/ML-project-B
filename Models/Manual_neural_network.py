import numpy as np
import copy

# NeuralNetwork Class
class NeuralNetwork:
    def __init__(
        self,
        input_size=6,
        output_size=1,
        learning_rate=0.01,
        epochs=10,
        batch_size=16,
        hidden_size=3,
        hidden_layers=1,
        momentum=0.9,
        lamb=0.3,
    ):
        """
        Initializes the neural network with the specified parameters and 
        random weights for the layers using He initialization.
        """
        np.random.seed(30)  # Set a seed for reproducibility
        
        # Set network parameters
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.momentum = momentum
        self.lamb = lamb
        
        # Initialize weights for each layer
        self.weights = [None] * (self.hidden_layers + 1)

        for i in range(self.hidden_layers + 1):
            if i == 0:  # Input to first hidden layer
                w = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
                b = np.zeros(hidden_size)
                self.weights[i] = [w, b]
            elif i == self.hidden_layers:  # Last hidden layer to output
                w = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
                b = np.zeros(output_size)
                self.weights[i] = [w, b]
            else:  # Hidden layers
                w = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
                b = np.zeros(hidden_size)
                self.weights[i] = [w, b]

    def train(self, X, y, X_val=None, y_val=None):
        """
        Trains the neural network using mini-batch gradient descent, 
        momentum, and Tikhonov regularization.
        """
        print(X.shape[0])  # Print the number of training samples
        
        # Initialize variables for forward and backward propagation
        loss = np.zeros(self.batch_size)
        count = 0
        atot = [None] * (self.hidden_layers)
        for i in range(self.hidden_layers):
            atot[i] = np.zeros((self.batch_size, self.hidden_size))
        output = np.zeros(self.batch_size)
        input_data = np.zeros((self.batch_size, X.shape[1]))
        dW = [None] * (self.hidden_layers + 1)
        dW_old = [(0, 0, 0)] * (self.hidden_layers + 1)
        anticipate = True  # Flag for anticipating weight updates

        for i in range(X.shape[0]):
            # Forward propagation
            input_data[count] = X[i]

            # Copy previous weight updates
            if any(x is not None for x in dW):
                dW_old = copy.deepcopy(dW)

            # Anticipate momentum-based updates for weights
            if anticipate:
                anticipate_weights = copy.deepcopy(self.weights)
                anticipate = False
                for i in range(self.hidden_layers + 1):
                    if i == self.hidden_layers:
                        anticipate_weights[i][0] = self.weights[i][0] + self.momentum * dW_old[i][0]
                        anticipate_weights[i][1] = self.weights[i][1] + self.momentum * dW_old[i][1]
                    else:
                        anticipate_weights[i][0] = self.weights[i][0] + self.momentum * dW_old[i][0]
                        sumdb = np.sum(dW_old[i][1], axis=0)
                        anticipate_weights[i][1] = self.weights[i][1] + self.momentum * sumdb

            # Forward propagation through the hidden layers
            for j in range(self.hidden_layers):
                if j == 0:  # Input to first hidden layer
                    a = self.relu(
                        np.dot(anticipate_weights[j][0], X[i][:, np.newaxis]).T
                        + anticipate_weights[j][1]
                    )
                    atot[j][count] = a
                else:  # Hidden layer to hidden layer
                    a = self.relu(
                        np.dot(atot[j - 1][count], anticipate_weights[j][0])
                        + anticipate_weights[j][1]
                    )
                    atot[j][count] = a

            # Output layer
            z1 = self.sigmoid(
                np.dot(
                    anticipate_weights[self.hidden_layers][0],
                    atot[self.hidden_layers - 1][count].T,
                )
                + anticipate_weights[self.hidden_layers][1]
            )[0]
            z1 = np.clip(z1, 1e-15, 1 - 1e-15)  # Avoid numerical instability
            output[count] = z1

            # Compute the loss for the batch
            loss[count] = - (y[i] * np.log(z1) + (1 - y[i]) * np.log(1 - z1))
            count += 1

            if count == self.batch_size:
                # Backpropagation to compute gradients
                for i in range(self.hidden_layers, -1, -1):
                    if i == self.hidden_layers:  # Output layer
                        dZ = loss * self.sigmoid_derivate(output)
                        dw = self.GradientProductOutput(dZ, atot[i - 1])
                        db = np.sum(dZ)
                        dW[i] = (dw, db, dZ)

                    elif i == (self.hidden_layers - 1):  # Last hidden layer
                        if self.hidden_layers != 1:
                            dA = self.error_hidden(anticipate_weights[i + 1][0], dW[i + 1][2])
                            dZ = dA * self.relu_derivative(atot[i])
                            dw = self.GradientProductHidden(dZ, atot[i - 1])
                            db = dZ
                            dW[i] = (dw, db, dZ)
                        else:
                            dA = self.error_hidden(anticipate_weights[i + 1][0], dW[i + 1][2])
                            dZ = dA * self.relu_derivative(atot[i])
                            dw = self.GradientProductInput(dZ, input_data)
                            db = dZ
                            dW[i] = (dw, db, dZ)

                    elif i == 0:  # Input to first hidden layer
                        dA = self.error_hidden_matrix(
                            anticipate_weights[i + 1][0], dW[i + 1][2]
                        )
                        dZ = dA * self.relu_derivative(atot[i])
                        dw = self.GradientProductInput(dZ, input_data)
                        db = dZ
                        dW[i] = (dw, db, dZ)

                    else:  # Hidden layer to hidden layer
                        dA = self.error_hidden_matrix(
                            anticipate_weights[i + 1][0], dW[i + 1][2]
                        )
                        dZ = dA * self.relu_derivative(atot[i])
                        dw = self.GradientProductHidden(dZ, atot[i - 1])
                        db = dZ
                        dW[i] = (dw, db, dZ)

                # Update weights using gradients, momentum, and regularization
                for i in range(self.hidden_layers + 1):
                    tikhonov_weight = self.lamb * self.weights[i][0]  # Regularization for weights
                    tikhonov_bias = self.lamb * self.weights[i][1]  # Regularization for biases
                    mom_weight = self.momentum * dW_old[i][0]
                    mom_bias = self.momentum * dW_old[i][1]
                    if i == self.hidden_layers:  # Output layer
                        self.weights[i][0] = self.weights[i][0] + self.learning_rate * dW[i][0] - tikhonov_weight + mom_weight
                        self.weights[i][1] = self.weights[i][1] + self.learning_rate * dW[i][1] - tikhonov_bias + mom_bias
                    else:  # Hidden layers
                        self.weights[i][0] = self.weights[i][0] + self.learning_rate * dW[i][0] - tikhonov_weight + mom_weight
                        sumdb = np.sum(dW[i][1], axis=0)
                        sumdbmom = np.sum(dW_old[i][1], axis=0)
                        mom_bias = self.momentum * sumdbmom
                        self.weights[i][1] = self.weights[i][1] + self.learning_rate * sumdb - tikhonov_bias + mom_bias

                anticipate = True  # Reset anticipation flag
                count = 0

    def relu(self, z):
        """
        Implements the ReLU activation function with a small slope (leaky ReLU).
        """
        alpha = 0.01
        return np.maximum(alpha * z, z)

    def relu_derivative(self, a):
        """
        Computes the derivative of the ReLU activation function.
        """
        alpha = 0.01
        return np.where(a > 0, 1, alpha)

    def sigmoid(self, z):
        """
        Implements the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivate(self, z):
        """
        Computes the derivative of the sigmoid activation function.
        """
        return z * (1 - z)

    def GradientProductInput(self, a, b):
        """
        Computes the gradient product between input layer activations and sigmas.
        """
        return np.dot(a.T, b)

    def GradientProductHidden(self, a, b):
        """
        Computes the gradient product between hidden layer activations and sigmas.
        """
        return np.dot(a.T, b)

    def GradientProductOutput(self, a, b):
        """
        Computes the gradient product for the output layer.
        """
        return np.dot(a, b).reshape(1, -1)

    def error_hidden(self, a, b):
        """
        Propagates the error back from a single hidden layer.
        """
        return b[:, np.newaxis] * a

    def error_hidden_matrix(self, a, b):
        """
        Propagates the error back from multiple hidden layers.
        """
        return np.dot(b, a.T)
