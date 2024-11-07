import numpy as np


class NeuralNetwork:
    def __init__(self, nx, nodes, classes):
        """
        nx: Number of input features
        nodes: Number of nodes in the hidden layer
        classes: Number of output classes
        """
        if not isinstance(nx, int) or nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int) or nodes < 1:
            raise ValueError("nodes must be a positive integer")
        if not isinstance(classes, int) or classes < 2:
            raise ValueError("classes must be an integer greater than 1")

        # Initialize weights and biases
        self.W1 = np.random.randn(nodes, nx) * np.sqrt(2. / nx)
        self.b1 = np.zeros((nodes, 1))
        self.W2 = np.random.randn(classes, nodes) * np.sqrt(2. / nodes)
        self.b2 = np.zeros((classes, 1))

        # Initialize velocities for LAP optimizer
        self.VdW1 = np.zeros_like(self.W1)
        self.Vdb1 = np.zeros_like(self.b1)
        self.VdW2 = np.zeros_like(self.W2)
        self.Vdb2 = np.zeros_like(self.b2)

        # Placeholder for cache
        self.A1 = None
        self.A2 = None

    def softmax(self, Z):
        """Softmax activation function"""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stability
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def tanh(self, Z):
        """Tanh activation function"""
        return np.tanh(Z)

    def tanh_derivative(self, A):
        """Derivative of tanh activation function"""
        return 1 - np.power(A, 2)

    def forward_prop(self, X):
        """Perform forward propagation"""
        Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.tanh(Z1)
        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.softmax(Z2)
        return self.A1, self.A2

    def cost(self, Y, A):
        """Compute the cost using categorical cross-entropy"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A + 1e-8))  # Add epsilon for numerical stability
        return cost

    def backward_prop(self, X, Y):
        """Perform backward propagation"""
        m = Y.shape[1]
        # Output layer gradients
        dZ2 = self.A2 - Y  # Shape: (classes, m)
        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)  # Shape: (classes, nodes)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # Shape: (classes, 1)

        # Hidden layer gradients
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.tanh_derivative(self.A1)  # Shape: (nodes, m)
        dW1 = (1 / m) * np.dot(dZ1, X.T)  # Shape: (nodes, nx)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # Shape: (nodes, 1)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def gradient_descent(self, grads, alpha=0.01):
        """Update parameters using gradient descent"""
        self.W1 -= alpha * grads["dW1"]
        self.b1 -= alpha * grads["db1"]
        self.W2 -= alpha * grads["dW2"]
        self.b2 -= alpha * grads["db2"]

    def least_action_optimizer(self, grads, alpha=0.01, gamma=0.9):
        """Update parameters using the Least Action Principle Optimizer"""
        # Update velocities
        self.VdW1 = gamma * self.VdW1 + alpha * grads["dW1"]
        self.Vdb1 = gamma * self.Vdb1 + alpha * grads["db1"]
        self.VdW2 = gamma * self.VdW2 + alpha * grads["dW2"]
        self.Vdb2 = gamma * self.Vdb2 + alpha * grads["db2"]

        # Update parameters
        self.W1 -= self.VdW1
        self.b1 -= self.Vdb1
        self.W2 -= self.VdW2
        self.b2 -= self.Vdb2

    def predict(self, X):
        """Make predictions"""
        _, A2 = self.forward_prop(X)
        predictions = np.argmax(A2, axis=0)
        return predictions

    def evaluate(self, X, Y):
        """Evaluate the model's predictions"""
        predictions = self.predict(X)
        cost = self.cost(Y, self.A2)
        accuracy = np.mean(predictions == np.argmax(Y, axis=0))
        return predictions, cost, accuracy

    def train_gradient_descent(self, X, Y, iterations=1000, alpha=0.01, verbose=True, step=100):
        """Train the neural network using Gradient Descent"""
        costs = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.A2)
            grads = self.backward_prop(X, Y)
            self.gradient_descent(grads, alpha)

            if verbose and i % step == 0:
                costs.append(cost)
                print(f"Iteration {i}: Cost = {cost}")

        self.costs = costs
        return self.evaluate(X, Y)

    def train_LAP(self, X, Y, iterations=1000, alpha=0.01, gamma=0.9, verbose=True, step=100):
        """Train the neural network using the Least Action Principle Optimizer"""
        costs = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.A2)
            grads = self.backward_prop(X, Y)
            self.least_action_optimizer(grads, alpha, gamma)

            if verbose and i % step == 0:
                costs.append(cost)
                print(f"Iteration {i}: Cost = {cost}")

        self.costs = costs
        return self.evaluate(X, Y)
