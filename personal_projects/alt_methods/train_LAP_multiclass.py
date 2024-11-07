import numpy as np
from neural_network_LAP_multiclass import NeuralNetwork
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Load the CIFAR-10 dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Preprocess the data
# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten the images
X_train = X_train.reshape(-1, 32*32*3).T  # Shape: (3072, number of samples)
X_test = X_test.reshape(-1, 32*32*3).T

# Convert labels to one-hot encoding
Y_train = to_categorical(Y_train, num_classes=10).T  # Shape: (10, number of samples)
Y_test = to_categorical(Y_test, num_classes=10).T

# Initialize the neural networks
nn_gd = NeuralNetwork(nx=3072, nodes=128, classes=10)
nn_lap = NeuralNetwork(nx=3072, nodes=128, classes=10)

# Train using Gradient Descent
print("Training with Gradient Descent:")
predictions_gd, cost_gd, accuracy_gd = nn_gd.train_gradient_descent(
    X_train, Y_train, iterations=1000, alpha=0.1, verbose=True, step=100)

print(f"\nTraining Cost (GD): {cost_gd}")
print(f"Training Accuracy (GD): {accuracy_gd * 100:.2f}%")

# Train using Least Action Principle Optimizer
print("\nTraining with Least Action Principle Optimizer:")
predictions_lap, cost_lap, accuracy_lap = nn_lap.train_LAP(
    X_train, Y_train, iterations=1000, alpha=0.1, gamma=0.9, verbose=True, step=100)

print(f"\nTraining Cost (LAP): {cost_lap}")
print(f"Training Accuracy (LAP): {accuracy_lap * 100:.2f}%")

# Evaluate on test set using GD model
print("\nEvaluating on test set with Gradient Descent model:")
test_predictions_gd, test_cost_gd, test_accuracy_gd = nn_gd.evaluate(X_test, Y_test)
print(f"Test Cost (GD): {test_cost_gd}")
print(f"Test Accuracy (GD): {test_accuracy_gd * 100:.2f}%")

# Evaluate on test set using LAP model
print("\nEvaluating on test set with LAP model:")
test_predictions_lap, test_cost_lap, test_accuracy_lap = nn_lap.evaluate(X_test, Y_test)
print(f"Test Cost (LAP): {test_cost_lap}")
print(f"Test Accuracy (LAP): {test_accuracy_lap * 100:.2f}%")

# Plotting training costs
plt.figure(figsize=(10, 6))
iterations = np.arange(0, 1001, 100)
plt.plot(iterations, nn_gd.costs, 'b-', label='Gradient Descent')
plt.plot(iterations, nn_lap.costs, 'r-', label='LAP Optimizer')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Training Cost Comparison on CIFAR-10')
plt.legend()
plt.show()
