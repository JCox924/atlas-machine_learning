import numpy as np
from neural_network_LAP import NeuralNetwork  # Your modified class
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_data, Y_data), (X_test, Y_test) = mnist.load_data()

# Preprocess the data
X_data = X_data.reshape(-1, 28*28) / 255.0  # Normalize pixel values
X_test = X_test.reshape(-1, 28*28) / 255.0

# Select a subset for binary classification (e.g., digits 0 and 1)
data_filter = np.where((Y_data == 0) | (Y_data == 1))
test_filter = np.where((Y_test == 0) | (Y_test == 1))

np.random.seed(42)

X_data, Y_data = X_data[data_filter], Y_data[data_filter]
X_test, Y_test = X_test[test_filter], Y_test[test_filter]

# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    X_data, Y_data, test_size=0.2, random_state=42)

# Transpose the data to match your model's input shape
X_train = X_train.T
Y_train = Y_train.reshape(1, -1)
X_val = X_val.T
Y_val = Y_val.reshape(1, -1)
X_test = X_test.T
Y_test = Y_test.reshape(1, -1)

# Initialize the neural network
nn_gd = NeuralNetwork(nx=784, nodes=128)
nn_lap = NeuralNetwork(nx=784, nodes=128)

# Train using gradient descent
print("Training with Gradient Descent:")
prediction_gd, cost_gd = nn_gd.train_gradient_descent(
    X_train, Y_train, X_val, Y_val, iterations=1000, alpha=0.1, verbose=True, graph=False, step=50)

# Train using Least Action Principle optimizer
print("\nTraining with Least Action Principle Optimizer:")
prediction_lap, cost_lap = nn_lap.train_LAP(
    X_train, Y_train, X_val, Y_val, iterations=1000, alpha=0.1, gamma=0.9, verbose=True, graph=False, step=50)

# Plotting both training and validation costs together on the same graph
plt.figure(figsize=(10, 6))
plt.plot(nn_gd.iteration_steps, nn_gd.costs, 'b-', label='GD Training Cost')
plt.plot(nn_gd.iteration_steps, nn_gd.val_costs, 'b--', label='GD Validation Cost')
plt.plot(nn_lap.iteration_steps, nn_lap.costs, 'r-', label='LAP Training Cost')
plt.plot(nn_lap.iteration_steps, nn_lap.val_costs, 'r--', label='LAP Validation Cost')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Training and Validation Cost Comparison')
plt.yscale('log')
plt.legend()
plt.show()

# Evaluate on test set
print("\nEvaluating on test set with Gradient Descent model:")
test_pred_gd, test_cost_gd = nn_gd.evaluate(X_test, Y_test)
accuracy_gd = np.mean(test_pred_gd == Y_test)
print(f"Test Cost (GD): {test_cost_gd}")
print(f"Test Accuracy (GD): {accuracy_gd * 100:.2f}%")

print("\nEvaluating on test set with LAP model:")
test_pred_lap, test_cost_lap = nn_lap.evaluate(X_test, Y_test)
accuracy_lap = np.mean(test_pred_lap == Y_test)
print(f"Test Cost (LAP): {test_cost_lap}")
print(f"Test Accuracy (LAP): {accuracy_lap * 100:.2f}%")

# Display training times
print(f"\nGradient Descent Training Time: {nn_gd.training_time:.2f} seconds")
print(f"LAP Optimizer Training Time: {nn_lap.training_time:.2f} seconds")
