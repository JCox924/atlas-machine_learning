import tensorflow as tf
from LAPAdam import preprocess_data
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(_, _), (X_test, y_test) = cifar10.load_data()

# Preprocess test data
X_test_p, y_test_p = preprocess_data(X_test, y_test)

# Load the trained model
cnn_model = tf.keras.models.load_model('cnn_model.h5')
print("Loaded model from cnn_model.h5")

# Evaluate the model
test_loss, test_accuracy = cnn_model.evaluate(X_test_p, y_test_p)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}")