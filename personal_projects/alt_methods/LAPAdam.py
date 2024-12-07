import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# We'll borrow the data preprocessor from transfer_learning and make some adjustments

def preprocess_data(X, Y):
    """
    Preprocesses the data for the model.

    Parameters:
        X: np.ndarray - shape (m, 32, 32, 3), containing CIFAR-10 data
        Y: np.ndarray - shape (m,), containing CIFAR-10 labels for X

    Returns:
        X_p: np.ndarray - preprocessed X
        Y_p: np.ndarray - preprocessed Y
    """
    X_p = np.array([tf.image.resize(img, (32, 32)).numpy() for img in X])
    # Apply MobileNetV2 preprocessing
    X_p = K.applications.mobilenet_v2.preprocess_input(X_p)
    # Convert labels to one-hot encoding
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def create_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # Output layer for 10 classes
    ])
    return model


class LAPAdamOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, gamma=0.9, name="LAPAdam", **kwargs):
        super().__init__(name, **kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.gamma = gamma
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.V = {}  # Velocity

    def _get_or_create_slot(self, var):
        """Ensure slots for moments and velocities are initialized for a variable."""
        var_ref = var.ref()
        if var_ref not in self.m:
            self.m[var_ref] = tf.Variable(tf.zeros_like(var), trainable=False)
        if var_ref not in self.v:
            self.v[var_ref] = tf.Variable(tf.zeros_like(var), trainable=False)
        if var_ref not in self.V:
            self.V[var_ref] = tf.Variable(tf.zeros_like(var), trainable=False)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        """Apply the gradients using LAPAdam logic."""
        var_ref = var.ref()
        self._get_or_create_slot(var)  # Ensure slots are initialized

        # Update moments
        self.m[var_ref].assign(self.beta_1 * self.m[var_ref] + (1.0 - self.beta_1) * grad)
        self.v[var_ref].assign(self.beta_2 * self.v[var_ref] + (1.0 - self.beta_2) * tf.square(grad))

        # Compute bias-corrected moments
        m_hat = self.m[var_ref] / (1.0 - self.beta_1)
        v_hat = self.v[var_ref] / (1.0 - self.beta_2)

        # Compute Adam update
        adam_update = self.learning_rate * m_hat / (tf.sqrt(v_hat) + self.epsilon)

        # Update velocity
        self.V[var_ref].assign(self.gamma * self.V[var_ref] + adam_update)

        # Update the variable
        var.assign_sub(self.V[var_ref])

    def apply_gradients(self, grads_and_vars, name=None):
        """Apply gradients to variables."""
        for grad, var in grads_and_vars:
            self._resource_apply_dense(grad, var)

