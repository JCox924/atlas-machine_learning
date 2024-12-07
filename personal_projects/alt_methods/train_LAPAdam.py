import tensorflow as tf
from LAPAdam import create_cnn, preprocess_data, LAPAdamOptimizer
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Preprocess data
X_train_p, y_train_p = preprocess_data(X_train, y_train)
X_test_p, y_test_p = preprocess_data(X_test, y_test)

# Create the model
cnn_model = create_cnn()

# Compile the model with LAPAdam optimizer
lap_adam_optimizer = LAPAdamOptimizer(learning_rate=0.001, gamma=0.9)
cnn_model.compile(optimizer=lap_adam_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model
cnn_model.fit(X_train_p, y_train_p, epochs=20, batch_size=32, validation_data=(X_test_p, y_test_p))

# Save the model
cnn_model.save('cnn_model.h5')
print("Model saved as cnn_model.h5")
