#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import random
import os

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer

lib = np.load('MNIST.npz')
X_3D = lib['X_train']
X = X_3D.reshape((X_3D.shape[0], -1))  # Flatten the images

# Normalize the data (optional but recommended)
X = X / 255.0

# Step 1: Define the input layer
input_layer = tf.keras.Input(shape=(X.shape[1],))

# Step 2: Create the batch normalization layer
output_layer = create_batch_norm_layer(input_layer, 256, tf.nn.tanh)

# Step 3: Build the model
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Step 4: Run the model on your data
a = model(X)

# Print the output
print(a)
