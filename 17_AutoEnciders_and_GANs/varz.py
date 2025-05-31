import numpy as np
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)
# get the test training dataset
train_data, test_data = keras.datasets.fashion_mnist.load_data()

# separate the labels 
X_train_full, y_train_full = train_data
X_test, y_test = test_data

X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
