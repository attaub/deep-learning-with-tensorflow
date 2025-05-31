import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from rnn_utils import generate_time_series

np.random.seed(42)
tf.random.set_seed(42)

np.random.seed(42)
n_steps = 50
series = generate_time_series(10000, n_steps + 1)  # n_steps + 1
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

print(X_train.shape, y_train.shape)

