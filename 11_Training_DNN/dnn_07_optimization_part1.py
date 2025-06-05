import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


(X_train_full, y_train_full), (
    X_test,
    y_test,
) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

#################################################################

pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds
#################################################################
# # Faster Optimizers

# Momentum optimization
optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

# Nesterov Accelerated Gradient
optimizer = keras.optimizers.SGD(
    learning_rate=0.001, momentum=0.9, nesterov=True
)

# AdaGrad
optimizer = keras.optimizers.Adagrad(learning_rate=0.001)

# RMSProp
optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

# Adam Optimization
optimizer = keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999
)

# Adamax Optimization
optimizer = keras.optimizers.Adamax(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999
)

# Nadam Optimization
optimizer = keras.optimizers.Nadam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999
)


# ## Learning Rate Scheduling
# ### Power Scheduling
# ```lr = lr0 / (1 + steps / s)**c```
# * Keras uses `c=1` and `s = 1 / decay`
