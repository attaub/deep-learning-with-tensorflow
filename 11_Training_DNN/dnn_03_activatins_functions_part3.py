import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


#################################################################
def logit(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-5, 5, 200)


def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha * z, z)
#################################################################


(X_train_full, y_train_full), (
    X_test,
    y_test,
) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


#################################################################
#################################################################
## ELU


def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)


plt.plot(z, elu(z), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1, -1], 'k--')
plt.plot([0, 0], [-2.2, 3.2], 'k-')
plt.grid(True)
plt.title(r"ELU activation function ($\alpha=1$)", fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])

plt.show()


# Implementing ELU in TensorFlow is trivial, just specify the activation function when building each layer:


keras.layers.Dense(10, activation="elu")

# #################################################################
# # ### SELU
