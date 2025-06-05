import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


#################################################################
def logit(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-5, 5, 200)

#################################################################
## Nonsaturating Activation Functions
# Leaky ReLU


def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha * z, z)


plt.plot(z, leaky_relu(z, 0.05), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([0, 0], [-0.5, 4.2], 'k-')
plt.grid(True)
props = dict(facecolor='black', shrink=0.1)
plt.annotate(
    'Leak',
    xytext=(-3.5, 0.5),
    xy=(-5, -0.2),
    arrowprops=props,
    fontsize=14,
    ha="center",
)
plt.title("Leaky ReLU activation function", fontsize=14)
plt.axis([-5, 5, -0.5, 4.2])

plt.show()

# [m for m in dir(keras.activations) if not m.startswith("_")]
# [m for m in dir(keras.layers) if "relu" in m.lower()]

for m in dir(keras.activations):
    if not m.startswith("_"):
        print(m)

for m in dir(keras.layers):
    if not m.startswith("_"):
        print(m)


#################################################################
# Let's train a neural network on Fashion MNIST using the Leaky ReLU:


(X_train_full, y_train_full), (
    X_test,
    y_test,
) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, kernel_initializer="he_normal"),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(100, kernel_initializer="he_normal"),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(10, activation="softmax"),
    ]
)


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1e-3),
    metrics=["accuracy"],
)


history = model.fit(
    X_train, y_train, epochs=10, validation_data=(X_valid, y_valid)
)

