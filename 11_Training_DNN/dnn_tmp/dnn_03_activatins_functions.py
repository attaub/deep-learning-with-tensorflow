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


[m for m in dir(keras.activations) if not m.startswith("_")]


[m for m in dir(keras.layers) if "relu" in m.lower()]

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

#################################################################
# Now let's try PReLU:


tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, kernel_initializer="he_normal"),
        keras.layers.PReLU(),
        keras.layers.Dense(100, kernel_initializer="he_normal"),
        keras.layers.PReLU(),
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

#################################################################
# ### SELU

# This activation function was proposed in this [great paper](https://arxiv.org/pdf/1706.02515.pdf) by Günter Klambauer, Thomas Unterthiner and Andreas Mayr, published in June 2017. During training, a neural network composed exclusively of a stack of dense layers using the SELU activation function and LeCun initialization will self-normalize: the output of each layer will tend to preserve the same mean and variance during training, which solves the vanishing/exploding gradients problem. As a result, this activation function outperforms the other activation functions very significantly for such neural nets, so you should really try it out. Unfortunately, the self-normalizing property of the SELU activation function is easily broken: you cannot use ℓ<sub>1</sub> or ℓ<sub>2</sub> regularization, regular dropout, max-norm, skip connections or other non-sequential topologies (so recurrent neural networks won't self-normalize). However, in practice it works quite well with sequential CNNs. If you break self-normalization, SELU will not necessarily outperform other activation functions.


from scipy.special import erfc

# alpha and scale to self normalize with mean 0 and standard deviation 1
# (see equation 14 in the paper):
alpha_0_1 = -np.sqrt(2 / np.pi) / (erfc(1 / np.sqrt(2)) * np.exp(1 / 2) - 1)
scale_0_1 = (
    (1 - erfc(1 / np.sqrt(2)) * np.sqrt(np.e))
    * np.sqrt(2 * np.pi)
    * (
        2 * erfc(np.sqrt(2)) * np.e**2
        + np.pi * erfc(1 / np.sqrt(2)) ** 2 * np.e
        - 2 * (2 + np.pi) * erfc(1 / np.sqrt(2)) * np.sqrt(np.e)
        + np.pi
        + 2
    )
    ** (-1 / 2)
)


def selu(z, scale=scale_0_1, alpha=alpha_0_1):
    return scale * elu(z, alpha)


plt.plot(z, selu(z), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1.758, -1.758], 'k--')
plt.plot([0, 0], [-2.2, 3.2], 'k-')
plt.grid(True)
plt.title("SELU activation function", fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])

plt.show()


# By default, the SELU hyperparameters (`scale` and `alpha`) are tuned in such a way that the mean output of each neuron remains close to 0, and the standard deviation remains close to 1 (assuming the inputs are standardized with mean 0 and standard deviation 1 too). Using this activation function, even a 1,000 layer deep neural network preserves roughly mean 0 and standard deviation 1 across all layers, avoiding the exploding/vanishing gradients problem:


np.random.seed(42)
Z = np.random.normal(size=(500, 100))  # standardized inputs
for layer in range(1000):
    W = np.random.normal(
        size=(100, 100), scale=np.sqrt(1 / 100)
    )  # LeCun initialization
    Z = selu(np.dot(Z, W))
    means = np.mean(Z, axis=0).mean()
    stds = np.std(Z, axis=0).mean()
    if layer % 100 == 0:
        print(
            "Layer {}: mean {:.2f}, std deviation {:.2f}".format(
                layer, means, stds
            )
        )


# Using SELU is easy:


keras.layers.Dense(10, activation="selu", kernel_initializer="lecun_normal")


# Let's create a neural net for Fashion MNIST with 100 hidden layers, using the SELU activation function:


np.random.seed(42)
tf.random.set_seed(42)


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(
    keras.layers.Dense(
        300, activation="selu", kernel_initializer="lecun_normal"
    )
)
for layer in range(99):
    model.add(
        keras.layers.Dense(
            100, activation="selu", kernel_initializer="lecun_normal"
        )
    )
model.add(keras.layers.Dense(10, activation="softmax"))


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1e-3),
    metrics=["accuracy"],
)


# Now let's train it. Do not forget to scale the inputs to mean 0 and standard deviation 1:


pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds


history = model.fit(
    X_train_scaled,
    y_train,
    epochs=5,
    validation_data=(X_valid_scaled, y_valid),
)


# Now look at what happens if we try to use the ReLU activation function instead:


np.random.seed(42)
tf.random.set_seed(42)


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(
    keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal")
)
for layer in range(99):
    model.add(
        keras.layers.Dense(
            100, activation="relu", kernel_initializer="he_normal"
        )
    )
model.add(keras.layers.Dense(10, activation="softmax"))


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1e-3),
    metrics=["accuracy"],
)


history = model.fit(
    X_train_scaled,
    y_train,
    epochs=5,
    validation_data=(X_valid_scaled, y_valid),
)
