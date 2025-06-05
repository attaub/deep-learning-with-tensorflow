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
# ### tf.keras schedulers

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(
            300, activation="selu", kernel_initializer="lecun_normal"
        ),
        keras.layers.Dense(
            100, activation="selu", kernel_initializer="lecun_normal"
        ),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
s = 20 * len(X_train) // 32  # number of steps in 20 epochs (batch size = 32)
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer = keras.optimizers.SGD(learning_rate)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
n_epochs = 25
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=n_epochs,
    validation_data=(X_valid_scaled, y_valid),
)


# For piecewise constant scheduling, try this:
n_steps = n_epochs * len(X_train) // 32
# steps = np.arange(n_steps)
# lrs = lr0 * 0.1 ** (steps / s)

learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[5.0 * n_steps_per_epoch, 15.0 * n_steps_per_epoch],
    values=[0.01, 0.005, 0.001],
)


# ### 1Cycle scheduling

K = keras.backend


class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        K.set_value(
            self.model.optimizer.learning_rate,
            self.model.optimizer.learning_rate * self.factor,
        )


def find_learning_rate(
    model, X, y, epochs=1, batch_size=32, min_rate=10**-5, max_rate=10
):
    init_weights = model.get_weights()
    iterations = math.ceil(len(X) / batch_size) * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.learning_rate)
    K.set_value(model.optimizer.learning_rate, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(
        X, y, epochs=epochs, batch_size=batch_size, callbacks=[exp_lr]
    )
    K.set_value(model.optimizer.learning_rate, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses


def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis(
        [min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2]
    )
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")


# **Warning**: In the `on_batch_end()` method, `logs["loss"]` used to contain the batch loss, but in TensorFlow 2.2.0 it was replaced with the mean loss (since the start of the epoch). This explains why the graph below is much smoother than in the book (if you are using TF 2.2 or above). It also means that there is a lag between the moment the batch loss starts exploding and the moment the explosion becomes clear in the graph. So you should choose a slightly smaller learning rate than you would have chosen with the "noisy" graph. Alternatively, you can tweak the `ExponentialLearningRate` callback above so it computes the batch loss (based on the current mean loss and the previous mean loss):
#
# ```python
# class ExponentialLearningRate(keras.callbacks.Callback):
#     def __init__(self, factor):
#         self.factor = factor
#         self.rates = []
#         self.losses = []
#     def on_epoch_begin(self, epoch, logs=None):
#         self.prev_loss = 0
#     def on_batch_end(self, batch, logs=None):
#         batch_loss = logs["loss"] * (batch + 1) - self.prev_loss * batch
#         self.prev_loss = logs["loss"]
#         self.rates.append(K.get_value(self.model.optimizer.learning_rate))
#         self.losses.append(batch_loss)
#         K.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)
# ```


tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(
            300, activation="selu", kernel_initializer="lecun_normal"
        ),
        keras.layers.Dense(
            100, activation="selu", kernel_initializer="lecun_normal"
        ),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1e-3),
    metrics=["accuracy"],
)


batch_size = 128
rates, losses = find_learning_rate(
    model, X_train_scaled, y_train, epochs=1, batch_size=batch_size
)
plot_lr_vs_loss(rates, losses)
plt.show()


class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(
        self,
        iterations,
        max_rate,
        start_rate=None,
        last_iterations=None,
        last_rate=None,
    ):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0

    def _interpolate(self, iter1, iter2, rate1, rate2):
        return (rate2 - rate1) * (self.iteration - iter1) / (
            iter2 - iter1
        ) + rate1

    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(
                0, self.half_iteration, self.start_rate, self.max_rate
            )
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(
                self.half_iteration,
                2 * self.half_iteration,
                self.max_rate,
                self.start_rate,
            )
        else:
            rate = self._interpolate(
                2 * self.half_iteration,
                self.iterations,
                self.start_rate,
                self.last_rate,
            )
        self.iteration += 1
        K.set_value(self.model.optimizer.learning_rate, rate)


n_epochs = 25
onecycle = OneCycleScheduler(
    math.ceil(len(X_train) / batch_size) * n_epochs,
    max_rate=0.05,
)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=n_epochs,
    batch_size=batch_size,
    validation_data=(X_valid_scaled, y_valid),
    callbacks=[onecycle],
)
