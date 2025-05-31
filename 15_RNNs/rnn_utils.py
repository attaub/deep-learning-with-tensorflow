import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from pathlib import Path

np.random.seed(42)
tf.random.set_seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LayerNormalization

#################################################################
### Generate the Dataset

#################################################################
def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mse(Y_true[:, -1], Y_pred[:, -1])

#################################################################
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # + noise
    return series[..., np.newaxis].astype(np.float32)

#################################################################
def plot_series(
    n_steps,
    series,
    y=None,
    y_pred=None,
    x_label="$t$",
    y_label="$x(t)$",
    legend=True,
):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bo", label="Target")
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "rx", markersize=10, label="Prediction")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])
    if legend and (y or y_pred):
        plt.legend(fontsize=14, loc="upper left")


#################################################################
def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(
        np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss"
    )
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)


#################################################################
def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(n_steps, X[0, :, 0])
    plt.plot(
        np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "bo-", label="Actual"
    )
    plt.plot(
        np.arange(n_steps, n_steps + ahead),
        Y_pred[0, :, 0],
        "rx-",
        label="Forecast",
        markersize=10,
    )
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)
#################################################################
# class LNSimpleRNNCell(keras.layers.Layer):
#     def __init__(self, units, activation="tanh", **kwargs):
#         super().__init__(**kwargs)
#         self.state_size = units
#         self.output_size = units
#         self.simple_rnn_cell = keras.layers.SimpleRNNCell(
#             units, activation=None
#         )
#         self.layer_norm = LayerNormalization()
#         self.activation = keras.activations.get(activation)

#     def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
#         if inputs is not None:
#             batch_size = tf.shape(inputs)[0]
#             dtype = inputs.dtype
#         return [tf.zeros([batch_size, self.state_size], dtype=dtype)]

#     def call(self, inputs, states):
#         outputs, new_states = self.simple_rnn_cell(inputs, states)
#         norm_outputs = self.activation(self.layer_norm(outputs))
#         return norm_outputs, [norm_outputs]
