# Deep RNN with Batch Norm

import numpy as np
import tensorflow as tf
from tensorflow import keras
from rnn_utils import last_time_step_mse
from rnn_utils import generate_time_series

# from rnn_10_utils import LNSimpleRNNCell

np.random.seed(42)
n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train = series[:7000, :n_steps]
X_valid = series[7000:9000, :n_steps]
X_test = series[9000:, :n_steps]
Y = np.empty((10000, n_steps, 10))

for step_ahead in range(1, 10 + 1):
    Y[..., step_ahead - 1] = series[..., step_ahead : step_ahead + n_steps, 0]

Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]
#
X_train.shape, Y_train.shape
#


np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential(
    [
        keras.layers.SimpleRNN(
            20, return_sequences=True, input_shape=[None, 1]
        ),
        keras.layers.BatchNormalization(),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.BatchNormalization(),
        keras.layers.TimeDistributed(keras.layers.Dense(10)),
    ]
)
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(
    X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid)
)
