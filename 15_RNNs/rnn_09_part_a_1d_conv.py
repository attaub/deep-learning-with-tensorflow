import tensorflow as tf
from tensorflow import keras
import numpy as np 
from rnn_utils import last_time_step_mse
import matplotlib.pyplot as plt  



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
#################################################################
# ## Using One-Dimensional Convolutional Layers to Process Sequences
# 1D conv layer with kernel size 4, stride 2, VALID padding:
#
#               |-----2-----|     |-----5---...------|     |-----23----|
#         |-----1-----|     |-----4-----|   ...      |-----22----|
#   |-----0----|      |-----3-----|     |---...|-----21----|
# X: 0  1  2  3  4  5  6  7  8  9  10 11 12 ... 42 43 44 45 46 47 48 49
# Y: 1  2  3  4  5  6  7  8  9  10 11 12 13 ... 43 44 45 46 47 48 49 50
#   /10 11 12 13 14 15 16 17 18 19 20 21 22 ... 52 53 54 55 56 57 58 59
#
# Output:
#
# X:     0/3   2/5   4/7   6/9   8/11 10/13 .../43 42/45 44/47 46/49
# Y:     4/13  6/15  8/17 10/19 12/21 14/23 .../53 46/55 48/57 50/59

np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential(
    [
        keras.layers.Conv1D(
            filters=20,
            kernel_size=4,
            strides=2,
            padding="valid",
            input_shape=[None, 1],
        ),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10)),
    ]
)
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(
    X_train,
    Y_train[:, 3::2],
    epochs=20,
    validation_data=(X_valid, Y_valid[:, 3::2]),
)

