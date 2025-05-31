import numpy as np
import tensorflow as tf
from tensorflow import keras
from rnn_utils import last_time_step_mse
from rnn_utils import generate_time_series
from rnn_utils import plot_multiple_forecasts
from rnn_utils import plot_learning_curves
import matplotlib.pyplot as plt


#################################################################
# LSTMs

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

np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential(
    [
        keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.LSTM(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10)),
    ]
)

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(
    X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid)
)

model.evaluate(X_valid, Y_valid)

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

np.random.seed(43)
series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, 50:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

plot_multiple_forecasts(X_new, Y_new, Y_pred)
plt.show()

