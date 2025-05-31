import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from rnn_utils import plot_series
from rnn_utils import plot_learning_curves
from rnn_utils import plot_multiple_forecasts

from rnn_utils import generate_time_series

#################################################################
## Forecasting Several Steps Ahead


#################################################################
"""#
Create an RNN that predicts the next 10 steps at each time step.
That is,
instead of just forecasting time steps 50 to 59 based on time steps 0 to 49,
it will forecast time steps 1 to 10 at time step 0,
then time steps 2 to 11 at time step 1, and so on,
and finally it will forecast time steps 50 to 59 at the last time step.
Notice that the model is causal: when it makes predictions at any time step, it can only see past time steps.
"""

tf.random.set_seed(42)
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
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10)),
    ]
)


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mse(Y_true[:, -1], Y_pred[:, -1])


model.compile(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=[last_time_step_mse],
)
history = model.fit(
    X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid)
)

np.random.seed(43)
series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, 50:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

plot_multiple_forecasts(X_new, Y_new, Y_pred)
plt.show()
