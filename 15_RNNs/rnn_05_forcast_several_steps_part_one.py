import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from rnn_utils import generate_time_series
from rnn_utils import plot_learning_curves
from rnn_utils import plot_series
from rnn_utils import plot_multiple_forecasts

#################################################################
## Forecasting Several Steps Ahead

np.random.seed(42)
tf.random.set_seed(42)

np.random.seed(42)
n_steps = 50
series = generate_time_series(10000, n_steps + 1)  # n_steps + 1
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

print(X_train.shape, y_train.shape)

np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential(
    [
        keras.layers.SimpleRNN(
            20, return_sequences=True, input_shape=[None, 1]
        ),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1),
    ]
)
model.compile(loss="mse", optimizer="adam")
history = model.fit(
    X_train, y_train, epochs=20, validation_data=(X_valid, y_valid)
)

model.evaluate(X_valid, y_valid)
#################################################################
plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

y_pred = model.predict(X_valid)
plot_series(n_steps, X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()
np.random.seed(43)

# not 42, as it would give the first series in the train set
series = generate_time_series(1, n_steps + 10)
X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
X = X_new
for step_ahead in range(10):
    y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)
Y_pred = X[:, n_steps:]

Y_pred.shape


plot_multiple_forecasts(X_new, Y_new, Y_pred)
plt.show()


#################################################################
"""
Use this model to predict the next 10 values. regenerate the sequences with 9 more time steps.
"""

np.random.seed(42)
n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

# now let's predict the next 10 values one by one:

X = X_valid
for step_ahead in range(10):
    y_pred_one = model.predict(X)[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)
Y_pred = X[:, n_steps:, 0]

Y_pred.shape

np.mean(keras.metrics.mse(Y_valid, Y_pred))
# np.mean(keras.metrics.mse(Y_valid, Y_pred))

#################################################################
# Let's compare this performance with some baselines: naive predictions and a simple linear model:
#
Y_naive_pred = np.tile(
    X_valid[:, -1], 10
)  # take the last time step value, and repeat it 10 times
# np.mean(keras.metrics.mse(Y_valid, Y_naive_pred))
np.mean(keras.metrics.mse(Y_valid, Y_naive_pred))

np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential(
    [keras.layers.Flatten(input_shape=[50, 1]), keras.layers.Dense(10)]
)
model.compile(loss="mse", optimizer="adam")
history = model.fit(
    X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid)
)
# Now let's create an RNN that predicts all 10 next values at once:
#
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential(
    [
        keras.layers.SimpleRNN(
            20, return_sequences=True, input_shape=[None, 1]
        ),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(10),
    ]
)
model.compile(loss="mse", optimizer="adam")
history = model.fit(
    X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid)
)
#
np.random.seed(43)
series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, -10:, :]
Y_pred = model.predict(X_new)[..., np.newaxis]
#
plot_multiple_forecasts(X_new, Y_new, Y_pred)
plt.show()
