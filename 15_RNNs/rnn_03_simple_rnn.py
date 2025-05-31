import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from rnn_utils import plot_series
from rnn_utils import plot_learning_curves

#################################################################
from rnn_utils import generate_time_series

np.random.seed(42)
tf.random.set_seed(42)

n_steps = 50
series = generate_time_series(10000, n_steps + 1)  # n_steps + 1
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

#################################################################
## Using a Simple RNN

np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential(
    [keras.layers.SimpleRNN(1, input_shape=[None, 1])]
)

optimizer = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(
    X_train, y_train, epochs=20, validation_data=(X_valid, y_valid)
)

model.evaluate(X_valid, y_valid)
plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

y_pred = model.predict(X_valid)
plot_series(n_steps, X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()
