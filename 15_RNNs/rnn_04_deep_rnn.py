import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from rnn_utils import plot_series
from rnn_utils import generate_time_series
from rnn_utils import plot_learning_curves

#################################################################
np.random.seed(42)
tf.random.set_seed(42)

np.random.seed(42)
n_steps = 50
series = generate_time_series(10000, n_steps + 1)  # n_steps + 1
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

print(X_train.shape, y_train.shape)



#################################################################
## Deep RNNs
#
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential(
    [
        keras.layers.SimpleRNN(
            20, return_sequences=True, input_shape=[None, 1]
        ),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.SimpleRNN(1),
    ]
)
model.compile(loss="mse", optimizer="adam")
history = model.fit(
    X_train, y_train, epochs=20, validation_data=(X_valid, y_valid)
)

#################################################################
model.evaluate(X_valid, y_valid)
plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()


y_pred = model.predict(X_valid)
plot_series(n_steps, X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()

#################################################################
# Make the second `SimpleRNN` layer return only the last output:
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
