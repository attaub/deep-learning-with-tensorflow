import tensorflow as tf
from tensorflow import keras
import numpy as np 
from rnn_utils import last_time_step_mse
from rnn_utils import generate_time_series
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

# X_train.shape, Y_train.shape
#################################################################
# ## WaveNet
# C2  /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\.../\ /\ /\ /\ /\ /\
#    \  /  \  /  \  /  \  /  \  /  \  /  \       /  \  /  \  /  \
#      /    \      /    \      /    \                 /    \
# C1  /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\  /\ /.../\ /\ /\ /\ /\ /\ /\
# X: 0  1  2  3  4  5  6  7  8  9  10 11 12 ... 43 44 45 46 47 48 49
# Y: 1  2  3  4  5  6  7  8  9  10 11 12 13 ... 44 45 46 47 48 49 50
#   /10 11 12 13 14 15 16 17 18 19 20 21 22 ... 53 54 55 56 57 58 59
#
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[None, 1]))
for rate in (1, 2, 4, 8) * 2:
    model.add(
        keras.layers.Conv1D(
            filters=20,
            kernel_size=2,
            padding="causal",
            activation="relu",
            dilation_rate=rate,
        )
    )
model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(
    X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid)
)

#################################################################
# Here is the original WaveNet defined in the paper: it uses Gated Activation Units instead of ReLU and parametrized skip connections, plus it pads with zeros on the left to avoid getting shorter and shorter sequences:

class GatedActivationUnit(keras.layers.Layer):
    def __init__(self, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)

    def call(self, inputs):
        n_filters = inputs.shape[-1] // 2
        linear_output = self.activation(inputs[..., :n_filters])
        gate = keras.activations.sigmoid(inputs[..., n_filters:])
        return self.activation(linear_output) * gate



def wavenet_residual_block(inputs, n_filters, dilation_rate):
    z = keras.layers.Conv1D(
        2 * n_filters,
        kernel_size=2,
        padding="causal",
        dilation_rate=dilation_rate,
    )(inputs)
    z = GatedActivationUnit()(z)
    z = keras.layers.Conv1D(n_filters, kernel_size=1)(z)
    return keras.layers.Add()([z, inputs]), z



#################################################################
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
n_layers_per_block = 3  # 10 in the paper
n_blocks = 1  # 3 in the paper
n_filters = 32  # 128 in the paper
n_outputs = 10  # 256 in the paper
inputs = keras.layers.Input(shape=[None, 1])
z = keras.layers.Conv1D(n_filters, kernel_size=2, padding="causal")(inputs)
skip_to_last = []
for dilation_rate in [2**i for i in range(n_layers_per_block)] * n_blocks:
    z, skip = wavenet_residual_block(z, n_filters, dilation_rate)
    skip_to_last.append(skip)
z = keras.activations.relu(keras.layers.Add()(skip_to_last))
z = keras.layers.Conv1D(n_filters, kernel_size=1, activation="relu")(z)
Y_proba = keras.layers.Conv1D(n_outputs, kernel_size=1, activation="softmax")(
    z
)
model = keras.models.Model(inputs=[inputs], outputs=[Y_proba])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(
    X_train, Y_train, epochs=2, validation_data=(X_valid, Y_valid)
)
