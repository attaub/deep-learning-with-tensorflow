import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
import tensorflow as tf
from tensorflow import keras

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


#################################################################
## Recurrent Autoencoders

recurrent_encoder = keras.models.Sequential(
    [
        keras.layers.LSTM(100, return_sequences=True, input_shape=[28, 28]),
        keras.layers.LSTM(30),
    ]
)
recurrent_decoder = keras.models.Sequential(
    [
        keras.layers.RepeatVector(28, input_shape=[30]),
        keras.layers.LSTM(100, return_sequences=True),
        keras.layers.TimeDistributed(
            keras.layers.Dense(28, activation="sigmoid")
        ),
    ]
)
recurrent_ae = keras.models.Sequential([recurrent_encoder, recurrent_decoder])
recurrent_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(0.1),
    metrics=[rounded_accuracy],
)


history = recurrent_ae.fit(
    X_train, X_train, epochs=10, validation_data=(X_valid, X_valid)
)

show_reconstructions(recurrent_ae)
plt.show()
