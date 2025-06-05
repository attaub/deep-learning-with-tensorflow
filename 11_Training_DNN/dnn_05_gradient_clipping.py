from tensorflow import keras


# ## Gradient Clipping
# All Keras optimizers accept `clipnorm` or `clipvalue` arguments:

optimizer = keras.optimizers.SGD(clipvalue=1.0)
optimizer = keras.optimizers.SGD(clipnorm=1.0)
