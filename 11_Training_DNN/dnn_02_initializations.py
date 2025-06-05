from tensorflow import keras

# ## Xavier and He Initialization
# [name for name in dir(keras.initializers) if not name.startswith("_")]

print()
for name in dir(keras.initializers):
    if not name.startswith("_"):
        print(name)


keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")


init = keras.initializers.VarianceScaling(
    scale=2.0, mode='fan_avg', distribution='uniform'
)

keras.layers.Dense(10, activation="relu", kernel_initializer=init)
