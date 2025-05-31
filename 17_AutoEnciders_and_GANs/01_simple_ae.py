import sklearn
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#############################################################
# pca with a linear Autoencoder
# Build 3D dataset:

np.random.seed(4)

# angles: cos(t) + sin(t)/2 + eps*X_m/2 
def generate_3d_data(m, w1=0.1, w2=0.3, noise=0.1):
    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5 # theta_r * 3pi/2 - 0.5
    data = np.empty((m, 3))

    # X1= cos(t) + sin(t)/2 + eps*X_m/2 
    data[:, 0] = (
        np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
    )

    # X2= sin(t) *0.7 + eps*X_m/2
    data[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2

    data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * np.random.randn(m)
    return data


X_train = generate_3d_data(10000)
X_train = X_train - X_train.mean(axis=0, keepdims=0)

# Now let's build the Autoencoder...

np.random.seed(42)
tf.random.set_seed(42)

encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])])
decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])
autoencoder = keras.models.Sequential([encoder, decoder])

autoencoder.compile(
    loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1.5)
)

history = autoencoder.fit(X_train, X_train, epochs=20)

codings = encoder.predict(X_train)

fig = plt.figure(figsize=(4, 3))
plt.plot(codings[:, 0], codings[:, 1], "b.")
plt.plot(X_train[:, 0], X_train[:, 1], "r.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
plt.show()
