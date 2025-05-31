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
## Extra Material

# ## Hashing Using a Binary Autoencoder

# Let's load the Fashion MNIST dataset again:

(X_train_full, y_train_full), (
    X_test,
    y_test,
) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# Let's train an autoencoder where the encoder has a 16-neuron output layer, using the sigmoid activation function, and heavy Gaussian noise just before it. During training, the noise layer will encourage the previous layer to learn to output large values, since small values will just be crushed by the noise. In turn, this means that the output layer will output values close to 0 or 1, thanks to the sigmoid activation function. Once we round the output values to 0s and 1s, we get a 16-bit "semantic" hash. If everything works well, images that look alike will have the same hash. This can be very useful for search engines: for example, if we store each image on a server identified by the image's semantic hash, then all similar images will end up on the same server. Users of the search engine can then provide an image to search for, and the search engine will compute the image's hash using the encoder, and quickly return all the images on the server identified by that hash.

tf.random.set_seed(42)
np.random.seed(42)

hashing_encoder = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(100, activation="selu"),
        keras.layers.GaussianNoise(15.0),
        keras.layers.Dense(16, activation="sigmoid"),
    ]
)

hashing_decoder = keras.models.Sequential(
    [
        keras.layers.Dense(100, activation="selu", input_shape=[16]),
        keras.layers.Dense(28 * 28, activation="sigmoid"),
        keras.layers.Reshape([28, 28]),
    ]
)

hashing_ae = keras.models.Sequential([hashing_encoder, hashing_decoder])

hashing_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Nadam(),
    metrics=[rounded_accuracy],
)
history = hashing_ae.fit(
    X_train, X_train, epochs=10, validation_data=(X_valid, X_valid)
)

# The autoencoder compresses the information so much (down to 16 bits!) that it's quite lossy, but that's okay, we're using it to produce semantic hashes, not to perfectly reconstruct the images:

show_reconstructions(hashing_ae)
plt.show()

# Notice that the outputs are indeed very close to 0 or 1 (left graph):
plot_activations_histogram(hashing_encoder)
plt.show()

# Now let's see what the hashes look like for the first few images in the validation set:

hashes = np.round(hashing_encoder.predict(X_valid)).astype(np.int32)
hashes *= np.array([[2**bit for bit in range(16)]])
hashes = hashes.sum(axis=1)
for h in hashes[:5]:
    print("{:016b}".format(h))
print("...")

# Now let's find the most common image hashes in the validation set, and display a few images for each hash. In the following image, all the images on a given row have the same hash:

from collections import Counter

n_hashes = 10
n_images = 8

top_hashes = Counter(hashes).most_common(n_hashes)

plt.figure(figsize=(n_images, n_hashes))
for hash_index, (image_hash, hash_count) in enumerate(top_hashes):
    indices = hashes == image_hash
    for index, image in enumerate(X_valid[indices][:n_images]):
        plt.subplot(n_hashes, n_images, hash_index * n_images + index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

