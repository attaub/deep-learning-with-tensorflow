import sklearn
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from utils import rounded_accuracy, plot_image, show_reconstructions
import matplotlib as mpl

# get the test training dataset
from varz import *

tf.random.set_seed(42)
np.random.seed(42)

stacked_encoder = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(100, activation="selu"),  # X_{n,}
        keras.layers.Dense(30, activation="selu"),  # [30, 100] [100, 1]
    ]
)

stacked_decoder = keras.models.Sequential(
    [
        keras.layers.Dense(
            100, activation="selu", input_shape=[30]
        ),  # [1,30] [30, 100]
        keras.layers.Dense(
            28 * 28, activation="sigmoid"
        ),  # [1, 100] [100, 784]
        keras.layers.Reshape([28, 28]),  # reshape [28, 28]
    ]
)
stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])

stacked_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1.5),
    metrics=[rounded_accuracy],
)

history = stacked_ae.fit(
    X_train, X_train, epochs=20, validation_data=(X_valid, X_valid)
)

# This function processes a few test images through the autoencoder and displays the original images and their reconstructions:


show_reconstructions(stacked_ae, X_valid)

#######################################
## Visualizing Fashion MNIST

np.random.seed(42)

from sklearn.manifold import TSNE

X_valid_compressed = stacked_encoder.predict(X_valid)
tsne = TSNE()

X_valid_2D = tsne.fit_transform(X_valid_compressed)

X_valid_2D = (X_valid_2D - X_valid_2D.min()) / (
    X_valid_2D.max() - X_valid_2D.min()
)

plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap="tab10")
plt.axis("off")
plt.show()

#################################################################
# Let's make this diagram a bit prettier:

# adapted from https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

plt.figure(figsize=(10, 8))
cmap = plt.cm.tab10

plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap=cmap)

image_positions = np.array([[1.0, 1.0]])

for index, position in enumerate(X_valid_2D):
    dist = np.sum((position - image_positions) ** 2, axis=1)
    if np.min(dist) > 0.02:  # if far enough from other images
        image_positions = np.r_[image_positions, [position]]
        imagebox = mpl.offsetbox.AnnotationBbox(
            mpl.offsetbox.OffsetImage(X_valid[index], cmap="binary"),
            position,
            bboxprops={"edgecolor": cmap(y_valid[index]), "lw": 2},
        )
        plt.gca().add_artist(imagebox)

plt.axis("off")
plt.show()
