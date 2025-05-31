import sys
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

from sklearn.datasets import load_sample_images
import tensorflow as tf

images = load_sample_images()["images"]
images = tf.keras.layers.CenterCrop(height=70, width=120)(images)
images = tf.keras.layers.Rescaling(scale=1 / 255)(images)

images.shape

tf.random.set_seed(42)
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7)

fmaps = conv_layer(images)

fmaps.shape

# extra code – displays the two output feature maps for each image
plt.figure(figsize=(15, 9))
for image_idx in (0, 1):
    for fmap_idx in (0, 1):
        plt.subplot(2, 2, image_idx * 2 + fmap_idx + 1)
        plt.imshow(fmaps[image_idx, :, :, fmap_idx], cmap="gray")
        plt.axis("off")
plt.show()

"""# As you can see, randomly generated filters typically act like edge
detectors, which is great since that's a useful tool in image processing, and
that's the type of filters that a convolutional layer typically starts
with. Then, during training, it gradually learns improved filters to recognize
useful patterns for the task."""

# Now let's use zero-padding:
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding="same")

fmaps = conv_layer(images)
fmaps.shape

# extra code – shows that the output shape when we set strides=2
conv_layer = tf.keras.layers.Conv2D(
    filters=32, kernel_size=7, padding="same", strides=2
)

fmaps = conv_layer(images)
fmaps.shape

"""# Extra code – This utility function can be useful to compute the size of the
feature maps output by a convolutional layer.
It also returns the number of ignored rows or columns if
padding="valid", or the number of zero-padded rows
or columns if padding='same'. """

import numpy as np


def conv_output_size(input_size, kernel_size, strides=1, padding="valid"):
    if padding == "valid":
        z = input_size - kernel_size + strides
        output_size = z // strides
        num_ignored = z % strides
        return output_size, num_ignored
    else:
        output_size = (input_size - 1) // strides + 1
        num_padded = (output_size - 1) * strides + kernel_size - input_size
        return output_size, num_padded


conv_output_size(np.array([70, 120]), kernel_size=7, strides=2, padding="same")

# Let's now look at the weights:
kernels, biases = conv_layer.get_weights()
kernels.shape
biases.shape

"""# extra code – shows how to use the tf.nn.conv2d() operation """

tf.random.set_seed(42)
filters = tf.random.normal([7, 7, 3, 2])
biases = tf.zeros([2])
fmaps = tf.nn.conv2d(images, filters, strides=1, padding="SAME") + biases
fmaps.shape

"""# Manually create two filters full of zeros, except for a vertical line of 1s in the first filter, and a horizontal one in the second filter (just like in Figure 14–5).
The two output feature maps highlight vertical lines and horizontal lines, respectively.
In practice you will probably never need to create filters manually, since the convolutional layers will learn them automatically.
extra code. 
"""

plt.figure(figsize=(15, 9))
filters = np.zeros([7, 7, 3, 2])
filters[:, 3, :, 0] = 1
filters[3, :, :, 1] = 1
fmaps = tf.nn.conv2d(images, filters, strides=1, padding="SAME") + biases
for image_idx in (0, 1):
    for fmap_idx in (0, 1):
        plt.subplot(2, 2, image_idx * 2 + fmap_idx + 1)
        plt.imshow(fmaps[image_idx, :, :, fmap_idx], cmap="gray")
        plt.axis("off")
plt.show()

""" Notice the dark lines at the top and bottom of the two images on the left, and on the left and right of the two images on the right? Can you guess what these are? Why were they not present in the previous figure?

You guessed it! These are artifacts due to the fact that we used zero padding in this case, while we did not use zero padding to create the feature maps in the previous figure. Because of zero padding, the two feature maps based on the vertical line filter (i.e., the two left images) could not fully activate near the top and bottom of the images. Similarly, the two feature maps based on the horizontal line filter (i.e., the two right images) could not fully activate near the left and right of the images.

"""

