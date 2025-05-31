import sys
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images
import tensorflow as tf
from functools import partial

# ## LeNet-5
# The famous LeNet-5 architecture had the following layers:
#
# Layer  | Type            | Maps | Size     | Kernel size | Stride | Activation
# -------|-----------------|------|----------|-------------|--------|-----------
#  Out   | Fully connected | –    | 10       | –           | –      | RBF
#  F6    | Fully connected | –    | 84       | –           | –      | tanh
#  C5    | Convolution     | 120  | 1 × 1    | 5 × 5       | 1      | tanh
#  S4    | Avg pooling     | 16   | 5 × 5    | 2 × 2       | 2      | tanh
#  C3    | Convolution     | 16   | 10 × 10  | 5 × 5       | 1      | tanh
#  S2    | Avg pooling     | 6    | 14 × 14  | 2 × 2       | 2      | tanh
#  C1    | Convolution     | 6    | 28 × 28  | 5 × 5       | 1      | tanh
#  In    | Input           | 1    | 32 × 32  | –           | –      | –
#
# There were a few tweaks here and there, which don't really matter much anymore, but in case you are interested, here they are:
#
# * MNIST images are 28 × 28 pixels, but they are zero-padded to 32 × 32 pixels and normalized before being fed to the network. The rest of the network does not use any padding, which is why the size keeps shrinking as the image progresses through the network.
# * The average pooling layers are slightly more complex than usual: each neuron computes the mean of its inputs, then multiplies the result by a learnable coefficient (one per map) and adds a learnable bias term (again, one per map), then finally applies the activation function.
# * Most neurons in C3 maps are connected to neurons in only three or four S2 maps (instead of all six S2 maps). See table 1 (page 8) in the [original paper](https://homl.info/lenet5) for details.

# * The output layer is a bit special: instead of computing the matrix multiplication of the inputs and the weight vector, each neuron outputs the square of the Euclidian distance between its input vector and its weight vector. Each output measures how much the image belongs to a particular digit class. The cross-entropy cost function is now preferred, as it penalizes bad predictions much more, producing larger gradients and converging faster.

#################################################################
#################################################################
# # Implementing a ResNet-34 CNN Using Keras
DefaultConv2D = partial(
    tf.keras.layers.Conv2D,
    kernel_size=3,
    strides=1,
    padding="same",
    kernel_initializer="he_normal",
    use_bias=False,
)

#################################################################
class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization(),
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization(),
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


model = tf.keras.Sequential(
    [
        DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 3]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
    ]
)
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))
# # Using Pretrained Models from Keras
model = tf.keras.applications.ResNet50(weights="imagenet")

# **Warning**: The expression `load_sample_images()["images"]` returns a Python list of images. However, in the latest versions, Keras does not accept Python lists anymore, so we must convert this list to a tensor. We can do this using `tf.constant()`, but here I have used `K.constant()` instead: it simply calls `tf.constant()` if you are using TensorFlow as the backend (as is the case here), but if you ever decide to use JAX or PyTorch as the backend instead, `K.constant()` will call the appropriate function from the chosen backend.
K = tf.keras.backend
images = K.constant(load_sample_images()["images"])
images_resized = tf.keras.layers.Resizing(
    height=224, width=224, crop_to_aspect_ratio=True
)(images)
inputs = tf.keras.applications.resnet50.preprocess_input(images_resized)
Y_proba = model.predict(inputs)
Y_proba.shape
top_K = tf.keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
    print(f"Image #{image_index}")
    for class_id, name, y_proba in top_K[image_index]:
        print(f"  {class_id} - {name:12s} {y_proba:.2%}")
# extra code – displays the cropped and resized images
plt.figure(figsize=(10, 6))
for idx in (0, 1):
    plt.subplot(1, 2, idx + 1)
    plt.imshow(images_resized[idx] / 255)
    plt.axis("off")
plt.show()


