import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from tensorflow import keras

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
np.random.seed(42)
tf.random.set_seed(42)


#################################################################
## Deep Convolutional GAN

tf.random.set_seed(42)
np.random.seed(42)

codings_size = 100

generator = keras.models.Sequential(
    [
        keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
        keras.layers.Reshape([7, 7, 128]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(
            64, kernel_size=5, strides=2, padding="SAME", activation="selu"
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(
            1, kernel_size=5, strides=2, padding="SAME", activation="tanh"
        ),
    ]
)

discriminator = keras.models.Sequential(
    [
        keras.layers.Conv2D(
            64,
            kernel_size=5,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
            input_shape=[28, 28, 1],
        ),
        keras.layers.Dropout(0.4),
        keras.layers.Conv2D(
            128,
            kernel_size=5,
            strides=2,
            padding="SAME",
            activation=keras.layers.LeakyReLU(0.2),
        ),
        keras.layers.Dropout(0.4),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
gan = keras.models.Sequential([generator, discriminator])


discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

# reshape and rescale
X_train_dcgan = X_train.reshape(-1, 28, 28, 1) * 2.0 - 1.0

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train_dcgan)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

train_gan(gan, dataset, batch_size, codings_size)

tf.random.set_seed(42)
np.random.seed(42)

noise = tf.random.normal(shape=[batch_size, codings_size])
generated_images = generator(noise)
plot_multiple_images(generated_images, 8)
