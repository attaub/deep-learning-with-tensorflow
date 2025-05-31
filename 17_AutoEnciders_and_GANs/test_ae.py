import sklearn
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from utils import rounded_accuracy, plot_image, show_reconstructions
import matplotlib as mpl


np.random.seed(42)


#################################################################
# simple autoencoder
def generate_3d_data(m, w1=0.1, w2=0.2, noise=0.1):

    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5  # theta_r * 3pi/2 - 0.5

    # X1: cos(t) + sin(t)/2 + eps*X_m/2
    X1 = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2

    # X2= sin(t) *0.7 + eps*X_m/2
    X2 = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2

    # X3 = w1X1+w2X2 + eps*X_rand
    X3 = X1 * w1 + X2 * w2 + noise * np.random.randn(m)

    data = np.empty((m, 3))
    data[:, 0] = X1
    data[:, 1] = X2
    data[:, 2] = X3
    return data


m = 1000
X_train = generate_3d_data(m)
X_train = X_train - X_train.mean(axis=0, keepdims=0)


np.random.seed(42)
tf.random.set_seed(42)


encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])])
decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])

autoencoder = keras.models.Sequential([encoder, decoder])

# optimizer, loss function, epochs
autoencoder.compile(
    loss='mse', optimizer=keras.optimizers.SGD(learning_rate=1.5)
)

history = autoencoder.fit(X_train, X_train, epochs=20)


codings = encoder.predict(X_train)

plt.figure()
plt.plot(codings[:, 0], codings[:, 1], ".b")
plt.plot(X_train[:, 0], X_train[:, 1], "r.")

plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
plt.show()


#################################################################
### Stacked autoencoder

# Load the dataset
train_data, test_data = keras.datasets.fashion_mnist.load_data()
X_train_full, y_train_full = train_data
X_test, y_test = test_data

# Scale data
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

# Split into training and validation sets
val_idx = -5000
X_train, X_valid = X_train_full[:val_idx], X_train_full[val_idx:]
y_train, y_valid = y_train_full[:val_idx], y_train_full[val_idx:]

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Build stacked autoencoder
stacked_encoder = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(100, activation='selu'),
        keras.layers.Dense(30, activation='selu'),
    ]
)

stacked_decoder = keras.models.Sequential(
    [
        keras.layers.Dense(100, activation='selu', input_shape=[30]),
        keras.layers.Dense(28 * 28, activation='sigmoid'),
        keras.layers.Reshape([28, 28]),
    ]
)

stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])

stacked_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1.5),
    metrics=[rounded_accuracy],
)

# Train model
history = stacked_ae.fit(
    X_train, X_train, epochs=20, validation_data=(X_valid, X_valid)
)

# Show reconstructions
show_reconstructions(stacked_ae, X_valid, n_images=12)
plt.show()

#################################################################
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = activation
        super.__init__(**kwargs)

    def build():
        self.bias = self.add_weight(
            name="bias",
            shape=[self.dense.input_shape[-1]],
            initializer="zeros",
        )
        super().build(batch_input_shape)

    def call():
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)


keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

dense_1 = keras.layers.Dense(100, activation="selu")
dense_2 = keras.layers.Dense(30, activation="selu")

tied_encoder = keras.models.Sequential(
    [
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        dense_1,
        dense_2,
    ]
)

tied_decoder = keras.models.Sequential(
    [
        DenseTranspose(dense_2, activation="selu"),
        DenseTranspose(dense_1, activation="sigmoid"),
        keras.layers.Reshape([28, 28]),
    ]
)

tied_ae = keras.models.Sequential([tied_encoder, tied_decoder])


tied_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1.5),
    metrics=[rounded_accuracy],
)

history = tied_ae.fit(
    X_train, X_train, epochs=10, validation_data=(X_valid, X_valid)
)
show_reconstructions(tied_ae)
plt.show()

#################################################################


conv_encoder = keras.models.Sequential(
    [
        keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
        keras.layers.Conv2D(
            16, kernel_size=3, padding="SAME", activation="selu"
        ),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            32, kernel_size=3, padding="SAME", activation="selu"
        ),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(
            64, kernel_size=3, padding="SAME", activation="selu"
        ),
        keras.layers.MaxPool2D(pool_size=2),
    ]
)

conv_decoder = keras.models.Sequential(
    [
        keras.layers.Conv2DTranspose(
            32,
            kernel_size=3,
            strides=2,
            padding="VALID",
            activation="selu",
            input_shape=[3, 3, 64],
        ),
        keras.layers.Conv2DTranspose(
            16, kernel_size=3, strides=2, padding="SAME", activation="selu"
        ),
        keras.layers.Conv2DTranspose(
            1, kernel_size=3, strides=2, padding="SAME", activation="sigmoid"
        ),
        keras.layers.Reshape([28, 28]),
    ]
)

conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

conv_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1.0),
    metrics=[rounded_accuracy],
)

history = conv_ae.fit(
    X_train, X_train, epochs=5, validation_data=(X_valid, X_valid)
)

conv_encoder.summary()
conv_decoder.summary()
conv_ae.summary()

# def show_reconstructions(model, images=X_valid, n_images=5):
show_reconstructions(conv_ae, X_valid)
plt.show()

#################################################################
# Recurrent Autoencoders
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
#################################################################
# Denoising Autoencoders

denoising_encoder = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.GaussianNoise(0.2),
        keras.layers.Dense(100, activation="selu"),
        keras.layers.Dense(30, activation="selu"),
    ]
)
denoising_decoder = keras.models.Sequential(
    [
        keras.layers.Dense(100, activation="selu", input_shape=[30]),
        keras.layers.Dense(28 * 28, activation="sigmoid"),
        keras.layers.Reshape([28, 28]),
    ]
)

#################################################################
# Dropout Autoencoders
dropout_encoder = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(100, activation="selu"),
        keras.layers.Dense(30, activation="selu"),
    ]
)

dropout_decoder = keras.models.Sequential(
    [
        keras.layers.Dense(100, activation="selu", input_shape=[30]),
        keras.layers.Dense(28 * 28, activation="sigmoid"),
        keras.layers.Reshape([28, 28]),
    ]
)

#################################################################
# Sparse Autoencoders

# l1

sparse_l1_encoder = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(100, activation="selu"),
        keras.layers.Dense(300, activation="sigmoid"),
        keras.layers.ActivityRegularization(l1=1e-3),
    ]
)

sparse_l1_decoder = keras.models.Sequential(
    [
        keras.layers.Dense(100, activation="selu", input_shape=[300]),
        keras.layers.Dense(28 * 28, activation="sigmoid"),
        keras.layers.Reshape([28, 28]),
    ]
)


# kld

""" 
class KLDivergenceRegularizer(keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target

    def __call__(self, inputs):
        mean_activities = K.mean(inputs, axis=0)
        return self.weight * (
            kl_divergence(self.target, mean_activities)
            + kl_divergence(1.0 - self.target, 1.0 - mean_activities)
        )
"""

kld_reg = KLDivergenceRegularizer(weight=0.05, target=0.1)
sparse_kl_encoder = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(100, activation="selu"),
        keras.layers.Dense(
            300, activation="sigmoid", activity_regularizer=kld_reg
        ),
    ]
)

sparse_kl_decoder = keras.models.Sequential(
    [
        keras.layers.Dense(100, activation="selu", input_shape=[300]),
        keras.layers.Dense(28 * 28, activation="sigmoid"),
        keras.layers.Reshape([28, 28]),
    ]
)


#################################################################
# Variational Autoencoders


class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean


codings_size = 10

inputs = keras.layers.Input(shape=[28, 28])

z = keras.layers.Flatten()(inputs)

z = keras.layers.Dense(150, activation="selu")(z)

z = keras.layers.Dense(100, activation="selu")(z)

codings_mean = keras.layers.Dense(codings_size)(z)

codings_log_var = keras.layers.Dense(codings_size)(z)

codings = Sampling()([codings_mean, codings_log_var])

variational_encoder = keras.models.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings]
)

decoder_inputs = keras.layers.Input(shape=[codings_size])
x = keras.layers.Dense(100, activation="selu")(decoder_inputs)
x = keras.layers.Dense(150, activation="selu")(x)
x = keras.layers.Dense(28 * 28, activation="sigmoid")(x)
outputs = keras.layers.Reshape([28, 28])(x)
variational_decoder = keras.models.Model(
    inputs=[decoder_inputs], outputs=[outputs]
)

_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = keras.models.Model(inputs=[inputs], outputs=[reconstructions])

latent_loss = -0.5 * K.sum(
    1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
    axis=-1,
)

variational_ae.add_loss(K.mean(latent_loss) / 784.0)

variational_ae.compile(
    loss="binary_crossentropy", optimizer="rmsprop", metrics=[rounded_accuracy]
)

history = variational_ae.fit(
    X_train,
    X_train,
    epochs=25,
    batch_size=128,
    validation_data=(X_valid, X_valid),
)


#################################################################
