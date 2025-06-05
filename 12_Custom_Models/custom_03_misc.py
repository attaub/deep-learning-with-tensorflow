# ## Other Custom Functions


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


def my_softplus(z):  # return value is just tf.nn.softplus(z)
    return tf.math.log(tf.exp(z) + 1.0)


def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2.0 / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)


def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))


def my_positive_weights(weights):  # return value is just tf.nn.relu(weights)
    return tf.where(weights < 0.0, tf.zeros_like(weights), weights)


layer = keras.layers.Dense(
    1,
    activation=my_softplus,
    kernel_initializer=my_glorot_initializer,
    kernel_regularizer=my_l1_regularizer,
    kernel_constraint=my_positive_weights,
)


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


model = keras.models.Sequential(
    [
        keras.layers.Dense(
            30,
            activation="selu",
            kernel_initializer="lecun_normal",
            input_shape=input_shape,
        ),
        keras.layers.Dense(
            1,
            activation=my_softplus,
            kernel_regularizer=my_l1_regularizer,
            kernel_constraint=my_positive_weights,
            kernel_initializer=my_glorot_initializer,
        ),
    ]
)


model.compile(loss="mse", optimizer="nadam", metrics=["mae"])


model.fit(
    X_train_scaled,
    y_train,
    epochs=2,
    validation_data=(X_valid_scaled, y_valid),
)


model.save("my_model_with_many_custom_parts.h5")


model = keras.models.load_model(
    "my_model_with_many_custom_parts.h5",
    custom_objects={
        "my_l1_regularizer": my_l1_regularizer,
        "my_positive_weights": my_positive_weights,
        "my_glorot_initializer": my_glorot_initializer,
        "my_softplus": my_softplus,
    },
)


class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))

    def get_config(self):
        return {"factor": self.factor}


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


model = keras.models.Sequential(
    [
        keras.layers.Dense(
            30,
            activation="selu",
            kernel_initializer="lecun_normal",
            input_shape=input_shape,
        ),
        keras.layers.Dense(
            1,
            activation=my_softplus,
            kernel_regularizer=MyL1Regularizer(0.01),
            kernel_constraint=my_positive_weights,
            kernel_initializer=my_glorot_initializer,
        ),
    ]
)


model.compile(loss="mse", optimizer="nadam", metrics=["mae"])


model.fit(
    X_train_scaled,
    y_train,
    epochs=2,
    validation_data=(X_valid_scaled, y_valid),
)


model.save("my_model_with_many_custom_parts.h5")


model = keras.models.load_model(
    "my_model_with_many_custom_parts.h5",
    custom_objects={
        "MyL1Regularizer": MyL1Regularizer,
        "my_positive_weights": my_positive_weights,
        "my_glorot_initializer": my_glorot_initializer,
        "my_softplus": my_softplus,
    },
)
