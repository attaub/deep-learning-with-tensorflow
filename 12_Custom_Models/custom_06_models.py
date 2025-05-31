# ## Custom Models
X_new_scaled = X_test_scaled


class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [
            keras.layers.Dense(
                n_neurons, activation="elu", kernel_initializer="he_normal"
            )
            for _ in range(n_layers)
        ]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z


class ResidualRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(
            30, activation="elu", kernel_initializer="he_normal"
        )
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = ResidualRegressor(1)
model.compile(loss="mse", optimizer="nadam")
history = model.fit(X_train_scaled, y_train, epochs=5)
score = model.evaluate(X_test_scaled, y_test)
y_pred = model.predict(X_new_scaled)

model.save("my_custom_model.ckpt")

model = keras.models.load_model("my_custom_model.ckpt")
history = model.fit(X_train_scaled, y_train, epochs=5)

# We could have defined the model using the sequential API instead:


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

block1 = ResidualBlock(2, 30)
model = keras.models.Sequential(
    [
        keras.layers.Dense(
            30, activation="elu", kernel_initializer="he_normal"
        ),
        block1,
        block1,
        block1,
        block1,
        ResidualBlock(2, 30),
        keras.layers.Dense(1),
    ]
)

model.compile(loss="mse", optimizer="nadam")
history = model.fit(X_train_scaled, y_train, epochs=5)
score = model.evaluate(X_test_scaled, y_test)
y_pred = model.predict(X_new_scaled)


# ## Losses and Metrics Based on Model Internals

# **Note**: the following code has two differences with the code in the book:
# 1. It creates a `keras.metrics.Mean()` metric in the constructor and uses it in the `call()` method to track the mean reconstruction loss. Since we only want to do this during training, we add a `training` argument to the `call()` method, and if `training` is `True`, then we update `reconstruction_mean` and we call `self.add_metric()` to ensure it's displayed properly.
# 2. Due to an issue introduced in TF 2.2 ([#46858](https://github.com/tensorflow/tensorflow/issues/46858)), we must not call `super().build()` inside the `build()` method.


class ReconstructingRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [
            keras.layers.Dense(
                30, activation="selu", kernel_initializer="lecun_normal"
            )
            for _ in range(5)
        ]
        self.out = keras.layers.Dense(output_dim)
        self.reconstruction_mean = keras.metrics.Mean(
            name="reconstruction_error"
        )

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs)
        # super().build(batch_input_shape)

    def call(self, inputs, training=None):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        if training:
            result = self.reconstruction_mean(recon_loss)
            self.add_metric(result)
        return self.out(Z)


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = ReconstructingRegressor(1)
model.compile(loss="mse", optimizer="nadam")
history = model.fit(X_train_scaled, y_train, epochs=2)
y_pred = model.predict(X_test_scaled)

# ## Computing Gradients with Autodiff


def f(w1, w2):
    return 3 * w1**2 + 2 * w1 * w2


w1, w2 = 5, 3
eps = 1e-6
(f(w1 + eps, w2) - f(w1, w2)) / eps
(f(w1, w2 + eps) - f(w1, w2)) / eps

w1, w2 = tf.Variable(5.0), tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
gradients

with tf.GradientTape() as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)
try:
    dz_dw2 = tape.gradient(z, w2)
except RuntimeError as ex:
    print(ex)

with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)
dz_dw2 = tape.gradient(z, w2)  # works now!
del tape

dz_dw1, dz_dw2

c1, c2 = tf.constant(5.0), tf.constant(3.0)
with tf.GradientTape() as tape:
    z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2])
gradients


with tf.GradientTape() as tape:
    tape.watch(c1)
    tape.watch(c2)
    z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2])
gradients


with tf.GradientTape() as tape:
    z1 = f(w1, w2 + 2.0)
    z2 = f(w1, w2 + 5.0)
    z3 = f(w1, w2 + 7.0)

tape.gradient([z1, z2, z3], [w1, w2])


with tf.GradientTape(persistent=True) as tape:
    z1 = f(w1, w2 + 2.0)
    z2 = f(w1, w2 + 5.0)
    z3 = f(w1, w2 + 7.0)

tf.reduce_sum(
    tf.stack([tape.gradient(z, [w1, w2]) for z in (z1, z2, z3)]), axis=0
)
del tape


with tf.GradientTape(persistent=True) as hessian_tape:
    with tf.GradientTape() as jacobian_tape:
        z = f(w1, w2)
    jacobians = jacobian_tape.gradient(z, [w1, w2])
hessians = [
    hessian_tape.gradient(jacobian, [w1, w2]) for jacobian in jacobians
]
del hessian_tape
jacobians
hessians


def f(w1, w2):
    return 3 * w1**2 + tf.stop_gradient(2 * w1 * w2)


with tf.GradientTape() as tape:
    z = f(w1, w2)

tape.gradient(z, [w1, w2])
x = tf.Variable(100.0)
with tf.GradientTape() as tape:
    z = my_softplus(x)

tape.gradient(z, [x])
tf.math.log(tf.exp(tf.constant(30.0, dtype=tf.float32)) + 1.0)
x = tf.Variable([100.0])
with tf.GradientTape() as tape:
    z = my_softplus(x)

tape.gradient(z, [x])


@tf.custom_gradient
def my_better_softplus(z):
    exp = tf.exp(z)

    def my_softplus_gradients(grad):
        return grad / (1 + 1 / exp)

    return tf.math.log(exp + 1), my_softplus_gradients


def my_better_softplus(z):
    return tf.where(z > 30.0, z, tf.math.log(tf.exp(z) + 1.0))


x = tf.Variable([1000.0])
with tf.GradientTape() as tape:
    z = my_better_softplus(x)

z, tape.gradient(z, [x])
