# Custom loss function
def my_mse(y_true, y_pred):
    print("Tracing loss my_mse()")
    return tf.reduce_mean(tf.square(y_pred - y_true))


# Custom metric function
def my_mae(y_true, y_pred):
    print("Tracing metric my_mae()")
    return tf.reduce_mean(tf.abs(y_pred - y_true))


# Custom layer
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[1], self.units),
            initializer='uniform',
            trainable=True,
        )
        self.biases = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, X):
        print("Tracing MyDense.call()")
        return self.activation(X @ self.kernel + self.biases)


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


# Custom model
class MyModel(keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = MyDense(30, activation="relu")
        self.hidden2 = MyDense(30, activation="relu")
        self.output_ = MyDense(1)

    def call(self, input):
        print("Tracing MyModel.call()")
        hidden1 = self.hidden1(input)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input, hidden2])
        output = self.output_(concat)
        return output


model = MyModel()


model.compile(loss=my_mse, optimizer="nadam", metrics=[my_mae])


model.fit(
    X_train_scaled,
    y_train,
    epochs=2,
    validation_data=(X_valid_scaled, y_valid),
)
model.evaluate(X_test_scaled, y_test)


# You can turn this off by creating the model with `dynamic=True` (or calling `super().__init__(dynamic=True, **kwargs)` in the model's constructor):


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


model = MyModel(dynamic=True)


model.compile(loss=my_mse, optimizer="nadam", metrics=[my_mae])


# Not the custom code will be called at each iteration. Let's fit, validate and evaluate with tiny datasets to avoid getting too much output:


model.fit(
    X_train_scaled[:64],
    y_train[:64],
    epochs=1,
    validation_data=(X_valid_scaled[:64], y_valid[:64]),
    verbose=0,
)
model.evaluate(X_test_scaled[:64], y_test[:64], verbose=0)


# Alternatively, you can compile a model with `run_eagerly=True`:


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


model = MyModel()


model.compile(
    loss=my_mse, optimizer="nadam", metrics=[my_mae], run_eagerly=True
)


model.fit(
    X_train_scaled[:64],
    y_train[:64],
    epochs=1,
    validation_data=(X_valid_scaled[:64], y_valid[:64]),
    verbose=0,
)
model.evaluate(X_test_scaled[:64], y_test[:64], verbose=0)
