# # Custom Training Loops
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

l2_reg = keras.regularizers.l2(0.05)
model = keras.models.Sequential(
    [
        keras.layers.Dense(
            30,
            activation="elu",
            kernel_initializer="he_normal",
            kernel_regularizer=l2_reg,
        ),
        keras.layers.Dense(1, kernel_regularizer=l2_reg),
    ]
)


def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(
        [
            "{}: {:.4f}".format(m.name, m.result())
            for m in [loss] + (metrics or [])
        ]
    )
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics, end=end)


import time

mean_loss = keras.metrics.Mean(name="loss")
mean_square = keras.metrics.Mean(name="mean_square")
for i in range(1, 50 + 1):
    loss = 1 / i
    mean_loss(loss)
    mean_square(i**2)
    print_status_bar(i, 50, mean_loss, [mean_square])
    time.sleep(0.05)


# A fancier version with a progress bar:


def progress_bar(iteration, total, size=30):
    running = iteration < total
    c = ">" if running else "="
    p = (size - 1) * iteration // total
    fmt = "{{:-{}d}}/{{}} [{{}}]".format(len(str(total)))
    params = [iteration, total, "=" * p + c + "." * (size - p - 1)]
    return fmt.format(*params)


progress_bar(3500, 10000, size=6)


def print_status_bar(iteration, total, loss, metrics=None, size=30):
    metrics = " - ".join(
        [
            "{}: {:.4f}".format(m.name, m.result())
            for m in [loss] + (metrics or [])
        ]
    )
    end = "" if iteration < total else "\n"
    print("\r{} - {}".format(progress_bar(iteration, total), metrics), end=end)


mean_loss = keras.metrics.Mean(name="loss")
mean_square = keras.metrics.Mean(name="mean_square")
for i in range(1, 50 + 1):
    loss = 1 / i
    mean_loss(loss)
    mean_square(i**2)
    print_status_bar(i, 50, mean_loss, [mean_square])
    time.sleep(0.05)


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = keras.losses.mean_squared_error
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.MeanAbsoluteError()]


for epoch in range(1, n_epochs + 1):
    print("Epoch {}/{}".format(epoch, n_epochs))
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train_scaled, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for variable in model.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))
        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)
        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_states()


try:
    from tqdm.notebook import trange
    from collections import OrderedDict

    with trange(1, n_epochs + 1, desc="All epochs") as epochs:
        for epoch in epochs:
            with trange(
                1, n_steps + 1, desc="Epoch {}/{}".format(epoch, n_epochs)
            ) as steps:
                for step in steps:
                    X_batch, y_batch = random_batch(X_train_scaled, y_train)
                    with tf.GradientTape() as tape:
                        y_pred = model(X_batch)
                        main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                        loss = tf.add_n([main_loss] + model.losses)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(gradients, model.trainable_variables)
                    )
                    for variable in model.variables:
                        if variable.constraint is not None:
                            variable.assign(variable.constraint(variable))
                    status = OrderedDict()
                    mean_loss(loss)
                    status["loss"] = mean_loss.result().numpy()
                    for metric in metrics:
                        metric(y_batch, y_pred)
                        status[metric.name] = metric.result().numpy()
                    steps.set_postfix(status)
            for metric in [mean_loss] + metrics:
                metric.reset_states()
except ImportError as ex:
    print(
        "To run this cell, please install tqdm, ipywidgets and restart Jupyter"
    )


# ## TensorFlow Functions


def cube(x):
    return x**3


cube(2)


cube(tf.constant(2.0))


tf_cube = tf.function(cube)
tf_cube


tf_cube(2)


tf_cube(tf.constant(2.0))


# ### TF Functions and Concrete Functions


concrete_function = tf_cube.get_concrete_function(tf.constant(2.0))
concrete_function.graph


concrete_function(tf.constant(2.0))


concrete_function is tf_cube.get_concrete_function(tf.constant(2.0))


# ### Exploring Function Definitions and Graphs


concrete_function.graph


ops = concrete_function.graph.get_operations()
ops


pow_op = ops[2]
list(pow_op.inputs)


pow_op.outputs


concrete_function.graph.get_operation_by_name('x')


concrete_function.graph.get_tensor_by_name('Identity:0')


concrete_function.function_def.signature


# ### How TF Functions Trace Python Functions to Extract Their Computation Graphs


@tf.function
def tf_cube(x):
    print("print:", x)
    return x**3


result = tf_cube(tf.constant(2.0))


result


result = tf_cube(2)
result = tf_cube(3)
result = tf_cube(tf.constant([[1.0, 2.0]]))  # New shape: trace!
result = tf_cube(tf.constant([[3.0, 4.0], [5.0, 6.0]]))  # New shape: trace!
result = tf_cube(
    tf.constant([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
)  # New shape: trace!


# It is also possible to specify a particular input signature:


@tf.function(input_signature=[tf.TensorSpec([None, 28, 28], tf.float32)])
def shrink(images):
    print("Tracing", images)
    return images[:, ::2, ::2]  # drop half the rows and columns


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


img_batch_1 = tf.random.uniform(shape=[100, 28, 28])
img_batch_2 = tf.random.uniform(shape=[50, 28, 28])
preprocessed_images = shrink(img_batch_1)  # Traces the function.
preprocessed_images = shrink(img_batch_2)  # Reuses the same concrete function.


img_batch_3 = tf.random.uniform(shape=[2, 2, 2])
try:
    preprocessed_images = shrink(
        img_batch_3
    )  # rejects unexpected types or shapes
except ValueError as ex:
    print(ex)


# ### Using Autograph To Capture Control Flow

# A "static" `for` loop using `range()`:


@tf.function
def add_10(x):
    for i in range(10):
        x += 1
    return x


add_10(tf.constant(5))


add_10.get_concrete_function(tf.constant(5)).graph.get_operations()


# A "dynamic" loop using `tf.while_loop()`:


@tf.function
def add_10(x):
    condition = lambda i, x: tf.less(i, 10)
    body = lambda i, x: (tf.add(i, 1), tf.add(x, 1))
    final_i, final_x = tf.while_loop(condition, body, [tf.constant(0), x])
    return final_x


add_10(tf.constant(5))


add_10.get_concrete_function(tf.constant(5)).graph.get_operations()


# A "dynamic" `for` loop using `tf.range()` (captured by autograph):


@tf.function
def add_10(x):
    for i in tf.range(10):
        x = x + 1
    return x


add_10.get_concrete_function(tf.constant(0)).graph.get_operations()


# ### Handling Variables and Other Resources in TF Functions


counter = tf.Variable(0)


@tf.function
def increment(counter, c=1):
    return counter.assign_add(c)


increment(counter)
increment(counter)


function_def = increment.get_concrete_function(counter).function_def
function_def.signature.input_arg[0]


counter = tf.Variable(0)


@tf.function
def increment(c=1):
    return counter.assign_add(c)


increment()
increment()


function_def = increment.get_concrete_function().function_def
function_def.signature.input_arg[0]


class Counter:
    def __init__(self):
        self.counter = tf.Variable(0)

    @tf.function
    def increment(self, c=1):
        return self.counter.assign_add(c)


c = Counter()
c.increment()
c.increment()


@tf.function
def add_10(x):
    for i in tf.range(10):
        x += 1
    return x


print(tf.autograph.to_code(add_10.python_function))


def display_tf_code(func):
    from IPython.display import display, Markdown

    if hasattr(func, "python_function"):
        func = func.python_function
    code = tf.autograph.to_code(func)
    display(Markdown('```python\n{}\n```'.format(code)))


display_tf_code(add_10)


# ## Using TF Functions with tf.keras (or Not)

# By default, tf.keras will automatically convert your custom code into TF Functions, no need to use
# `tf.function()`:
