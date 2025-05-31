from tensorflow import keras 
import numpy as np 
import tensorflow as tf


#################################################################
# Char-RNN
shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()
# print(shakespeare_text[:148])
"".join(sorted(set(shakespeare_text.lower())))

# tokenizer.get_config()
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(shakespeare_text)
tokenizer.texts_to_sequences(["First"]) # [[20, 6, 9, 8, 3]]
tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]])

max_id = len(tokenizer.word_index)  # number of distinct characters
dataset_size = tokenizer.document_count  # total number of characters

[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
n_steps = 100

window_length = n_steps + 1  # target = input shifted 1 character ahead
dataset = dataset.window(window_length, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))

np.random.seed(42)
tf.random.set_seed(42)

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch)
)

dataset = dataset.prefetch(1)

for X_batch, Y_batch in dataset.take(1):
    print(X_batch.shape, Y_batch.shape)

#################################################################
## Creating and Training the Model

"""# Warning: the following code may take up to 24 hours to run, depending on your hardware.
#Note: the GRU class will only use the GPU (if you have one) when using the default values for the following arguments: activation, recurrent_activation, recurrent_dropout, unroll, use_bias and reset_after. This is why I commented out recurrent_dropout=0.2 (compared to the book).
"""

model = keras.models.Sequential(
    [
        keras.layers.GRU(
            128,
            return_sequences=True,
            input_shape=[None, max_id],
            # dropout=0.2, recurrent_dropout=0.2),
            dropout=0.2,
        ),
        keras.layers.GRU(
            128,
            return_sequences=True,
            # dropout=0.2, recurrent_dropout=0.2),
            dropout=0.2,
        ),
        keras.layers.TimeDistributed(
            keras.layers.Dense(max_id, activation="softmax")
        ),
    ]
)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

history = model.fit(dataset, epochs=10)

#################################################################
# Using the Model to Generate Text

def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)


# Warning: the predict_classes() method is deprecated. Instead, we must use `np.argmax(model(X_new), axis=-1)`.
# Y_pred = model.predict_classes(X_new)

X_new = preprocess(["How are yo"])
Y_pred = np.argmax(model(X_new), axis=-1)
tokenizer.sequences_to_texts(Y_pred + 1)[0][-1]  # 1st sentence, last char

tf.random.set_seed(42)

tf.random.categorical(
    [[np.log(0.5), np.log(0.4), np.log(0.1)]], num_samples=40
).numpy()


def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]


tf.random.set_seed(42)

next_char("How are yo", temperature=1)


def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


tf.random.set_seed(42)
print(complete_text("t", temperature=0.2))

print(complete_text("t", temperature=1))

print(complete_text("t", temperature=2))

###############################################################
## Stateful RNN
tf.random.set_seed(42)

dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))
dataset = dataset.batch(1)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch)
)

dataset = dataset.prefetch(1)

batch_size = 32

encoded_parts = np.array_split(encoded[:train_size], batch_size)

datasets = []

for encoded_part in encoded_parts:
    dataset = tf.data.Dataset.from_tensor_slices(encoded_part)
    dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    datasets.append(dataset)

dataset = tf.data.Dataset.zip(tuple(datasets)).map(
    lambda *windows: tf.stack(windows)
)

dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch)
)

dataset = dataset.prefetch(1)

#################################################################
"""#Note: once again, I commented out recurrent_dropout=0.2 (compared to the
book) so you can get GPU acceleration (if you have one)."""

model = keras.models.Sequential(
    [
        keras.layers.GRU(
            128,
            return_sequences=True,
            stateful=True,
            # dropout=0.2, recurrent_dropout=0.2,
            dropout=0.2,
            batch_input_shape=[batch_size, None, max_id],
        ),
        keras.layers.GRU(
            128,
            return_sequences=True,
            stateful=True,
            # dropout=0.2, recurrent_dropout=0.2),
            dropout=0.2,
        ),
        keras.layers.TimeDistributed(
            keras.layers.Dense(max_id, activation="softmax")
        ),
    ]
)

class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
history = model.fit(dataset, epochs=50, callbacks=[ResetStatesCallback()])

#################################################################
# To use the model with different batch sizes, we need to create a stateless copy. We can get rid of dropout since it is only used during training:


stateless_model = keras.models.Sequential(
    [
        keras.layers.GRU(
            128, return_sequences=True, input_shape=[None, max_id]
        ),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.TimeDistributed(
            keras.layers.Dense(max_id, activation="softmax")
        ),
    ]
)

# to set the weights, we first need to build the model (so the weights get created):

stateless_model.build(tf.TensorShape([None, None, max_id]))

stateless_model.set_weights(model.get_weights())

model = stateless_model

tf.random.set_seed(42)

print(complete_text("t"))
