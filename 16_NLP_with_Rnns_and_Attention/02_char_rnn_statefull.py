from tensorflow import keras
import numpy as np
import tensorflow as tf


#################################################################
# Char-RNN
shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)

with open(filepath) as f:
    shakespeare_text = f.read()

print(shakespeare_text[:148])

"".join(sorted(set(shakespeare_text.lower())))

# tokenizer.get_config()
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(shakespeare_text)

tokenizer.texts_to_sequences(["First"])  # [[20, 6, 9, 8, 3]]
tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]])

max_id = len(tokenizer.word_index)  # number of distinct characters
dataset_size = tokenizer.document_count  # total number of characters

[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
train_size = dataset_size * 90 // 100
n_steps = 100
window_length = n_steps + 1  # target = input shifted 1 character ahead
#################################################################
## Stateful RNN
# create a dataset from the training portion of the encoded character IDs
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

# create sliding windows of length (n_steps + 1), shifted by n_steps, and drop incomplete windows
dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)

# flatten the dataset of window datasets into a dataset of tensors of shape (window_length,)
dataset = dataset.flat_map(lambda window: window.batch(window_length))

# batch each window individually so each item has shape (1, window_length)
# this is required for stateful RNNs to maintain sequence continuity across epochs
dataset = dataset.batch(1)

# split each window into input (first n_steps) and target (last n_steps, shifted by 1)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

# one-hot encode the input; the target remains as integer indices
dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch)
)

# prefetch 1 batch to improve pipeline performance
dataset = dataset.prefetch(1)

batch_size = 32  # Number of parallel sequences for training a stateful RNN

# Split the encoded training data into `batch_size` parts to maintain sequence continuity within each batch
encoded_parts = np.array_split(encoded[:train_size], batch_size)

datasets = []  # List to hold individual datasets for each batch

# For each partitioned sequence
for encoded_part in encoded_parts:
    # Create a dataset from the encoded part
    dataset = tf.data.Dataset.from_tensor_slices(encoded_part)

    # Create sliding windows of size (n_steps + 1) with shift=n_steps (non-overlapping)
    dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)

    # Convert each window dataset to a tensor of shape (window_length,)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))

    # Append this dataset to the list
    datasets.append(dataset)

# Zip all the datasets together, so each element is a tuple of sequences (one from each stream)

dataset = tf.data.Dataset.zip(tuple(datasets)).map(
    lambda *windows: tf.stack(windows)
    # Stack into a single tensor of shape (batch_size, window_length)
)

# split into input (first n_steps) and target (next n_steps) sequences

dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

# One-hot encode the input sequences, targets remain as integer labels
dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch)
)

# prefetch the next batch while the current one is being processed to improve performance
dataset = dataset.prefetch(1)
#################################################################
"""#Note: once again, I commented out recurrent_dropout=0.2 (compared to the book) so you can get GPU acceleration (if you have one)."""

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
