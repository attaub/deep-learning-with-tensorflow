from tensorflow import keras
import numpy as np
import tensorflow as tf


#################################################################
# Char-RNN

## Loading the Data and Preparing the Dataset

shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)

with open(filepath) as f:
    shakespeare_text = f.read()

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(shakespeare_text)
[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1


dataset_size = tokenizer.document_count  # total number of characters
train_size = dataset_size * 90 // 100

dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

n_steps = 100
window_length = n_steps + 1
dataset = dataset.window(window_length, shift=1, drop_remainder=True)

dataset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 32

dataset = dataset.shuffle(10000).batch(batch_size)

dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))


max_id = len(tokenizer.word_index)  # number of distinct characters

dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch)
)

dataset = dataset.prefetch(1)

for X_batch, Y_batch in dataset.take(1):
    print(X_batch.shape, Y_batch.shape)


model = keras.models.Sequential(
    [
        keras.layers.GRU(
            128,
            return_sequences=True,
            input_shape=[None, max_id],
            dropout=0.2,
        ),
        keras.layers.GRU(
            128,
            return_sequences=True,
            dropout=0.2,
        ),
        keras.layers.TimeDistributed(
            keras.layers.Dense(max_id, activation='softmax'),
        ),
    ]
)


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

history = model.fit(dataset, epochs=10)

# preprocess new text
def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)


def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]


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


dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
dataset = dataset.window()
