# Sentiment Analysis
import os
import sklearn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from collections import Counter
import tensorflow_hub as hub
import tensorflow_datasets as tfds

imdb_data = keras.datasets.imdb.load_data()
data_train, data_test = imdb_data
X_train, y_train = data_train
X_test, y_test = data_test
X_train[0][:10]

word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
    id_to_word[id_] = token
" ".join([id_to_word[id_] for id_ in X_train[0][:10]])

#################################################################
datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
datasets.keys()  # get dictionary keys of dataset splits (e.g., 'train', 'test')
train_size = info.splits["train"].num_examples
test_size = info.splits["test"].num_examples
print(train_size, test_size)

for X_batch, y_batch in datasets["train"].batch(2).take(1):
    for review, label in zip(X_batch.numpy(), y_batch.numpy()):
        print("Review:", review.decode("utf-8")[:200], "...")
        print("Label:", label, "= Positive" if label else "= Negative")
        print("\n\n")


def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch


preprocess(X_batch, y_batch)

vocabulary = Counter()
i = 0
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    print(f"Batch: {i}")
    for review in X_batch:
        vocabulary.update(list(review.numpy()))

vocabulary.most_common()[:3]
len(vocabulary)

vocab_size = 10000
truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]
]

word_to_id = {word: index for index, word in enumerate(truncated_vocabulary)}

for word in b"This movie was faaaaaantastic".split():
    print(word_to_id.get(word) or vocab_size)

# convert the truncated vocabulary list into a TensorFlow constant
words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)
table.lookup(tf.constant([b"This movie was faaaaaantastic".split()]))


def encode_words(X_batch, y_batch):
    # replace each word in the batch with its corresponding ID using the lookup table
    return table.lookup(X_batch), y_batch


train_set = datasets["train"].batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

for X_batch, y_batch in train_set.take(1):
    print(X_batch)
    print(y_batch)

embed_size = 128

model = keras.models.Sequential(
    [
        keras.layers.Embedding(
            vocab_size + num_oov_buckets,
            embed_size,
            mask_zero=True,  # not shown in the book
            input_shape=[None],
        ),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.GRU(128),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)
history = model.fit(train_set, epochs=5)

#################################################################
# or using manual masking:

K = keras.backend
embed_size = 128
inputs = keras.layers.Input(shape=[None])
mask = keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
z = keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size)(inputs)
z = keras.layers.GRU(128, return_sequences=True)(z, mask=mask)
z = keras.layers.GRU(128)(z, mask=mask)
outputs = keras.layers.Dense(1, activation="sigmoid")(z)
model = keras.models.Model(inputs=[inputs], outputs=[outputs])
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)
history = model.fit(train_set, epochs=5)

#################################################################
## Reusing Pretrained Embeddings
tf.random.set_seed(42)
TFHUB_CACHE_DIR = os.path.join(os.curdir, "my_tfhub_cache")
os.environ["TFHUB_CACHE_DIR"] = TFHUB_CACHE_DIR

model = keras.Sequential(
    [
        hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
            dtype=tf.string,
            input_shape=[],
            output_shape=[50],
        ),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

for dirpath, dirnames, filenames in os.walk(TFHUB_CACHE_DIR):
    for filename in filenames:
        print(os.path.join(dirpath, filename))

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples
batch_size = 32
train_set = datasets["train"].batch(batch_size).prefetch(1)
history = model.fit(train_set, epochs=5)
