import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

with open("./spa.txt", "r") as f:
    text = f.read()

text = text.replace("¡", "").replace("¿", "")
pairs = [line.split("\t") for line in text.splitlines()]

np.random.seed(42)
np.random.shuffle(pairs)

sentences_en, sentences_es = zip(*pairs)

for i in range(3):
    print(sentences_en[i], "=>", sentences_es[i])

# define vocabulary size for text vectorization (maximum number of unique words)
# and define the maximum sequence length for input/output sequences

vocab_size = 1000
max_length = 50

# create a TextVectorization layers, limiting to vocab_size and max_length
text_vec_layer_en = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length
)

text_vec_layer_es = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length
)

# adapt the vectorization layers to the sentences to build its vocabulary

text_vec_layer_en.adapt(sentences_en)
text_vec_layer_es.adapt([f"startofseq {s} endofseq" for s in sentences_es])

text_vec_layer_en.get_vocabulary()[:10]
text_vec_layer_es.get_vocabulary()[:10]

X_train = tf.constant(sentences_en[:100_000])
X_valid = tf.constant(sentences_en[100_000:])

# training and validation decoder inputs
X_train_dec = tf.constant([f"startofseq {s}" for s in sentences_es[:100_000]])
X_valid_dec = tf.constant([f"startofseq {s}" for s in sentences_es[100_000:]])

# vectorize the spanish training and validationsentences
Y_train = text_vec_layer_es([f"{s} endofseq" for s in sentences_es[:100_000]])
Y_valid = text_vec_layer_es([f"{s} endofseq" for s in sentences_es[100_000:]])

tf.random.set_seed(42)

# define the encoder and decoder input layer, expecting a string tensor with no fixed shape
encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

embed_size = 128

# convert input strings to token IDs
encoder_input_ids = text_vec_layer_en(encoder_inputs)
decoder_input_ids = text_vec_layer_es(decoder_inputs)

# create an embedding layer for the encoder and decoder
encoder_embedding_layer = tf.keras.layers.Embedding(
    vocab_size, embed_size, mask_zero=True
)

decoder_embedding_layer = tf.keras.layers.Embedding(
    vocab_size, embed_size, mask_zero=True
)

# convert encoder input IDs to dense embeddings
encoder_embeddings = encoder_embedding_layer(encoder_input_ids)

# convert decoder input IDs to dense embeddings
decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

# LSTM layer for the encoder with 512 units, returning final states
encoder = tf.keras.layers.LSTM(512, return_state=True)


# pass encoder embeddings through the LSTM, capturing outputs and states
encoder_outputs, *encoder_state = encoder(encoder_embeddings)

# define an LSTM layer for the decoder with 512 units, returning full sequences
decoder = tf.keras.layers.LSTM(512, return_sequences=True)

# pass decoder embeddings through the LSTM, using encoder states as initial states
decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)

# define a dense output layer with vocab_size units and softmax activation
output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")

# generate probability distributions over the vocabulary for each decoder output
Y_proba = output_layer(decoder_outputs)

model = tf.keras.Model(
    inputs=[encoder_inputs, decoder_inputs], outputs=[Y_proba]
)

model = tf.keras.Model(
    inputs=[encoder_inputs, decoder_inputs], outputs=Y_proba
)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="nadam",
    metrics=["accuracy"],
)

model.fit(
    (X_train, X_train_dec),
    Y_train,
    epochs=10,
    validation_data=((X_valid, X_valid_dec), Y_valid),
)
