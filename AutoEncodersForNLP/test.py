import numpy as np
import tensorflow as tf

with open("./spa.txt", "r") as f:
    text = f.read()
text = text.replace("¡", "").replace("¿", "")
pairs = [line.split("\t") for line in text.splitlines()]

np.random.seed(42)
np.random.shuffle(pairs)

sentences_en, sentences_es = zip(*pairs)

vocab_size = 1000
max_length = 50
embed_size = 128

# text vectorization
text_vec_layer_en = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length
)

text_vec_layer_es = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length
)

text_vec_layer_en.adapt(sentences_en)
text_vec_layer_es.adapt([f"startofseq {s} endofseq" for s in sentences_es])

# train encoder on english sentences
X_train = tf.constant(sentences_en[:100_000])
X_valid = tf.constant(sentences_en[100_000:])

# decoder inputs (spanish sentences)
X_train_dec = tf.constant([f"startofseq {s}" for s in sentences_es[:100_000]])
X_valid_dec = tf.constant([f"startofseq {s}" for s in sentences_es[100_000:]])

# train and validation target
Y_train = text_vec_layer_es([f"{s} endofseq" for s in sentences_es[:100_000]])
Y_valid = text_vec_layer_es([f"{s} endofseq" for s in sentences_es[100_000:]])

tf.random.set_seed(42)

# encoder and decoder inputs
encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

# set length of each sentence to max_length and reduce teh vocab to vocab_size
encoder_input_ids = text_vec_layer_en(encoder_inputs)
decoder_input_ids = text_vec_layer_es(decoder_inputs)

# embeddings
encoder_embedding_layer = tf.keras.layers.Embedding(
    vocab_size, embed_size, mask_zero=True
)

decoder_embedding_layer = tf.keras.layers.Embedding(
    vocab_size, embed_size, mask_zero=True
)

encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

encoder = tf.keras.layers.LSTM(512, return_state=True)
decoder = tf.keras.layers.LSTM(512, return_sequences=True)

encoder_outputs, *encoder_state = encoder(encoder_embeddings)
decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)

output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")

Y_proba = output_layer(decoder_outputs)

model = tf.keras.Model(
    inputs=[encoder_inputs, decoder_inputs], outputs=[Y_proba]
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
