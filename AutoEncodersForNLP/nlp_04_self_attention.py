import numpy as np 


# Attention Mechanisms
# We need to feed all the encoder's outputs to the `Attention` layer, so we must add `return_sequences=True` to the encoder:

encoder = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
)

encoder_outputs, *encoder_state = encoder(encoder_embeddings)

encoder_state = [
    tf.concat(encoder_state[::2], axis=-1),  # short-term (0 & 2)
    tf.concat(encoder_state[1::2], axis=-1),
]  # long-term (1 & 3)

decoder = tf.keras.layers.LSTM(512, return_sequences=True)
decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)


# And finally, let's add the `Attention` layer and the output layer:


attention_layer = tf.keras.layers.Attention()
attention_outputs = attention_layer([decoder_outputs, encoder_outputs])
output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")
Y_proba = output_layer(attention_outputs)


# Warning: the following cell will take a while to run (possibly a couple hours if you are not using a GPU).

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


translate("I like soccer and also going to the beach")

beam_search(
    "I like soccer and also going to the beach", beam_width=3, verbose=True
)

#################################################################
