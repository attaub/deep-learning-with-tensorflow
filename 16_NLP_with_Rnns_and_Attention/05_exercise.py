import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)


# Exercise solutions
# ## 8.
# _Exercise:_ Embedded Reber grammars _were used by Hochreiter and Schmidhuber in [their paper](https://homl.info/93) about LSTMs. They are artificial grammars that produce strings such as "BPBTSXXVPSEPE." Check out Jenny Orr's [nice introduction](https://homl.info/108) to this topic. Choose a particular embedded Reber grammar (such as the one represented on Jenny Orr's page), then train an RNN to identify whether a string respects that grammar or not. You will first need to write a function capable of generating a training batch containing about 50% strings that respect the grammar, and 50% that don't._

# First we need to build a function that generates strings based on a grammar. The grammar will be represented as a list of possible transitions for each state. A transition specifies the string to output (or a grammar to generate it) and the next state.

# In[78]:


default_reber_grammar = [
    [("B", 1)],  # (state 0) =B=>(state 1)
    [("T", 2), ("P", 3)],  # (state 1) =T=>(state 2) or =P=>(state 3)
    [("S", 2), ("X", 4)],  # (state 2) =S=>(state 2) or =X=>(state 4)
    [("T", 3), ("V", 5)],  # and so on...
    [("X", 3), ("S", 6)],
    [("P", 4), ("V", 6)],
    [("E", None)],
]  # (state 6) =E=>(terminal state)

embedded_reber_grammar = [
    [("B", 1)],
    [("T", 2), ("P", 3)],
    [(default_reber_grammar, 4)],
    [(default_reber_grammar, 5)],
    [("T", 6)],
    [("P", 6)],
    [("E", None)],
]


def generate_string(grammar):
    state = 0
    output = []
    while state is not None:
        index = np.random.randint(len(grammar[state]))
        production, state = grammar[state][index]
        if isinstance(production, list):
            production = generate_string(grammar=production)
        output.append(production)
    return "".join(output)


# Let's generate a few strings based on the default Reber grammar:

# In[79]:


np.random.seed(42)

for _ in range(25):
    print(generate_string(default_reber_grammar), end=" ")


# Looks good. Now let's generate a few strings based on the embedded Reber grammar:

# In[80]:


np.random.seed(42)

for _ in range(25):
    print(generate_string(embedded_reber_grammar), end=" ")


# Okay, now we need a function to generate strings that do not respect the grammar. We could generate a random string, but the task would be a bit too easy, so instead we will generate a string that respects the grammar, and we will corrupt it by changing just one character:

# In[81]:


POSSIBLE_CHARS = "BEPSTVX"


def generate_corrupted_string(grammar, chars=POSSIBLE_CHARS):
    good_string = generate_string(grammar)
    index = np.random.randint(len(good_string))
    good_char = good_string[index]
    bad_char = np.random.choice(sorted(set(chars) - set(good_char)))
    return good_string[:index] + bad_char + good_string[index + 1 :]


# Let's look at a few corrupted strings:

# In[82]:


np.random.seed(42)

for _ in range(25):
    print(generate_corrupted_string(embedded_reber_grammar), end=" ")


# We cannot feed strings directly to an RNN, so we need to encode them somehow. One option would be to one-hot encode each character. Another option is to use embeddings. Let's go for the second option (but since there are just a handful of characters, one-hot encoding would probably be a good option as well). For embeddings to work, we need to convert each string into a sequence of character IDs. Let's write a function for that, using each character's index in the string of possible characters "BEPSTVX":

# In[83]:


def string_to_ids(s, chars=POSSIBLE_CHARS):
    return [chars.index(c) for c in s]


# In[84]:


string_to_ids("BTTTXXVVETE")


# We can now generate the dataset, with 50% good strings, and 50% bad strings:

# In[85]:


def generate_dataset(size):
    good_strings = [
        string_to_ids(generate_string(embedded_reber_grammar))
        for _ in range(size // 2)
    ]
    bad_strings = [
        string_to_ids(generate_corrupted_string(embedded_reber_grammar))
        for _ in range(size - size // 2)
    ]
    all_strings = good_strings + bad_strings
    X = tf.ragged.constant(all_strings, ragged_rank=1)
    y = np.array(
        [[1.0] for _ in range(len(good_strings))]
        + [[0.0] for _ in range(len(bad_strings))]
    )
    return X, y


# In[86]:


np.random.seed(42)

X_train, y_train = generate_dataset(10000)
X_valid, y_valid = generate_dataset(2000)


# Let's take a look at the first training sequence:

# In[87]:


X_train[0]


# What class does it belong to?

# In[88]:


y_train[0]


# Perfect! We are ready to create the RNN to identify good strings. We build a simple sequence binary classifier:

# In[89]:


np.random.seed(42)
tf.random.set_seed(42)

embedding_size = 5

model = keras.models.Sequential(
    [
        keras.layers.InputLayer(
            input_shape=[None], dtype=tf.int32, ragged=True
        ),
        keras.layers.Embedding(
            input_dim=len(POSSIBLE_CHARS), output_dim=embedding_size
        ),
        keras.layers.GRU(30),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
optimizer = keras.optimizers.SGD(
    learning_rate=0.02, momentum=0.95, nesterov=True
)
model.compile(
    loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)
history = model.fit(
    X_train, y_train, epochs=20, validation_data=(X_valid, y_valid)
)


# Now let's test our RNN on two tricky strings: the first one is bad while the second one is good. They only differ by the second to last character. If the RNN gets this right, it shows that it managed to notice the pattern that the second letter should always be equal to the second to last letter. That requires a fairly long short-term memory (which is the reason why we used a GRU cell).

# In[90]:


test_strings = [
    "BPBTSSSSSSSXXTTVPXVPXTTTTTVVETE",
    "BPBTSSSSSSSXXTTVPXVPXTTTTTVVEPE",
]
X_test = tf.ragged.constant(
    [string_to_ids(s) for s in test_strings], ragged_rank=1
)

y_proba = model.predict(X_test)
print()
print("Estimated probability that these are Reber strings:")
for index, string in enumerate(test_strings):
    print("{}: {:.2f}%".format(string, 100 * y_proba[index][0]))


# Ta-da! It worked fine. The RNN found the correct answers with very high confidence. :)

# ## 9.
# _Exercise: Train an Encoder–Decoder model that can convert a date string from one format to another (e.g., from "April 22, 2019" to "2019-04-22")._

# Let's start by creating the dataset. We will use random days between 1000-01-01 and 9999-12-31:

# In[91]:


from datetime import date

# cannot use strftime()'s %B format since it depends on the locale
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def random_dates(n_dates):
    min_date = date(1000, 1, 1).toordinal()
    max_date = date(9999, 12, 31).toordinal()

    ordinals = np.random.randint(max_date - min_date, size=n_dates) + min_date
    dates = [date.fromordinal(ordinal) for ordinal in ordinals]

    x = [MONTHS[dt.month - 1] + " " + dt.strftime("%d, %Y") for dt in dates]
    y = [dt.isoformat() for dt in dates]
    return x, y


# Here are a few random dates, displayed in both the input format and the target format:

# In[92]:


np.random.seed(42)

n_dates = 3
x_example, y_example = random_dates(n_dates)
print("{:25s}{:25s}".format("Input", "Target"))
print("-" * 50)
for idx in range(n_dates):
    print("{:25s}{:25s}".format(x_example[idx], y_example[idx]))


# Let's get the list of all possible characters in the inputs:

# In[93]:


INPUT_CHARS = "".join(sorted(set("".join(MONTHS) + "0123456789, ")))
INPUT_CHARS


# And here's the list of possible characters in the outputs:

# In[94]:


OUTPUT_CHARS = "0123456789-"


# Let's write a function to convert a string to a list of character IDs, as we did in the previous exercise:

# In[95]:


def date_str_to_ids(date_str, chars=INPUT_CHARS):
    return [chars.index(c) for c in date_str]


# In[96]:


date_str_to_ids(x_example[0], INPUT_CHARS)


# In[97]:


date_str_to_ids(y_example[0], OUTPUT_CHARS)


# In[98]:


def prepare_date_strs(date_strs, chars=INPUT_CHARS):
    X_ids = [date_str_to_ids(dt, chars) for dt in date_strs]
    X = tf.ragged.constant(X_ids, ragged_rank=1)
    return (X + 1).to_tensor()  # using 0 as the padding token ID


def create_dataset(n_dates):
    x, y = random_dates(n_dates)
    return prepare_date_strs(x, INPUT_CHARS), prepare_date_strs(
        y, OUTPUT_CHARS
    )


# In[99]:


np.random.seed(42)

X_train, Y_train = create_dataset(10000)
X_valid, Y_valid = create_dataset(2000)
X_test, Y_test = create_dataset(2000)


# In[100]:


Y_train[0]


# ### First version: a very basic seq2seq model

# Let's first try the simplest possible model: we feed in the input sequence, which first goes through the encoder (an embedding layer followed by a single LSTM layer), which outputs a vector, then it goes through a decoder (a single LSTM layer, followed by a dense output layer), which outputs a sequence of vectors, each representing the estimated probabilities for all possible output character.
#
# Since the decoder expects a sequence as input, we repeat the vector (which is output by the encoder) as many times as the longest possible output sequence.

# In[101]:


embedding_size = 32
max_output_length = Y_train.shape[1]

np.random.seed(42)
tf.random.set_seed(42)

encoder = keras.models.Sequential(
    [
        keras.layers.Embedding(
            input_dim=len(INPUT_CHARS) + 1,
            output_dim=embedding_size,
            input_shape=[None],
        ),
        keras.layers.LSTM(128),
    ]
)

decoder = keras.models.Sequential(
    [
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.Dense(len(OUTPUT_CHARS) + 1, activation="softmax"),
    ]
)

model = keras.models.Sequential(
    [encoder, keras.layers.RepeatVector(max_output_length), decoder]
)

optimizer = keras.optimizers.Nadam()
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
history = model.fit(
    X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid)
)


# Looks great, we reach 100% validation accuracy! Let's use the model to make some predictions. We will need to be able to convert a sequence of character IDs to a readable string:

# In[102]:


def ids_to_date_strs(ids, chars=OUTPUT_CHARS):
    return [
        "".join([("?" + chars)[index] for index in sequence])
        for sequence in ids
    ]


# Now we can use the model to convert some dates

# In[103]:


X_new = prepare_date_strs(["September 17, 2009", "July 14, 1789"])


# In[104]:


# ids = model.predict_classes(X_new)
ids = np.argmax(model.predict(X_new), axis=-1)
for date_str in ids_to_date_strs(ids):
    print(date_str)


# Perfect! :)

# However, since the model was only trained on input strings of length 18 (which is the length of the longest date), it does not perform well if we try to use it to make predictions on shorter sequences:

# In[105]:


X_new = prepare_date_strs(["May 02, 2020", "July 14, 1789"])


# In[106]:


# ids = model.predict_classes(X_new)
ids = np.argmax(model.predict(X_new), axis=-1)
for date_str in ids_to_date_strs(ids):
    print(date_str)


# Oops! We need to ensure that we always pass sequences of the same length as during training, using padding if necessary. Let's write a little helper function for that:

# In[107]:


max_input_length = X_train.shape[1]


def prepare_date_strs_padded(date_strs):
    X = prepare_date_strs(date_strs)
    if X.shape[1] < max_input_length:
        X = tf.pad(X, [[0, 0], [0, max_input_length - X.shape[1]]])
    return X


def convert_date_strs(date_strs):
    X = prepare_date_strs_padded(date_strs)
    # ids = model.predict_classes(X)
    ids = np.argmax(model.predict(X), axis=-1)
    return ids_to_date_strs(ids)


# In[108]:


convert_date_strs(["May 02, 2020", "July 14, 1789"])


# Cool! Granted, there are certainly much easier ways to write a date conversion tool (e.g., using regular expressions or even basic string manipulation), but you have to admit that using neural networks is way cooler. ;-)

# However, real-life sequence-to-sequence problems will usually be harder, so for the sake of completeness, let's build a more powerful model.

# ### Second version: feeding the shifted targets to the decoder (teacher forcing)

# Instead of feeding the decoder a simple repetition of the encoder's output vector, we can feed it the target sequence, shifted by one time step to the right. This way, at each time step the decoder will know what the previous target character was. This should help is tackle more complex sequence-to-sequence problems.
#
# Since the first output character of each target sequence has no previous character, we will need a new token to represent the start-of-sequence (sos).
#
# During inference, we won't know the target, so what will we feed the decoder? We can just predict one character at a time, starting with an sos token, then feeding the decoder all the characters that were predicted so far (we will look at this in more details later in this notebook).
#
# But if the decoder's LSTM expects to get the previous target as input at each step, how shall we pass it it the vector output by the encoder? Well, one option is to ignore the output vector, and instead use the encoder's LSTM state as the initial state of the decoder's LSTM (which requires that encoder's LSTM must have the same number of units as the decoder's LSTM).
#
# Now let's create the decoder's inputs (for training, validation and testing). The sos token will be represented using the last possible output character's ID + 1.

# In[109]:


sos_id = len(OUTPUT_CHARS) + 1


def shifted_output_sequences(Y):
    sos_tokens = tf.fill(dims=(len(Y), 1), value=sos_id)
    return tf.concat([sos_tokens, Y[:, :-1]], axis=1)


X_train_decoder = shifted_output_sequences(Y_train)
X_valid_decoder = shifted_output_sequences(Y_valid)
X_test_decoder = shifted_output_sequences(Y_test)


# Let's take a look at the decoder's training inputs:

# In[110]:


X_train_decoder


# Now let's build the model. It's not a simple sequential model anymore, so let's use the functional API:

# In[111]:


encoder_embedding_size = 32
decoder_embedding_size = 32
lstm_units = 128

np.random.seed(42)
tf.random.set_seed(42)

encoder_input = keras.layers.Input(shape=[None], dtype=tf.int32)
encoder_embedding = keras.layers.Embedding(
    input_dim=len(INPUT_CHARS) + 1, output_dim=encoder_embedding_size
)(encoder_input)
_, encoder_state_h, encoder_state_c = keras.layers.LSTM(
    lstm_units, return_state=True
)(encoder_embedding)
encoder_state = [encoder_state_h, encoder_state_c]

decoder_input = keras.layers.Input(shape=[None], dtype=tf.int32)
decoder_embedding = keras.layers.Embedding(
    input_dim=len(OUTPUT_CHARS) + 2, output_dim=decoder_embedding_size
)(decoder_input)
decoder_lstm_output = keras.layers.LSTM(lstm_units, return_sequences=True)(
    decoder_embedding, initial_state=encoder_state
)
decoder_output = keras.layers.Dense(
    len(OUTPUT_CHARS) + 1, activation="softmax"
)(decoder_lstm_output)

model = keras.models.Model(
    inputs=[encoder_input, decoder_input], outputs=[decoder_output]
)

optimizer = keras.optimizers.Nadam()
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
history = model.fit(
    [X_train, X_train_decoder],
    Y_train,
    epochs=10,
    validation_data=([X_valid, X_valid_decoder], Y_valid),
)


# This model also reaches 100% validation accuracy, but it does so even faster.

# Let's once again use the model to make some predictions. This time we need to predict characters one by one.

# In[112]:


sos_id = len(OUTPUT_CHARS) + 1


def predict_date_strs(date_strs):
    X = prepare_date_strs_padded(date_strs)
    Y_pred = tf.fill(dims=(len(X), 1), value=sos_id)
    for index in range(max_output_length):
        pad_size = max_output_length - Y_pred.shape[1]
        X_decoder = tf.pad(Y_pred, [[0, 0], [0, pad_size]])
        Y_probas_next = model.predict([X, X_decoder])[:, index : index + 1]
        Y_pred_next = tf.argmax(Y_probas_next, axis=-1, output_type=tf.int32)
        Y_pred = tf.concat([Y_pred, Y_pred_next], axis=1)
    return ids_to_date_strs(Y_pred[:, 1:])


# In[113]:


predict_date_strs(["July 14, 1789", "May 01, 2020"])


# Works fine! :)

# ### Third version: using TF-Addons's seq2seq implementation

# Let's build exactly the same model, but using TF-Addon's seq2seq API. The implementation below is almost very similar to the TFA example higher in this notebook, except without the model input to specify the output sequence length, for simplicity (but you can easily add it back in if you need it for your projects, when the output sequences have very different lengths).

# In[114]:


import tensorflow_addons as tfa

np.random.seed(42)
tf.random.set_seed(42)

encoder_embedding_size = 32
decoder_embedding_size = 32
units = 128

encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
sequence_lengths = keras.layers.Input(shape=[], dtype=np.int32)

encoder_embeddings = keras.layers.Embedding(
    len(INPUT_CHARS) + 1, encoder_embedding_size
)(encoder_inputs)

decoder_embedding_layer = keras.layers.Embedding(
    len(OUTPUT_CHARS) + 2, decoder_embedding_size
)
decoder_embeddings = decoder_embedding_layer(decoder_inputs)

encoder = keras.layers.LSTM(units, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
encoder_state = [state_h, state_c]

sampler = tfa.seq2seq.sampler.TrainingSampler()

decoder_cell = keras.layers.LSTMCell(units)
output_layer = keras.layers.Dense(len(OUTPUT_CHARS) + 1)

decoder = tfa.seq2seq.basic_decoder.BasicDecoder(
    decoder_cell, sampler, output_layer=output_layer
)
final_outputs, final_state, final_sequence_lengths = decoder(
    decoder_embeddings, initial_state=encoder_state
)
Y_proba = keras.layers.Activation("softmax")(final_outputs.rnn_output)

model = keras.models.Model(
    inputs=[encoder_inputs, decoder_inputs], outputs=[Y_proba]
)
optimizer = keras.optimizers.Nadam()
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
history = model.fit(
    [X_train, X_train_decoder],
    Y_train,
    epochs=15,
    validation_data=([X_valid, X_valid_decoder], Y_valid),
)


# And once again, 100% validation accuracy! To use the model, we can just reuse the `predict_date_strs()` function:

# In[115]:


predict_date_strs(["July 14, 1789", "May 01, 2020"])


# However, there's a much more efficient way to perform inference. Until now, during inference, we've run the model once for each new character. Instead, we can create a new decoder, based on the previously trained layers, but using a `GreedyEmbeddingSampler` instead of a `TrainingSampler`.
#
# At each time step, the `GreedyEmbeddingSampler` will compute the argmax of the decoder's outputs, and run the resulting token IDs through the decoder's embedding layer. Then it will feed the resulting embeddings to the decoder's LSTM cell at the next time step. This way, we only need to run the decoder once to get the full prediction.

# In[116]:


inference_sampler = tfa.seq2seq.sampler.GreedyEmbeddingSampler(
    embedding_fn=decoder_embedding_layer
)
inference_decoder = tfa.seq2seq.basic_decoder.BasicDecoder(
    decoder_cell,
    inference_sampler,
    output_layer=output_layer,
    maximum_iterations=max_output_length,
)
batch_size = tf.shape(encoder_inputs)[:1]
start_tokens = tf.fill(dims=batch_size, value=sos_id)
final_outputs, final_state, final_sequence_lengths = inference_decoder(
    start_tokens,
    initial_state=encoder_state,
    start_tokens=start_tokens,
    end_token=0,
)

inference_model = keras.models.Model(
    inputs=[encoder_inputs], outputs=[final_outputs.sample_id]
)


# A few notes:
# * The `GreedyEmbeddingSampler` needs the `start_tokens` (a vector containing the start-of-sequence ID for each decoder sequence), and the `end_token` (the decoder will stop decoding a sequence once the model outputs this token).
# * We must set `maximum_iterations` when creating the `BasicDecoder`, or else it may run into an infinite loop (if the model never outputs the end token for at least one of the sequences). This would force you would to restart the Jupyter kernel.
# * The decoder inputs are not needed anymore, since all the decoder inputs are generated dynamically based on the outputs from the previous time step.
# * The model's outputs are `final_outputs.sample_id` instead of the softmax of `final_outputs.rnn_outputs`. This allows us to directly get the argmax of the model's outputs. If you prefer to have access to the logits, you can replace `final_outputs.sample_id` with `final_outputs.rnn_outputs`.

# Now we can write a simple function that uses the model to perform the date format conversion:

# In[117]:


def fast_predict_date_strs(date_strs):
    X = prepare_date_strs_padded(date_strs)
    Y_pred = inference_model.predict(X)
    return ids_to_date_strs(Y_pred)


# In[118]:


fast_predict_date_strs(["July 14, 1789", "May 01, 2020"])


# Let's check that it really is faster:

# In[119]:


get_ipython().run_line_magic(
    'timeit', 'predict_date_strs(["July 14, 1789", "May 01, 2020"])'
)


# In[120]:


get_ipython().run_line_magic(
    'timeit', 'fast_predict_date_strs(["July 14, 1789", "May 01, 2020"])'
)


# That's more than a 10x speedup! And it would be even more if we were handling longer sequences.

# ### Fourth version: using TF-Addons's seq2seq implementation with a scheduled sampler

# **Warning**: due to a TF bug, this version only works using TensorFlow 2.2 or above.

# When we trained the previous model, at each time step _t_ we gave the model the target token for time step _t_ - 1. However, at inference time, the model did not get the previous target at each time step. Instead, it got the previous prediction. So there is a discrepancy between training and inference, which may lead to disappointing performance. To alleviate this, we can gradually replace the targets with the predictions, during training. For this, we just need to replace the `TrainingSampler` with a `ScheduledEmbeddingTrainingSampler`, and use a Keras callback to gradually increase the `sampling_probability` (i.e., the probability that the decoder will use the prediction from the previous time step rather than the target for the previous time step).

# In[121]:


import tensorflow_addons as tfa

np.random.seed(42)
tf.random.set_seed(42)

n_epochs = 20
encoder_embedding_size = 32
decoder_embedding_size = 32
units = 128

encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
sequence_lengths = keras.layers.Input(shape=[], dtype=np.int32)

encoder_embeddings = keras.layers.Embedding(
    len(INPUT_CHARS) + 1, encoder_embedding_size
)(encoder_inputs)

decoder_embedding_layer = keras.layers.Embedding(
    len(OUTPUT_CHARS) + 2, decoder_embedding_size
)
decoder_embeddings = decoder_embedding_layer(decoder_inputs)

encoder = keras.layers.LSTM(units, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
encoder_state = [state_h, state_c]

sampler = tfa.seq2seq.sampler.ScheduledEmbeddingTrainingSampler(
    sampling_probability=0.0, embedding_fn=decoder_embedding_layer
)
# we must set the sampling_probability after creating the sampler
# (see https://github.com/tensorflow/addons/pull/1714)
sampler.sampling_probability = tf.Variable(0.0)

decoder_cell = keras.layers.LSTMCell(units)
output_layer = keras.layers.Dense(len(OUTPUT_CHARS) + 1)

decoder = tfa.seq2seq.basic_decoder.BasicDecoder(
    decoder_cell, sampler, output_layer=output_layer
)
final_outputs, final_state, final_sequence_lengths = decoder(
    decoder_embeddings, initial_state=encoder_state
)
Y_proba = keras.layers.Activation("softmax")(final_outputs.rnn_output)

model = keras.models.Model(
    inputs=[encoder_inputs, decoder_inputs], outputs=[Y_proba]
)
optimizer = keras.optimizers.Nadam()
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)


def update_sampling_probability(epoch, logs):
    proba = min(1.0, epoch / (n_epochs - 10))
    sampler.sampling_probability.assign(proba)


sampling_probability_cb = keras.callbacks.LambdaCallback(
    on_epoch_begin=update_sampling_probability
)
history = model.fit(
    [X_train, X_train_decoder],
    Y_train,
    epochs=n_epochs,
    validation_data=([X_valid, X_valid_decoder], Y_valid),
    callbacks=[sampling_probability_cb],
)


# Not quite 100% validation accuracy, but close enough!

# For inference, we could do the exact same thing as earlier, using a `GreedyEmbeddingSampler`. However, just for the sake of completeness, let's use a `SampleEmbeddingSampler` instead. It's almost the same thing, except that instead of using the argmax of the model's output to find the token ID, it treats the outputs as logits and uses them to sample a token ID randomly. This can be useful when you want to generate text. The `softmax_temperature` argument serves the
# same purpose as when we generated Shakespeare-like text (the higher this argument, the more random the generated text will be).

# In[122]:


softmax_temperature = tf.Variable(1.0)

inference_sampler = tfa.seq2seq.sampler.SampleEmbeddingSampler(
    embedding_fn=decoder_embedding_layer,
    softmax_temperature=softmax_temperature,
)
inference_decoder = tfa.seq2seq.basic_decoder.BasicDecoder(
    decoder_cell,
    inference_sampler,
    output_layer=output_layer,
    maximum_iterations=max_output_length,
)
batch_size = tf.shape(encoder_inputs)[:1]
start_tokens = tf.fill(dims=batch_size, value=sos_id)
final_outputs, final_state, final_sequence_lengths = inference_decoder(
    start_tokens,
    initial_state=encoder_state,
    start_tokens=start_tokens,
    end_token=0,
)

inference_model = keras.models.Model(
    inputs=[encoder_inputs], outputs=[final_outputs.sample_id]
)


# In[123]:


def creative_predict_date_strs(date_strs, temperature=1.0):
    softmax_temperature.assign(temperature)
    X = prepare_date_strs_padded(date_strs)
    Y_pred = inference_model.predict(X)
    return ids_to_date_strs(Y_pred)


# In[124]:


tf.random.set_seed(42)

creative_predict_date_strs(["July 14, 1789", "May 01, 2020"])


# Dates look good at room temperature. Now let's heat things up a bit:

# In[125]:


tf.random.set_seed(42)

creative_predict_date_strs(["July 14, 1789", "May 01, 2020"], temperature=5.0)


# Oops, the dates are overcooked, now. Let's call them "creative" dates.

# ### Fifth version: using TFA seq2seq, the Keras subclassing API and attention mechanisms

# The sequences in this problem are pretty short, but if we wanted to tackle longer sequences, we would probably have to use attention mechanisms. While it's possible to code our own implementation, it's simpler and more efficient to use TF-Addons's implementation instead. Let's do that now, this time using Keras' subclassing API.
#
# **Warning**: due to a TensorFlow bug (see [this issue](https://github.com/tensorflow/addons/issues/1153) for details), the `get_initial_state()` method fails in eager mode, so for now we have to use the subclassing API, as Keras automatically calls `tf.function()` on the `call()` method (so it runs in graph mode).

# In this implementation, we've reverted back to using the `TrainingSampler`, for simplicity (but you can easily tweak it to use a `ScheduledEmbeddingTrainingSampler` instead). We also use a `GreedyEmbeddingSampler` during inference, so this class is pretty easy to use:

# In[126]:


class DateTranslation(keras.models.Model):
    def __init__(
        self,
        units=128,
        encoder_embedding_size=32,
        decoder_embedding_size=32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_embedding = keras.layers.Embedding(
            input_dim=len(INPUT_CHARS) + 1, output_dim=encoder_embedding_size
        )
        self.encoder = keras.layers.LSTM(
            units, return_sequences=True, return_state=True
        )
        self.decoder_embedding = keras.layers.Embedding(
            input_dim=len(OUTPUT_CHARS) + 2, output_dim=decoder_embedding_size
        )
        self.attention = tfa.seq2seq.LuongAttention(units)
        decoder_inner_cell = keras.layers.LSTMCell(units)
        self.decoder_cell = tfa.seq2seq.AttentionWrapper(
            cell=decoder_inner_cell, attention_mechanism=self.attention
        )
        output_layer = keras.layers.Dense(len(OUTPUT_CHARS) + 1)
        self.decoder = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.TrainingSampler(),
            output_layer=output_layer,
        )
        self.inference_decoder = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.GreedyEmbeddingSampler(
                embedding_fn=self.decoder_embedding
            ),
            output_layer=output_layer,
            maximum_iterations=max_output_length,
        )

    def call(self, inputs, training=None):
        encoder_input, decoder_input = inputs
        encoder_embeddings = self.encoder_embedding(encoder_input)
        encoder_outputs, encoder_state_h, encoder_state_c = self.encoder(
            encoder_embeddings, training=training
        )
        encoder_state = [encoder_state_h, encoder_state_c]

        self.attention(encoder_outputs, setup_memory=True)

        decoder_embeddings = self.decoder_embedding(decoder_input)

        decoder_initial_state = self.decoder_cell.get_initial_state(
            decoder_embeddings
        )
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=encoder_state
        )

        if training:
            decoder_outputs, _, _ = self.decoder(
                decoder_embeddings,
                initial_state=decoder_initial_state,
                training=training,
            )
        else:
            start_tokens = tf.zeros_like(encoder_input[:, 0]) + sos_id
            decoder_outputs, _, _ = self.inference_decoder(
                decoder_embeddings,
                initial_state=decoder_initial_state,
                start_tokens=start_tokens,
                end_token=0,
            )

        return tf.nn.softmax(decoder_outputs.rnn_output)


# In[127]:


np.random.seed(42)
tf.random.set_seed(42)

model = DateTranslation()
optimizer = keras.optimizers.Nadam()
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
history = model.fit(
    [X_train, X_train_decoder],
    Y_train,
    epochs=25,
    validation_data=([X_valid, X_valid_decoder], Y_valid),
)


# Not quite 100% validation accuracy, but close. It took a bit longer to converge this time, but there were also more parameters and more computations per iteration. And we did not use a scheduled sampler.
#
# To use the model, we can write yet another little function:

# In[128]:


def fast_predict_date_strs_v2(date_strs):
    X = prepare_date_strs_padded(date_strs)
    X_decoder = tf.zeros(shape=(len(X), max_output_length), dtype=tf.int32)
    Y_probas = model.predict([X, X_decoder])
    Y_pred = tf.argmax(Y_probas, axis=-1)
    return ids_to_date_strs(Y_pred)


# In[129]:


fast_predict_date_strs_v2(["July 14, 1789", "May 01, 2020"])


# There are still a few interesting features from TF-Addons that you may want to look at:
# * Using a `BeamSearchDecoder` rather than a `BasicDecoder` for inference. Instead of outputing the character with the highest probability, this decoder keeps track of the several candidates, and keeps only the most likely sequences of candidates (see chapter 16 in the book for more details).
# * Setting masks or specifying `sequence_length` if the input or target sequences may have very different lengths.
# * Using a `ScheduledOutputTrainingSampler`, which gives you more flexibility than the `ScheduledEmbeddingTrainingSampler` to decide how to feed the output at time _t_ to the cell at time _t_+1. By default it feeds the outputs directly to cell, without computing the argmax ID and passing it through an embedding layer. Alternatively, you specify a `next_inputs_fn` function that will be used to convert the cell outputs to inputs at the next step.

# ## 10.
# _Exercise: Go through TensorFlow's [Neural Machine Translation with Attention tutorial](https://homl.info/nmttuto)._

# Simply open the Colab and follow its instructions. Alternatively, if you want a simpler example of using TF-Addons's seq2seq implementation for Neural Machine Translation (NMT), look at the solution to the previous question. The last model implementation will give you a simpler example of using TF-Addons to build an NMT model using attention mechanisms.

# ## 11.
# _Exercise: Use one of the recent language models (e.g., GPT) to generate more convincing Shakespearean text._

# The simplest way to use recent language models is to use the excellent [transformers library](https://huggingface.co/transformers/), open sourced by Hugging Face. It provides many modern neural net architectures (including BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet and more) for Natural Language Processing (NLP), including many pretrained models. It relies on either TensorFlow or PyTorch. Best of all: it's amazingly simple to use.

# First, let's load a pretrained model. In this example, we will use OpenAI's GPT model, with an additional Language Model on top (just a linear layer with weights tied to the input embeddings). Let's import it and load the pretrained weights (this will download about 445MB of data to `~/.cache/torch/transformers`):

# In[130]:


from transformers import TFOpenAIGPTLMHeadModel

model = TFOpenAIGPTLMHeadModel.from_pretrained("openai-gpt")


# Next we will need a specialized tokenizer for this model. This one will try to use the [spaCy](https://spacy.io/) and [ftfy](https://pypi.org/project/ftfy/) libraries if they are installed, or else it will fall back to BERT's `BasicTokenizer` followed by Byte-Pair Encoding (which should be fine for most use cases).

# In[131]:


from transformers import OpenAIGPTTokenizer

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")


# Now let's use the tokenizer to tokenize and encode the prompt text:

# In[132]:


prompt_text = "This royal throne of kings, this sceptred isle"
encoded_prompt = tokenizer.encode(
    prompt_text, add_special_tokens=False, return_tensors="tf"
)
encoded_prompt


# Easy! Next, let's use the model to generate text after the prompt. We will generate 5 different sentences, each starting with the prompt text, followed by 40 additional tokens. For an explanation of what all the hyperparameters do, make sure to check out this great [blog post](https://huggingface.co/blog/how-to-generate) by Patrick von Platen (from Hugging Face). You can play around with the hyperparameters to try to obtain better results.

# In[133]:


num_sequences = 5
length = 40

generated_sequences = model.generate(
    input_ids=encoded_prompt,
    do_sample=True,
    max_length=length + len(encoded_prompt[0]),
    temperature=1.0,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1.0,
    num_return_sequences=num_sequences,
)

generated_sequences


# Now let's decode the generated sequences and print them:

# In[134]:


for sequence in generated_sequences:
    text = tokenizer.decode(sequence, clean_up_tokenization_spaces=True)
    print(text)
    print("-" * 80)


# You can try more recent (and larger) models, such as GPT-2, CTRL, Transformer-XL or XLNet, which are all available as pretrained models in the transformers library, including variants with Language Models on top. The preprocessing steps vary slightly between models, so make sure to check out this [generation example](https://github.com/huggingface/transformers/blob/master/examples/run_generation.py) from the transformers documentation (this example uses PyTorch, but it will work with very little tweaks, such as adding `TF` at the beginning of the model class name, removing the `.to()` method calls, and using `return_tensors="tf"` instead of `"pt"`.

# Hope you enjoyed this chapter! :)
