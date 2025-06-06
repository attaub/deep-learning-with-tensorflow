
#################################################################
# ## 9. Tackling the SketchRNN Dataset
# _Exercise: Train a classification model for the SketchRNN dataset, available in TensorFlow Datasets._
# The dataset is not available in TFDS yet, the [pull request](https://github.com/tensorflow/datasets/pull/361) is still work in progress. Luckily, the data is conveniently available as TFRecords, so let's download it (it might take a while, as it's about 1 GB large, with 3,450,000 training sketches and 345,000 test sketches):
#
DOWNLOAD_ROOT = "http://download.tensorflow.org/data/"
FILENAME = "quickdraw_tutorial_dataset_v1.tar.gz"
filepath = keras.utils.get_file(
    FILENAME,
    DOWNLOAD_ROOT + FILENAME,
    cache_subdir="datasets/quickdraw",
    extract=True,
)
#
quickdraw_dir = Path(filepath).parent
train_files = sorted(
    [str(path) for path in quickdraw_dir.glob("training.tfrecord-*")]
)
eval_files = sorted(
    [str(path) for path in quickdraw_dir.glob("eval.tfrecord-*")]
)
#
train_files
#
eval_files
#
with open(quickdraw_dir / "eval.tfrecord.classes") as test_classes_file:
    test_classes = test_classes_file.readlines()
with open(quickdraw_dir / "training.tfrecord.classes") as train_classes_file:
    train_classes = train_classes_file.readlines()
#
assert train_classes == test_classes
class_names = [name.strip().lower() for name in train_classes]
#
sorted(class_names)
#
def parse(data_batch):
    feature_descriptions = {
        "ink": tf.io.VarLenFeature(dtype=tf.float32),
        "shape": tf.io.FixedLenFeature([2], dtype=tf.int64),
        "class_index": tf.io.FixedLenFeature([1], dtype=tf.int64),
    }
    examples = tf.io.parse_example(data_batch, feature_descriptions)
    flat_sketches = tf.sparse.to_dense(examples["ink"])
    sketches = tf.reshape(flat_sketches, shape=[tf.size(data_batch), -1, 3])
    lengths = examples["shape"][:, 0]
    labels = examples["class_index"][:, 0]
    return sketches, lengths, labels


#
def quickdraw_dataset(
    filepaths,
    batch_size=32,
    shuffle_buffer_size=None,
    n_parse_threads=5,
    n_read_threads=5,
    cache=False,
):
    dataset = tf.data.TFRecordDataset(
        filepaths, num_parallel_reads=n_read_threads
    )
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse, num_parallel_calls=n_parse_threads)
    return dataset.prefetch(1)


#
train_set = quickdraw_dataset(train_files, shuffle_buffer_size=10000)
valid_set = quickdraw_dataset(eval_files[:5])
test_set = quickdraw_dataset(eval_files[5:])
#
for sketches, lengths, labels in train_set.take(1):
    print("sketches =", sketches)
    print("lengths =", lengths)
    print("labels =", labels)
#
def draw_sketch(sketch, label=None):
    origin = np.array([[0.0, 0.0, 0.0]])
    sketch = np.r_[origin, sketch]
    stroke_end_indices = np.argwhere(sketch[:, -1] == 1.0)[:, 0]
    coordinates = np.cumsum(sketch[:, :2], axis=0)
    strokes = np.split(coordinates, stroke_end_indices + 1)
    title = class_names[label.numpy()] if label is not None else "Try to guess"
    plt.title(title)
    plt.plot(coordinates[:, 0], -coordinates[:, 1], "y:")
    for stroke in strokes:
        plt.plot(stroke[:, 0], -stroke[:, 1], ".-")
    plt.axis("off")


def draw_sketches(sketches, lengths, labels):
    n_sketches = len(sketches)
    n_cols = 4
    n_rows = (n_sketches - 1) // n_cols + 1
    plt.figure(figsize=(n_cols * 3, n_rows * 3.5))
    for index, sketch, length, label in zip(
        range(n_sketches), sketches, lengths, labels
    ):
        plt.subplot(n_rows, n_cols, index + 1)
        draw_sketch(sketch[:length], label)
    plt.show()


for sketches, lengths, labels in train_set.take(1):
    draw_sketches(sketches, lengths, labels)
# Most sketches are composed of less than 100 points:
#
lengths = np.concatenate([lengths for _, lengths, _ in train_set.take(1000)])
plt.hist(lengths, bins=150, density=True)
plt.axis([0, 200, 0, 0.03])
plt.xlabel("length")
plt.ylabel("density")
plt.show()
#
def crop_long_sketches(dataset, max_length=100):
    return dataset.map(
        lambda inks, lengths, labels: (inks[:, :max_length], labels)
    )


cropped_train_set = crop_long_sketches(train_set)
cropped_valid_set = crop_long_sketches(valid_set)
cropped_test_set = crop_long_sketches(test_set)
#
model = keras.models.Sequential(
    [
        keras.layers.Conv1D(32, kernel_size=5, strides=2, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(64, kernel_size=5, strides=2, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(128, kernel_size=3, strides=2, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.LSTM(128),
        keras.layers.Dense(len(class_names), activation="softmax"),
    ]
)
optimizer = keras.optimizers.SGD(learning_rate=1e-2, clipnorm=1.0)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy", "sparse_top_k_categorical_accuracy"],
)
history = model.fit(
    cropped_train_set, epochs=2, validation_data=cropped_valid_set
)
#
y_test = np.concatenate([labels for _, _, labels in test_set])
y_probas = model.predict(test_set)
#
np.mean(keras.metrics.sparse_top_k_categorical_accuracy(y_test, y_probas))
#
n_new = 10
Y_probas = model.predict(sketches)
top_k = tf.nn.top_k(Y_probas, k=5)
for index in range(n_new):
    plt.figure(figsize=(3, 3.5))
    draw_sketch(sketches[index])
    plt.show()
    print("Top-5 predictions:".format(index + 1))
    for k in range(5):
        class_name = class_names[top_k.indices[index, k]]
        proba = 100 * top_k.values[index, k]
        print("  {}. {} {:.3f}%".format(k + 1, class_name, proba))
    print("Answer: {}".format(class_names[labels[index].numpy()]))
#
model.save("my_sketchrnn")
# ## 10. Bach Chorales
# _Exercise: Download the [Bach chorales](https://homl.info/bach) dataset and unzip it. It is composed of 382 chorales composed by Johann Sebastian Bach. Each chorale is 100 to 640 time steps long, and each time step contains 4 integers, where each integer corresponds to a note's index on a piano (except for the value 0, which means that no note is played). Train a model—recurrent, convolutional, or both—that can predict the next time step (four notes), given a sequence of time steps from a chorale. Then use this model to generate Bach-like music, one note at a time: you can do this by giving the model the start of a chorale and asking it to predict the next time step, then appending these time steps to the input sequence and asking the model for the next note, and so on. Also make sure to check out [Google's Coconet model](https://homl.info/coconet), which was used for a nice [Google doodle about Bach](https://www.google.com/doodles/celebrating-johann-sebastian-bach)._
#
#
#
DOWNLOAD_ROOT = (
    "https://github.com/ageron/handson-ml2/raw/master/datasets/jsb_chorales/"
)
FILENAME = "jsb_chorales.tgz"
filepath = keras.utils.get_file(
    FILENAME,
    DOWNLOAD_ROOT + FILENAME,
    cache_subdir="datasets/jsb_chorales",
    extract=True,
)
#
jsb_chorales_dir = Path(filepath).parent
train_files = sorted(jsb_chorales_dir.glob("train/chorale_*.csv"))
valid_files = sorted(jsb_chorales_dir.glob("valid/chorale_*.csv"))
test_files = sorted(jsb_chorales_dir.glob("test/chorale_*.csv"))
#
import pandas as pd


def load_chorales(filepaths):
    return [pd.read_csv(filepath).values.tolist() for filepath in filepaths]


train_chorales = load_chorales(train_files)
valid_chorales = load_chorales(valid_files)
test_chorales = load_chorales(test_files)
#
train_chorales[0]
# Notes range from 36 (C1 = C on octave 1) to 81 (A5 = A on octave 5), plus 0 for silence:
#
notes = set()
for chorales in (train_chorales, valid_chorales, test_chorales):
    for chorale in chorales:
        for chord in chorale:
            notes |= set(chord)
n_notes = len(notes)
min_note = min(notes - {0})
max_note = max(notes)
assert min_note == 36
assert max_note == 81
# Let's write a few functions to listen to these chorales (you don't need to understand the details here, and in fact there are certainly simpler ways to do this, for example using MIDI players, but I just wanted to have a bit of fun writing a synthesizer):
#
from IPython.display import Audio


def notes_to_frequencies(notes):
    # Frequency doubles when you go up one octave; there are 12 semi-tones
    # per octave; Note A on octave 4 is 440 Hz, and it is note number 69.
    return 2 ** ((np.array(notes) - 69) / 12) * 440


def frequencies_to_samples(frequencies, tempo, sample_rate):
    note_duration = 60 / tempo  # the tempo is measured in beats per minutes
    # To reduce click sound at every beat, we round the frequencies to try to
    # get the samples close to zero at the end of each note.
    frequencies = np.round(note_duration * frequencies) / note_duration
    n_samples = int(note_duration * sample_rate)
    time = np.linspace(0, note_duration, n_samples)
    sine_waves = np.sin(2 * np.pi * frequencies.reshape(-1, 1) * time)
    # Removing all notes with frequencies ≤ 9 Hz (includes note 0 = silence)
    sine_waves *= (frequencies > 9.0).reshape(-1, 1)
    return sine_waves.reshape(-1)


def chords_to_samples(chords, tempo, sample_rate):
    freqs = notes_to_frequencies(chords)
    freqs = np.r_[freqs, freqs[-1:]]  # make last note a bit longer
    merged = np.mean(
        [
            frequencies_to_samples(melody, tempo, sample_rate)
            for melody in freqs.T
        ],
        axis=0,
    )
    n_fade_out_samples = sample_rate * 60 // tempo  # fade out last note
    fade_out = np.linspace(1.0, 0.0, n_fade_out_samples) ** 2
    merged[-n_fade_out_samples:] *= fade_out
    return merged


def play_chords(
    chords, tempo=160, amplitude=0.1, sample_rate=44100, filepath=None
):
    samples = amplitude * chords_to_samples(chords, tempo, sample_rate)
    if filepath:
        from scipy.io import wavfile

        samples = (2**15 * samples).astype(np.int16)
        wavfile.write(filepath, sample_rate, samples)
        return display(Audio(filepath))
    else:
        return display(Audio(samples, rate=sample_rate))


# Now let's listen to a few chorales:
#
for index in range(3):
    play_chords(train_chorales[index])
# Divine! :)
# In order to be able to generate new chorales, we want to train a model that can predict the next chord given all the previous chords. If we naively try to predict the next chord in one shot, predicting all 4 notes at once, we run the risk of getting notes that don't go very well together (believe me, I tried). It's much better and simpler to predict one note at a time. So we will need to preprocess every chorale, turning each chord into an arpegio (i.e., a sequence of notes rather than notes played simultaneuously). So each chorale will be a long sequence of notes (rather than chords), and we can just train a model that can predict the next note given all the previous notes. We will use a sequence-to-sequence approach, where we feed a window to the neural net, and it tries to predict that same window shifted one time step into the future.
#
# We will also shift the values so that they range from 0 to 46, where 0 represents silence, and values 1 to 46 represent notes 36 (C1) to 81 (A5).
#
# And we will train the model on windows of 128 notes (i.e., 32 chords).
#
# Since the dataset fits in memory, we could preprocess the chorales in RAM using any Python code we like, but I will demonstrate here how to do all the preprocessing using tf.data (there will be more details about creating windows using tf.data in the next chapter).
#
def create_target(batch):
    X = batch[:, :-1]
    Y = batch[:, 1:]  # predict next note in each arpegio, at each step
    return X, Y


def preprocess(window):
    window = tf.where(
        window == 0, window, window - min_note + 1
    )  # shift values
    return tf.reshape(window, [-1])  # convert to arpegio


def bach_dataset(
    chorales,
    batch_size=32,
    shuffle_buffer_size=None,
    window_size=32,
    window_shift=16,
    cache=True,
):
    def batch_window(window):
        return window.batch(window_size + 1)

    def to_windows(chorale):
        dataset = tf.data.Dataset.from_tensor_slices(chorale)
        dataset = dataset.window(
            window_size + 1, window_shift, drop_remainder=True
        )
        return dataset.flat_map(batch_window)

    chorales = tf.ragged.constant(chorales, ragged_rank=1)
    dataset = tf.data.Dataset.from_tensor_slices(chorales)
    dataset = dataset.flat_map(to_windows).map(preprocess)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(create_target)
    return dataset.prefetch(1)


# Now let's create the training set, the validation set and the test set:
#
train_set = bach_dataset(train_chorales, shuffle_buffer_size=1000)
valid_set = bach_dataset(valid_chorales)
test_set = bach_dataset(test_chorales)
# Now let's create the model:
#
# * We could feed the note values directly to the model, as floats, but this would probably not give good results. Indeed, the relationships between notes are not that simple: for example, if you replace a C3 with a C4, the melody will still sound fine, even though these notes are 12 semi-tones apart (i.e., one octave). Conversely, if you replace a C3 with a C\#3, it's very likely that the chord will sound horrible, despite these notes being just next to each other. So we will use an `Embedding` layer to convert each note to a small vector representation (see Chapter 16 for more details on embeddings). We will use 5-dimensional embeddings, so the output of this first layer will have a shape of `[batch_size, window_size, 5]`.
# * We will then feed this data to a small WaveNet-like neural network, composed of a stack of 4 `Conv1D` layers with doubling dilation rates. We will intersperse these layers with `BatchNormalization` layers for faster better convergence.
# * Then one `LSTM` layer to try to capture long-term patterns.
# * And finally a `Dense` layer to produce the final note probabilities. It will predict one probability for each chorale in the batch, for each time step, and for each possible note (including silence). So the output shape will be `[batch_size, window_size, 47]`.
#
n_embedding_dims = 5
model = keras.models.Sequential(
    [
        keras.layers.Embedding(
            input_dim=n_notes, output_dim=n_embedding_dims, input_shape=[None]
        ),
        keras.layers.Conv1D(
            32, kernel_size=2, padding="causal", activation="relu"
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(
            48,
            kernel_size=2,
            padding="causal",
            activation="relu",
            dilation_rate=2,
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(
            64,
            kernel_size=2,
            padding="causal",
            activation="relu",
            dilation_rate=4,
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(
            96,
            kernel_size=2,
            padding="causal",
            activation="relu",
            dilation_rate=8,
        ),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(256, return_sequences=True),
        keras.layers.Dense(n_notes, activation="softmax"),
    ]
)
model.summary()
# Now we're ready to compile and train the model!
#
optimizer = keras.optimizers.Nadam(learning_rate=1e-3)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
model.fit(train_set, epochs=20, validation_data=valid_set)
# I have not done much hyperparameter search, so feel free to iterate on this model now and try to optimize it. For example, you could try removing the `LSTM` layer and replacing it with `Conv1D` layers. You could also play with the number of layers, the learning rate, the optimizer, and so on.
# Once you're satisfied with the performance of the model on the validation set, you can save it and evaluate it one last time on the test set:
#
model.save("my_bach_model.h5")
model.evaluate(test_set)
# **Note:** There's no real need for a test set in this exercise, since we will perform the final evaluation by just listening to the music produced by the model. So if you want, you can add the test set to the train set, and train the model again, hopefully getting a slightly better model.
# Now let's write a function that will generate a new chorale. We will give it a few seed chords, it will convert them to arpegios (the format expected by the model), and use the model to predict the next note, then the next, and so on. In the end, it will group the notes 4 by 4 to create chords again, and return the resulting chorale.
# **Warning**: `model.predict_classes(X)` is deprecated. It is replaced with `np.argmax(model.predict(X), axis=-1)`.
#
def generate_chorale(model, seed_chords, length):
    arpegio = preprocess(tf.constant(seed_chords, dtype=tf.int64))
    arpegio = tf.reshape(arpegio, [1, -1])
    for chord in range(length):
        for note in range(4):
            # next_note = model.predict_classes(arpegio)[:1, -1:]
            next_note = np.argmax(model.predict(arpegio), axis=-1)[:1, -1:]
            arpegio = tf.concat([arpegio, next_note], axis=1)
    arpegio = tf.where(arpegio == 0, arpegio, arpegio + min_note - 1)
    return tf.reshape(arpegio, shape=[-1, 4])


# To test this function, we need some seed chords. Let's use the first 8 chords of one of the test chorales (it's actually just 2 different chords, each played 4 times):
#
seed_chords = test_chorales[2][:8]
play_chords(seed_chords, amplitude=0.2)
# Now we are ready to generate our first chorale! Let's ask the function to generate 56 more chords, for a total of 64 chords, i.e., 16 bars (assuming 4 chords per bar, i.e., a 4/4 signature):
#
new_chorale = generate_chorale(model, seed_chords, 56)
play_chords(new_chorale)
# This approach has one major flaw: it is often too conservative. Indeed, the model will not take any risk, it will always choose the note with the highest score, and since repeating the previous note generally sounds good enough, it's the least risky option, so the algorithm will tend to make notes last longer and longer. Pretty boring. Plus, if you run the model multiple times, it will always generate the same melody.
#
# So let's spice things up a bit! Instead of always picking the note with the highest score, we will pick the next note randomly, according to the predicted probabilities. For example, if the model predicts a C3 with 75% probability, and a G3 with a 25% probability, then we will pick one of these two notes randomly, with these probabilities. We will also add a `temperature` parameter that will control how "hot" (i.e., daring) we want the system to feel. A high temperature will bring the predicted probabilities closer together, reducing the probability of the likely notes and increasing the probability of the unlikely ones.
#
def generate_chorale_v2(model, seed_chords, length, temperature=1):
    arpegio = preprocess(tf.constant(seed_chords, dtype=tf.int64))
    arpegio = tf.reshape(arpegio, [1, -1])
    for chord in range(length):
        for note in range(4):
            next_note_probas = model.predict(arpegio)[0, -1:]
            rescaled_logits = tf.math.log(next_note_probas) / temperature
            next_note = tf.random.categorical(rescaled_logits, num_samples=1)
            arpegio = tf.concat([arpegio, next_note], axis=1)
    arpegio = tf.where(arpegio == 0, arpegio, arpegio + min_note - 1)
    return tf.reshape(arpegio, shape=[-1, 4])


# Let's generate 3 chorales using this new function: one cold, one medium, and one hot (feel free to experiment with other seeds, lengths and temperatures). The code saves each chorale to a separate file. You can run these cells over an over again until you generate a masterpiece!
#
# **Please share your most beautiful generated chorale with me on Twitter @aureliengeron, I would really appreciate it! :))**
#
new_chorale_v2_cold = generate_chorale_v2(
    model, seed_chords, 56, temperature=0.8
)
play_chords(new_chorale_v2_cold, filepath="bach_cold.wav")
#
new_chorale_v2_medium = generate_chorale_v2(
    model, seed_chords, 56, temperature=1.0
)
play_chords(new_chorale_v2_medium, filepath="bach_medium.wav")
# In[100]:
new_chorale_v2_hot = generate_chorale_v2(
    model, seed_chords, 56, temperature=1.5
)
play_chords(new_chorale_v2_hot, filepath="bach_hot.wav")
# Lastly, you can try a fun social experiment: send your friends a few of your favorite generated chorales, plus the real chorale, and ask them to guess which one is the real one!
# In[101]:
play_chords(test_chorales[2][:64], filepath="bach_test_4.wav")
