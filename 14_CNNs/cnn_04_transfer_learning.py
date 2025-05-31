import sys
import numpy as np 
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images
import tensorflow as tf
#################################################################
# # Pretrained Models for Transfer Learning
import tensorflow_datasets as tfds

dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size
class_names
n_classes
test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
    as_supervised=True,
)
# extra code – displays the first 9 images in the validation set
plt.figure(figsize=(12, 10))
index = 0
for image, label in valid_set_raw.take(9):
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title(f"Class: {class_names[label]}")
    plt.axis("off")
plt.show()
# All three datasets contain individual images. We need to batch them, but for this we first need to ensure they all have the same size, or else batching will not work. We can use a `Resizing` layer for this. We must also call the `tf.keras.applications.xception.preprocess_input()` function to preprocess the images appropriately for the Xception model. We will also add shuffling and prefetching to the training dataset.
tf.keras.backend.clear_session()  # extra code – resets layer name counter
batch_size = 32
preprocess = tf.keras.Sequential(
    [
        tf.keras.layers.Resizing(
            height=224, width=224, crop_to_aspect_ratio=True
        ),
        tf.keras.layers.Lambda(
            tf.keras.applications.xception.preprocess_input
        ),
    ]
)
train_set = train_set_raw.map(lambda X, y: (preprocess(X), y))
train_set = train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(lambda X, y: (preprocess(X), y)).batch(
    batch_size
)
test_set = test_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)
# Let's take a look again at the first 9 images from the validation set: they're all 224x224 now, with values ranging from -1 to 1:
# extra code – displays the first 9 images in the first batch of valid_set
plt.figure(figsize=(12, 12))
for X_batch, y_batch in valid_set.take(1):
    for index in range(9):
        plt.subplot(3, 3, index + 1)
        plt.imshow((X_batch[index] + 1) / 2)  # rescale to 0–1 for imshow()
        plt.title(f"Class: {class_names[y_batch[index]]}")
        plt.axis("off")
plt.show()
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
        tf.keras.layers.RandomRotation(factor=0.05, seed=42),
        tf.keras.layers.RandomContrast(factor=0.2, seed=42),
    ]
)
# Try running the following cell multiple times to see different random data augmentations:
# extra code – displays the same first 9 images, after augmentation
plt.figure(figsize=(12, 12))
for X_batch, y_batch in valid_set.take(1):
    X_batch_augmented = data_augmentation(X_batch, training=True)
    for index in range(9):
        plt.subplot(3, 3, index + 1)
        # We must rescale the images to the 0-1 range for imshow(), and also
        # clip the result to that range, because data augmentation may
        # make some values go out of bounds (e.g., RandomContrast in this case).
        plt.imshow(np.clip((X_batch_augmented[index] + 1) / 2, 0, 1))
        plt.title(f"Class: {class_names[y_batch[index]]}")
        plt.axis("off")
plt.show()
# Now let's load the pretrained model, without its top layers, and replace them with our own, for the flower classification task:
tf.random.set_seed(42)  # extra code – ensures reproducibility
base_model = tf.keras.applications.xception.Xception(
    weights="imagenet", include_top=False
)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
for layer in base_model.layers:
    layer.trainable = False
# Let's train the model for a few epochs, while keeping the base model weights fixed:
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
history = model.fit(train_set, validation_data=valid_set, epochs=3)
for indices in zip(range(33), range(33, 66), range(66, 99), range(99, 132)):
    for idx in indices:
        print(f"{idx:3}: {base_model.layers[idx].name:22}", end="")
    print()
# Now that the weights of our new top layers are not too bad, we can make the top part of the base model trainable again, and continue training, but with a lower learning rate:
for layer in base_model.layers[56:]:
    layer.trainable = True
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
history = model.fit(train_set, validation_data=valid_set, epochs=10)
# # Classification and Localization
# **Note**: the code below used to reuse the optimizer from the previous model. This was fine in earlier versions of TensorFlow, but in more recent versions it can cause some issues, so I added a line to create a new optimizer here. Also, recent versions of Keras expect one metric per output, so I added the `"mse"` metric to the list.
tf.random.set_seed(42)  # extra code – ensures reproducibility
base_model = tf.keras.applications.xception.Xception(
    weights="imagenet", include_top=False
)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
class_output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
loc_output = tf.keras.layers.Dense(4)(avg)
model = tf.keras.Model(
    inputs=base_model.input, outputs=[class_output, loc_output]
)
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9
)  # added this line
model.compile(
    loss=["sparse_categorical_crossentropy", "mse"],
    loss_weights=[0.8, 0.2],  # depends on what you care most about
    optimizer=optimizer,
    metrics=["accuracy", "mse"],
)
# extra code – fits the model using random target bounding boxes (in real life
#              you would need to create proper targets instead)
def add_random_bounding_boxes(images, labels):
    fake_bboxes = tf.random.uniform([tf.shape(images)[0], 4])
    return images, (labels, fake_bboxes)


fake_train_set = train_set.take(5).repeat(2).map(add_random_bounding_boxes)
model.fit(fake_train_set, epochs=2)
# # Extra Material – How mAP Relates to Precision/Recall
def maximum_precisions(precisions):
    return np.flip(np.maximum.accumulate(np.flip(precisions)))


recalls = np.linspace(0, 1, 11)
precisions = [0.91, 0.94, 0.96, 0.94, 0.95, 0.92, 0.80, 0.60, 0.45, 0.20, 0.10]
max_precisions = maximum_precisions(precisions)
mAP = max_precisions.mean()
plt.plot(recalls, precisions, "ro--", label="Precision")
plt.plot(recalls, max_precisions, "bo-", label="Max Precision")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot([0, 1], [mAP, mAP], "g:", linewidth=3, label="mAP")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower center")
plt.show()

