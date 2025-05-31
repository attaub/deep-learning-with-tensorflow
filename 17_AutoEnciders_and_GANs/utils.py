import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

#################################################################
# A couple utility functions to plot grayscale 28x28 image:
def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


def show_reconstructions(model, images, n_images=5):
    # def show_reconstructions(model, images=X_valid, n_images=5):
    reconstructions = model.predict(images[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
    plt.show()

def plot_percent_hist(ax, data, bins):
    counts, _ = np.histogram(data, bins=bins)
    widths = bins[1:] - bins[:-1]
    x = bins[:-1] + widths / 2
    ax.bar(x, counts / len(data), width=widths * 0.8)
    ax.xaxis.set_ticks(bins)
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(
            lambda y, position: "{}%".format(int(np.round(100 * y)))
        )
    )
    ax.grid(True)


def plot_activations_histogram(encoder, height=1, n_bins=10):
    X_valid_codings = encoder(X_valid).numpy()
    activation_means = X_valid_codings.mean(axis=0)
    mean = activation_means.mean()
    bins = np.linspace(0, 1, n_bins + 1)

    fig, [ax1, ax2] = plt.subplots(
        figsize=(10, 3), nrows=1, ncols=2, sharey=True
    )
    plot_percent_hist(ax1, X_valid_codings.ravel(), bins)
    ax1.plot(
        [mean, mean],
        [0, height],
        "k--",
        label="Overall Mean = {:.2f}".format(mean),
    )
    ax1.legend(loc="upper center", fontsize=14)
    ax1.set_xlabel("Activation")
    ax1.set_ylabel("% Activations")
    ax1.axis([0, 1, 0, height])
    plot_percent_hist(ax2, activation_means, bins)
    ax2.plot([mean, mean], [0, height], "k--")
    ax2.set_xlabel("Neuron Mean Activation")
    ax2.set_ylabel("% Neurons")
    ax2.axis([0, 1, 0, height])



# Define the training function for the GAN
def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):

    # extract the generator
    generator, discriminator = gan.layers

    # loop over the specified number of epochs
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))

        # loop over batches in the dataset
        for X_batch in dataset:

            ####################################################
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])

            # generate input 
            generated_images = generator(noise)

            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)

            # labels 
            y1 = tf.constant([[0.0]] * batch_size + [[1.0]] * batch_size)

            discriminator.trainable = True

            # train the discriminator 
            discriminator.train_on_batch(X_fake_and_real, y1)

            ####################################################
            # phase 2 - training the generator

            # generate new random noise
            noise = tf.random.normal(shape=[batch_size, codings_size])

            # create label 1, try to fool discriminator
            y2 = tf.constant([[1.0]] * batch_size)

            # freeze the discriminator weights 
            discriminator.trainable = False

            # train the GAN (only update generator weights) to make
            # discriminator output 1 for fake images
            gan.train_on_batch(noise, y2)

        plot_multiple_images(generated_images, 8)  # not shown
        plt.show()
