import tensorflow as tf

# Define a small vocabulary and tokenizer

vocab = ["", "I", "love", "AI"]  # "" for padding token

text_vec_layer = tf.keras.layers.TextVectorization(
    vocabulary_size=5, output_sequence_length=10
)

# tokenize the sentence
sentence = ["I love AI"]
text_vec_layer_en.adapt(sentence)
token_ids = text_vec_layer(sentence)  # Output: [[1, 2, 3]]

# embedding layer
embed_size = 5
embedding_layer = tf.keras.layers.Embedding(
    input_dim=len(vocab), output_dim=embed_size
)

# Get embeddings
embeddings = embedding_layer(token_ids)
print(
    embeddings.shape
)  # Output: (1, 3, 5) because batch_size=1, sequence_length=3, embed_size=5
print(embeddings[0])  # Tensor of shape (3, 5) for the sentence
