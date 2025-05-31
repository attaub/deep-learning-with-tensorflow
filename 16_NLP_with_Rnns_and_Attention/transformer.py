import math
import numpy as np

# step 1: input embedding (simulate random word embeddings)
input_embeddings = np.random.rand(4, 6)  # (seq_len, embed_dim)

# step 2: positional encoding
def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
    return pe

pos_enc = positional_encoding(seq_len=4, d_model=6)
x = input_embeddings + pos_enc  # Add position info to embeddings

# step 3: scaled dot-product attention
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / math.sqrt(d_k)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)  # softmax
    output = np.matmul(weights, V)
    return output

# step 4: multi-head attention
def multi_head_attention(x, num_heads=2):
    d_model = x.shape[1]
    d_head = d_model // num_heads
    heads = []

    for _ in range(num_heads):
        # random linear projections (simulate with random matrices for demo)
        W_Q = np.random.rand(d_model, d_head)
        W_K = np.random.rand(d_model, d_head)
        W_V = np.random.rand(d_model, d_head)

        Q = np.dot(x, W_Q)
        K = np.dot(x, W_K)
        V = np.dot(x, W_V)

        head = scaled_dot_product_attention(Q, K, V)
        heads.append(head)

    concat = np.concatenate(heads, axis=-1)  # (seq_len, d_model)
    return concat

# step 5: feed forward network
def feed_forward(x):
    # two-layer feed-forward network
    W1 = np.random.rand(x.shape[1], x.shape[1]*2)
    b1 = np.random.rand(x.shape[1]*2)
    W2 = np.random.rand(x.shape[1]*2, x.shape[1])
    b2 = np.random.rand(x.shape[1])

    return np.dot(np.maximum(0, np.dot(x, W1) + b1), W2) + b2

# step 6: layer normalization
def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

# putting it all together:
# input with positional encoding
x = input_embeddings + positional_encoding(4, 6)

# multi-head attention
attn_output = multi_head_attention(x)

# add & norm
x = layer_norm(x + attn_output)

# feed forward
ff_output = feed_forward(x)

# add & norm
output = layer_norm(x + ff_output)

print("Final encoder output:\n", output)
