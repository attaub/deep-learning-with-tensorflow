"""
1. initialize the environment:
   - import numpy for matrix operations.
   - set random seed for reproducibility.
"""


"""
2. prepare the data:
   - collect a dataset of paired english-spanish sentences.
   - tokenize sentences: split into words, convert to lowercase, remove punctuation.
   - build vocabularies for english and spanish (map words to unique indices).
   - create word embeddings (simplified: random vectors for each word).
   - pad or truncate sentences to a fixed length (e.g., max 10 words).
   - convert sentences to sequences of embedding vectors.
   - split data into training, validation, and test sets.
"""

import numpy as np
import string

np.random.seed(42)

# load dataset
with open(
    "./spa.txt", encoding="utf-8"
) as f:  # added encoding for spanish characters
    x = f.read()

x_pair_list = x.split("\n")
x_en_es = []
for pair in x_pair_list:
    if pair.strip():  # skip empty lines
        parts = pair.split("\t")
        if len(parts) >= 2:  # ensure at least two elements
            x_en_es.append([parts[0], parts[1]])

# tokenize
x_pair_tokens = []
for arr in x_en_es:  # fixed: removed [:-1] to include all pairs
    en_sentence = arr[0]
    es_sentence = arr[1]

    en_words = en_sentence.lower().split()
    es_words = es_sentence.lower().split()

    en_strip_punctuations = [
        word.strip(string.punctuation)
        for word in en_words
        if word.strip(string.punctuation)  # filter out empty strings
    ]

    es_strip_punctuations = [
        word.strip(string.punctuation)
        for word in es_words
        if word.strip(string.punctuation)  # filter out empty strings
    ]

    x_pair_tokens.append([en_strip_punctuations, es_strip_punctuations])

# build vocabularies
# initialize sets for unique words
en_unique_words = set()
es_unique_words = set()

# collect unique words from tokenized sentences
for en_tokens, es_tokens in x_pair_tokens:
    en_unique_words.update(en_tokens)  # add all english tokens
    es_unique_words.update(es_tokens)  # add all spanish tokens

# convert sets to sorted lists for consistent indexing
en_unique_words = sorted(list(en_unique_words))
es_unique_words = sorted(list(es_unique_words))

# define special tokens
special_tokens = ['<unk>', '<eos>']  # you can add '<pad>', '<sos>' if needed

# create word-to-index and index-to-word dictionaries
en_word2idx = {
    word: idx for idx, word in enumerate(special_tokens)
}  # start with special tokens
es_word2idx = {word: idx for idx, word in enumerate(special_tokens)}

# add regular words starting from index 2
for word in en_unique_words:
    en_word2idx[word] = len(en_word2idx)  # next available index
for word in es_unique_words:
    es_word2idx[word] = len(es_word2idx)

# create index-to-word dictionaries
en_idx2word = {idx: word for word, idx in en_word2idx.items()}
es_idx2word = {idx: word for word, idx in es_word2idx.items()}

# print vocabulary sizes and sample mappings for inspection
print(f"english vocabulary size: {len(en_word2idx)}")
print(f"spanish vocabulary size: {len(es_word2idx)}")
print(
    "sample english mappings:",
    {k: en_word2idx[k] for k in list(en_word2idx)[:5]},
)
print(
    "sample spanish mappings:",
    {k: es_word2idx[k] for k in list(es_word2idx)[:5]},
)
