import sys
import os
sys.path.append('..')  # So it can find common and dataset modules

import numpy as np
from common.util import (
    most_similar,
    create_co_matrix,
    ppmi
)
from dataset.ptb import load_data

# Parameters
window_size = 2
wordvec_size = 100

# Load the PTB dataset
corpus, word_to_id, id_to_word = load_data('train')
vocab_size = len(word_to_id)

# Show dataset info
print(f"Corpus size: {len(corpus)}")
print(f"Vocabulary size: {vocab_size}")

# Step 1: Count co-occurrences
print('Counting co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)

# Step 2: Calculate PPMI matrix
print('Calculating PPMI ...')
W = ppmi(C, verbose=True)

# Step 3: Reduce dimensionality with SVD
print('Calculating SVD ...')
try:
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=42)
except ImportError:
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

# Step 4: Test queries
query_words = ['you', 'year', 'car']
for query in query_words:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
