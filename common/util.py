import numpy as np
import re

import re

def ptb_contraction_tokenize(text):
    # Common contractions
    text = re.sub(r"(?i)\b(can)(not)\b", r"\1 \2", text)
    text = re.sub(r"(?i)\b(d)('ye)\b", r"\1 \2", text)
    text = re.sub(r"(?i)\b(gim)(me)\b", r"\1 \2", text)
    text = re.sub(r"(?i)\b(gon)(na)\b", r"\1 \2", text)
    text = re.sub(r"(?i)\b(got)(ta)\b", r"\1 \2", text)
    text = re.sub(r"(?i)\b(lem)(me)\b", r"\1 \2", text)
    text = re.sub(r"(?i)\b(mor)('n)\b", r"\1 \2", text)
    text = re.sub(r"(?i)\b(wan)(na)\b", r"\1 \2", text)
    
    # Split ending contractions
    text = re.sub(r"(?i)\b(\w+)'(ll|re|ve|d|m|s)\b", r"\1 '\2", text)
    
    # Split n't
    text = re.sub(r"(?i)\b(\w+)n't\b", r"\1 n't", text)
    
    return text

def split_punctuation(text):
    # Add spaces around punctuation (except apostrophes)
    text = re.sub(r"([!?.,;:()])", r" \1 ", text)
    return text

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
            
    corpus = np.array([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
                
    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return
        
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M