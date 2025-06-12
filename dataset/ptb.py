import os
import urllib.request
import numpy as np

# URL and filenames
url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
key_file = {
    'train': 'ptb.train.txt',
    'test': 'ptb.test.txt',
    'valid': 'ptb.valid.txt'
}
vocab_file = 'ptb.vocab.npy'

dataset_dir = os.path.dirname(os.path.abspath(__file__))


def _download(file_name):
    file_path = os.path.join(dataset_dir, file_name)
    if os.path.exists(file_path):
        return
    print(f'Downloading {file_name}...')
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print('Done')


def _load_vocab():
    vocab_path = os.path.join(dataset_dir, vocab_file)
    if os.path.exists(vocab_path):
        word_to_id = np.load(vocab_path, allow_pickle=True).item()
        id_to_word = {i: w for w, i in word_to_id.items()}
        return word_to_id, id_to_word

    word_to_id = {}
    id_to_word = {}
    for name in key_file:
        path = os.path.join(dataset_dir, key_file[name])
        with open(path, 'r') as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    if word not in word_to_id:
                        new_id = len(word_to_id)
                        word_to_id[word] = new_id
                        id_to_word[new_id] = word

    np.save(vocab_path, word_to_id)
    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    words = []
    with open(filename, 'r') as f:
        for line in f:
            words += line.strip().split()
    return np.array([word_to_id[w] for w in words if w in word_to_id], dtype=np.int32)


def load_data(name='train'):
    if name not in key_file:
        raise ValueError(f"Invalid dataset name: {name}")

    # Download files if necessary
    for file in key_file.values():
        _download(file)

    # Load vocab
    word_to_id, id_to_word = _load_vocab()

    # Convert file to word ID list
    corpus_path = os.path.join(dataset_dir, key_file[name])
    corpus = _file_to_word_ids(corpus_path, word_to_id)

    return corpus, word_to_id, id_to_word
