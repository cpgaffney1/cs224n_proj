import time
import numpy as np
from collections import Counter
import parser_util
import codecs

class Config(object):
    large_embedding_file = 'data//glove.6B.100d.txt'
    xl_embedding_file = 'data//glove.6B.300d.txt'
    small_embedding_file = 'data//glove.6B.50d.txt'
    unlabeled = True
    lowercase = True
    use_pos = True
    use_dep = True
    use_dep = use_dep and (not unlabeled)


embed_size = 0
vocab_size = 0
UNK = '<unk>'
START = '<start>'
END = '<end>'
PAD = '<pad>'
tok2id = None
id2tok = None

def get_id_mapping(document):
    global UNK
    global START
    global END
    global PAD

    tok2id = {}
    # Build dictionary for words
    tok2id.update(build_dict([w for sentence in document for w in sentence.split(' ')]))
    tok2id[UNK] = len(tok2id)
    tok2id[PAD] = len(tok2id)
    tok2id[START] = len(tok2id)
    tok2id[END] = len(tok2id)

    id2tok = {v: k for (k, v) in tok2id.items()}

    return tok2id, id2tok


def build_dict(keys):
    count = Counter()
    for key in keys:
        count[key] += 1
    ls = sorted(count.most_common(), key=lambda x: (-x[1], x[0]))
    return {w[0]: index for (index, w) in enumerate(ls)}

def get_embeddings_matrix(n_tokens, large=False):
    global embed_size
    config = Config()
    embed_size = -1
    word_vectors = {}
    if large:
        embed_file = config.large_embedding_file
    else:
        embed_file = config.small_embedding_file
    for line in codecs.open(embed_file, encoding='utf8', errors='ignore').readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
        embed_size = len(word_vectors[sp[0]])
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (n_tokens, embed_size)), dtype='float32')
    return embeddings_matrix, word_vectors

def load_embeddings(large=False, mode='full'):
    print("Loading pretrained embeddings...")
    start = time.time()
    global embed_size
    global vocab_size
    global tok2id
    global id2tok

    normal, simple = parser_util.parse_pwkp(mode=mode)
    tok2id, id2tok = get_id_mapping(normal + simple)

    embeddings_matrix, word_vectors = get_embeddings_matrix(len(tok2id), large=large)

    print('Loading word vector mapping...')


    for token in tok2id:
        i = tok2id[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]
            embed_size = len(embeddings_matrix[i])
        elif token.lower() in word_vectors:
            embeddings_matrix[i] = word_vectors[token.lower()]

    print("took {:.2f} seconds".format(time.time() - start))
    vocab_size = len(embeddings_matrix)

    return embeddings_matrix, normal, simple, tok2id, id2tok

