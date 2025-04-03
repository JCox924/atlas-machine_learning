#!/usr/bin/env python3
"""
Bag of Words embedding
"""
import numpy as np

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences (list): list of sentences to analyze
        vocab (list): vocabulary words to use; if None, extract from sentences

    Returns:
        embeddings (np.ndarray): shape (s, f), the embeddings matrix
        features (list): list of features used (words)
    """
    if vocab is None:
        flag = 1
        vocab = []
    else:
        flag = 0

    split_sentences = []

    for i in range(len(sentences)):
        split_sentence = []

        lower = sentences[i].lower()

        doc_vocab = lower.split(' ')
        for token in doc_vocab:

            # remove any 's
            if token.endswith("'s"):
                token = token[:-2]
            token = ''.join(
                filter(
                    lambda x: x.islower() or x.isspace(),
                    token))

            split_sentence.append(token)

            if flag == 1:
                if token not in vocab:
                    vocab.append(token)
        split_sentences.append(split_sentence)
    if flag == 1:
        vocab = sorted(vocab)

    embeddings = np.zeros((len(sentences), len(vocab)))

    for x in range(embeddings.shape[0]):
        for y in range(len(split_sentences[x])):
            if split_sentences[x][y] in vocab:
                embeddings[x][vocab.index(split_sentences[x][y])] += 1

    return embeddings.astype(int), vocab
