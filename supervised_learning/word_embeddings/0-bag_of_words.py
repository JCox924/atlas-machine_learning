#!/usr/bin/env python3
"""
Module 0-bag_of_words contains:
    functions:
        - bag_of_words(sentences, vocab=None)
"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Create a bag-of-words embedding matrix for a list of sentences.

    Args:
        sentences (list of str): The sentences to analyze.
        vocab (list of str, optional): A list of vocabulary words to use for the analysis.
                                       If None, all unique words from the sentences are used.

    Returns:
        tuple: A tuple (embeddings, features) where:
            - embeddings (numpy.ndarray): An array of shape (s, f) where s is the number of sentences
              and f is the number of features. Each entry [i, j] contains the count of the j-th feature
              in the i-th sentence.
            - features (list of str): The list of vocabulary words (features) used for the embeddings.
    """
    if vocab is None:
        features_set = set()
        for sentence in sentences:
            words = sentence.split()
            features_set.update(words)
        features = sorted(features_set)
    else:
        features = vocab

    feature_index = {word: idx for idx, word in enumerate(features)}

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for word in words:
            if word in feature_index:
                embeddings[i, feature_index[word]] += 1

    return embeddings, features
