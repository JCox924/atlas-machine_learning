#!/usr/bin/env python3
"""
Module 0-bag_of_words contains:
    functions:
        - bag_of_words(sentences, vocab=None)
"""
import re
import numpy as np

def bag_of_words(sentences, vocab=None):
    """
    Create a bag-of-words embedding matrix for a list of sentences.

    Args:
        sentences (list of str): The sentences to analyze.
        vocab (list of str, optional): A list of vocabulary words to use for the analysis.
                                       If None, all unique words from the sentences are used.
                                       Words are normalized to lowercase and punctuation is removed.

    Returns:
        tuple: A tuple (embeddings, features) where:
            - embeddings (numpy.ndarray): An array of shape (s, f) where s is the number of sentences
              and f is the number of features. Each entry [i, j] contains the count of the j-th feature
              in the i-th sentence.
            - features (list of str): The list of vocabulary words (features) used for the embeddings,
              sorted in alphabetical order.
    """
    processed_sentences = []
    all_words = []
    for sentence in sentences:
        words = re.findall(r'\b[a-z]+\b', sentence.lower())
        words = [word for word in words if word != 's']
        processed_sentences.append(words)
        all_words.extend(words)

    if vocab is None:
        features = sorted(set(all_words))
    else:
        features = vocab

    feature_to_index = {word: idx for idx, word in enumerate(features)}

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, words in enumerate(processed_sentences):
        for word in words:
            if word in feature_to_index:
                embeddings[i, feature_to_index[word]] += 1

    return embeddings, features
