#!/usr/bin/env python3
import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Create a TF-IDF embedding matrix for a list of sentences.

    Args:
        sentences (list of str): The sentences to analyze.
        vocab (list of str, optional): A list of vocabulary words to use for the analysis.
            If None, all unique words from the sentences are used.
            Words are normalized to lowercase and punctuation is removed.

    Returns:
        tuple: A tuple (embeddings, features) where:
            - embeddings (numpy.ndarray): An array of shape (s, f) where s is the number of sentences and
              f is the number of features. Each entry [i, j] contains the normalized TF-IDF value for the j-th feature
              in the i-th sentence.
            - features (list): The list of vocabulary words (features) used for the embeddings.
              If vocab is provided, the same order is preserved; otherwise, features are sorted alphabetically.

    TF-IDF is computed using the smoothed formula:
        idf(word) = log((1 + N) / (1 + df(word))) + 1
    where N is the number of sentences and df(word) is the number of sentences containing the word.
    Each sentence’s TF-IDF vector is then L2 normalized.
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

    N = len(sentences)

    df = np.zeros(len(features), dtype=float)
    for words in processed_sentences:
        unique_words = set(words)
        for word in unique_words:
            if word in feature_to_index:
                df[feature_to_index[word]] += 1

    idf = np.log((1 + N) / (1 + df)) + 1

    embeddings = np.zeros((N, len(features)), dtype=float)

    for i, words in enumerate(processed_sentences):
        tf = {}
        for word in words:
            if word in feature_to_index:
                tf[word] = tf.get(word, 0) + 1
        for word, count in tf.items():
            j = feature_to_index[word]
            embeddings[i, j] = count * idf[j]

        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] /= norm

    return embeddings, features
