#!/usr/bin/env python3
"""
Module that implements a bag-of-words embedding matrix generator.

This module provides a function bag_of_words that converts a list of sentences
into a bag-of-words embedding matrix using a specified or derived vocabulary.
"""

import numpy as np
import re
import string


def preprocess_word(word):
    """
    Preprocess a word by converting it to lowercase,
    removing trailing "'s", and stripping punctuation.

    Args:
        word (str): The input word.

    Returns:
        str: The processed word.
    """
    word = word.lower()
    if word.endswith("'s"):
        word = word[:-2]
    word = word.translate(str.maketrans('', '', string.punctuation))
    return word


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag-of-words embedding matrix from a list of sentences.

    Each sentence is tokenized and processed, then
    represented as a frequency vector
    over the vocabulary. If vocab is None, all unique
    words (after processing) from the sentences
    are used, sorted alphabetically.

    Args:
        sentences (list[str]): A list of sentences to analyze.
        vocab (list[str] or None): A list of vocabulary words to use. If None, all words
            found in the sentences are used.

    Returns:
        tuple: A tuple (embeddings, features) where
            embeddings is a numpy.ndarray of shape
               (number of sentences, number of features)
               containing the embeddings, and
               features is a list of the vocabulary words
                used for the analysis.
    """
    processed_sentences = []
    all_words = []
    for sentence in sentences:
        tokens = sentence.split()
        processed_tokens = []
        for token in tokens:
            p_token = preprocess_word(token)
            if p_token != "":
                processed_tokens.append(p_token)
        processed_sentences.append(processed_tokens)
        all_words.extend(processed_tokens)

    if vocab is None:
        features = sorted(list(set(all_words)))
    else:
        features = [preprocess_word(word) for word in vocab]

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)
    for i, tokens in enumerate(processed_sentences):
        for token in tokens:
            if token in features:
                j = features.index(token)
                embeddings[i, j] += 1

    return embeddings, features
