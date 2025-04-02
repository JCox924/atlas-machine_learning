#!/usr/bin/env python3
"""
Bag of Words embedding
"""
import numpy as np
from collections import Counter
import re

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
    words_in_sentences = []
    for sentence in sentences:
        clean_sentence = re.sub(r"'s\b|'\b", '', sentence.lower())
        clean_sentence = re.sub(r"[^\w\s]", '', clean_sentence)
        words = clean_sentence.split()
        words_in_sentences.append(words)

    if vocab is None:
        all_words = [word for words in words_in_sentences for word in words]
        vocab = sorted(set(all_words))
    else:
        cleaned_vocab = []
        for word in vocab:
            word = re.sub(r"'s\b|'\b", '', word.lower())
            word = re.sub(r"[^\w\s]", '', word)
            if word:
                cleaned_vocab.append(word)
        vocab = cleaned_vocab

    features = np.array(vocab)
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, words in enumerate(words_in_sentences):
        word_counts = Counter(words)
        for j, word in enumerate(features):
            embeddings[i, j] = word_counts[word]

    return embeddings, features.tolist()
