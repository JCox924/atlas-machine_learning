#!/usr/bin/env python3
""" Term Frequency-Inverse Document Frequency """
import numpy as np
from collections import Counter
import re
import math


def tf_idf(sentences, vocab=None):
    """ Create a TF-IDF embedding matrix."""
    words_in_sentences = []
    for sentence in sentences:
        clean_sentence = re.sub(r'\'s\b|\'\b', '', sentence.lower())
        clean_sentence = re.sub(r'[^\w\s]', '', clean_sentence)
        words = clean_sentence.split()
        words_in_sentences.append(words)

    if vocab is None:
        all_words = [word for words in words_in_sentences for word in words]
        vocab = sorted(set(all_words))
    else:
        cleaned_vocab = []
        for word in vocab:
            word = re.sub(r'\'s\b|\'\b', '', word.lower())
            word = re.sub(r'[^\w\s]', '', word)
            if word:
                cleaned_vocab.append(word)
        vocab = cleaned_vocab

    features = np.array(vocab)

    document_frequency = {}
    for word in features:
        document_frequency[word] = sum(1 for sentence_words in
                                       words_in_sentences
                                       if word in set(sentence_words))

    num_documents = len(sentences)

    embeddings = np.zeros((len(sentences), len(features)), dtype=np.float64)

    for i, words in enumerate(words_in_sentences):
        unique_words = set(words)

        total_words = len(words)

        word_counts = Counter(words)

        for j, word in enumerate(features):
            if word in word_counts:
                # term frequency is the count in this document
                tf = word_counts[word] / total_words

                idf = math.log((1 + num_documents) / (1 +
                               document_frequency[word])) + 1

                embeddings[i, j] = tf * idf

    for i in range(len(sentences)):
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:  # no division by zero
            embeddings[i] = embeddings[i] / norm

    return embeddings, features