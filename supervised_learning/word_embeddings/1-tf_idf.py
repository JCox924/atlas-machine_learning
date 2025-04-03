#!/usr/bin/env python3
""" Term Frequency-Inverse Document Frequency """
import numpy as np
from collections import Counter
import re
import math


def tf_idf(sentences, vocab=None):
    """ Create a TF-IDF embedding matrix."""
    # preprocess to get words
    words_in_sentences = []
    for sentence in sentences:
        clean_sentence = re.sub(r'\'s\b|\'\b', '', sentence.lower())
        clean_sentence = re.sub(r'[^\w\s]', '', clean_sentence)
        words = clean_sentence.split()
        words_in_sentences.append(words)

    if vocab is None:
        # flatten the list of words from all sentences
        all_words = [word for words in words_in_sentences for word in words]
        # unique words, sort for consistency
        vocab = sorted(set(all_words))
    else:
        # vocab is provided, clean it too to be consistent
        cleaned_vocab = []
        for word in vocab:
            word = re.sub(r'\'s\b|\'\b', '', word.lower())
            word = re.sub(r'[^\w\s]', '', word)
            if word:  # Only add non-empty strings
                cleaned_vocab.append(word)
        # remove duplicates that might result from cleaning
        # don't sort?
        vocab = cleaned_vocab

    # convert features to a numpy array
    features = np.array(vocab)

    # calculate document frequency for each word
    document_frequency = {}
    for word in features:
        document_frequency[word] = sum(1 for sentence_words in
                                       words_in_sentences
                                       if word in set(sentence_words))

    # total number of documents (sentences)
    num_documents = len(sentences)

    # init embedding matrix with zeros (as FLOATS for TF-IDF values)
    embeddings = np.zeros((len(sentences), len(features)), dtype=np.float64)

    # fill embedding matrix with TF-IDF values
    for i, words in enumerate(words_in_sentences):
        # get unique words for this document (sentence)
        unique_words = set(words)

        # get total count of words in this sentence
        total_words = len(words)

        # count occurrences of each word (term frequency)
        word_counts = Counter(words)

        # calculate TF-IDF for each word in the vocabulary
        for j, word in enumerate(features):
            if word in word_counts:
                # term frequency is the count in this document
                tf = word_counts[word] / total_words

                # inverse document frequency:
                # log((1 + N) / (1 + df)) + 1
                idf = math.log((1 + num_documents) / (1 +
                               document_frequency[word])) + 1

                # TF-IDF value
                embeddings[i, j] = tf * idf

    for i in range(len(sentences)):
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:  # no division by zero
            embeddings[i] = embeddings[i] / norm

    return embeddings, features
