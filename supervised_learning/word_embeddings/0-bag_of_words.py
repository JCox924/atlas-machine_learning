#!/usr/bin/env python3
"""
Bag of Words embedding
"""
from sklearn.feature_extraction.text import CountVectorizer

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
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names()
    else:
        vectorizer = CountVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)
    embedding = X.toarray()

    return embedding, vocab
