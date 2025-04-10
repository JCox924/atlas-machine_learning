#!/usr/bin/env python3
"""
Module that implements a function to create and train a gensim fastText model.

This module provides the function fasttext_model
which builds and trains a fastText
model on a list of sentences with configurable parameters.
"""

from gensim.models import FastText


def fasttext_model(sentences,
                   vector_size=100,
                   min_count=5,
                   negative=5,
                   window=5,
                   cbow=True,
                   epochs=5,
                   seed=0,
                   workers=1):
    """
    Creates, builds, and trains a gensim fastText model.

    Args:
        sentences (list[list[str]]): A list of tokenized sentences to train on.
        vector_size (int): Dimensionality of the embedding vectors.
        min_count (int): Minimum number of occurrences for a word to be included in training.
        negative (int): Size of negative sampling.
        window (int): Maximum distance between the current and predicted word within a sentence.
        cbow (bool): If True, use Continuous Bag-of-Words (CBOW) model; if False, use Skip-gram.
        epochs (int): Number of training iterations over the corpus.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads to train the model.

    Returns:
        gensim.models.FastText: The trained fastText model.
    """
    sg = 0 if cbow else 1

    model = FastText(vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     workers=workers,
                     sg=sg,
                     seed=seed,
                     negative=negative)

    model.build_vocab(sentences)

    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
