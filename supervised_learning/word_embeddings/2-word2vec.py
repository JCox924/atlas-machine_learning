#!/usr/bin/env python3
"""
Module
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5, negative=5,
                   cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim Word2Vec model.

    Args:
        sentences (list of list of str): The sentences to be trained on.
        vector_size (int): The dimensionality of the embedding layer.
        min_count (int): The minimum number of occurrences of a word for use in training.
        window (int): The maximum distance between the current and predicted word within a sentence.
        negative (int): The size of negative sampling.
        cbow (bool): If True, uses the Continuous Bag-of-Words (CBOW) architecture;
                     if False, uses the Skip-gram architecture.
        epochs (int): The number of iterations (epochs) over the corpus.
        seed (int): The seed for the random number generator.
        workers (int): The number of worker threads to train the model.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(sentences=sentences,
                                   vector_size=vector_size,
                                   window=window,
                                   min_count=min_count,
                                   negative=negative,
                                   sg=sg,
                                   seed=seed,
                                   workers=workers)

    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
