#!/usr/bin/env python3
"""
3-semantic_search.py

Performs semantic search over a corpus of text files using
Universal Sentence Encoder from TensorFlow Hub.
"""

import os
import tensorflow as tf
import tensorflow_hub as hub

# Load the Universal Sentence Encoder model
_USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def semantic_search(corpus_path, sentence):
    """
    Finds and returns the document text in `corpus_path` most similar to `sentence`.

    Args:
        corpus_path (str): path to directory of text files (.md, .txt, etc.).
        sentence (str): query sentence.

    Returns:
        str: full text of the most similar document; None if directory empty.
    """
    # Gather documents
    docs = []
    for fname in sorted(os.listdir(corpus_path)):
        fpath = os.path.join(corpus_path, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, 'r') as file:
            docs.append(file.read())
    if not docs:
        return None

    doc_embeddings = _USE(docs)           # shape (num_docs, embed_dim)
    query_embedding = _USE([sentence])[0] # shape (embed_dim,)

    doc_norms = tf.linalg.norm(doc_embeddings, axis=1)
    query_norm = tf.linalg.norm(query_embedding)
    dots = tf.tensordot(doc_embeddings, query_embedding, axes=1)
    sims = dots / (doc_norms * query_norm)

    best_idx = int(tf.argmax(sims).numpy())
    return docs[best_idx]
