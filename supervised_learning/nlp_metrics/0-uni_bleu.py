#!/usr/bin/env python3
"""
This module contains function(s):
    uni_bleu(references, sentence)
"""
import math


def uni_bleu(references, sentence):
    """
    Function to calculate unigram bleu score
    Arguments:
        references: reference translations
        sentence: a list containing the model proposed sentence
    Returns:
        The unigram BLEU score
    """

    ground_truth = references.split()
    prediction = sentence.split()

    ref_counts = {}
    for word in ground_truth:
        ref_counts[word] = ref_counts.get(word, 0) + 1

    match = 0
    cand_counts = {}
    for word in prediction:
        cand_counts[word] = cand_counts.get(word, 0) + 1

    for word in cand_counts:
        if word in ref_counts:
            match += min(cand_counts[word], ref_counts[word])

    precision = match / len(prediction)

    ref_len = len(ground_truth)
    cand_len = len(prediction)
    if cand_len > ref_len:
        bp = 1
    else:
        bp = math.exp(1 - ref_len / cand_len)

    bleu1 = bp * precision
    return bleu1
