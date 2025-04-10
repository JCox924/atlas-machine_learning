#!/usr/bin/env python3
"""
This module contains function(s):
    ngram_bleu(references, sentence, n)
        - calculates the n-gram BLEU score for a sentence
"""
import math
from collections import Counter


def ngram_bleu(references, sentence, n):
    """
    
    :param references: list of reference translations
    :param sentence: list containing the model proposed sentence
    :param n: size of the n-gram to use for evaluation
    :return: the n-gram BLEU score
    """
    def get_ngrams(words, n):
        """
        Generate n-grams from a list of words.

        Args:
            words (list[str]): The list of words.
            n (int): The size of the n-gram.

        Returns:
            list[tuple]: A list of n-grams represented as tuples.
        """
        return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]

    sentence_ngrams = get_ngrams(sentence, n)
    cand_counts = Counter(sentence_ngrams)

    if len(sentence_ngrams) == 0:
        return 0.0

    max_ref_counts = {}
    for ref in references:
        ref_ngrams = get_ngrams(ref, n)
        ref_counts = Counter(ref_ngrams)
        for ngram, count in ref_counts.items():
            if ngram in max_ref_counts:
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
            else:
                max_ref_counts[ngram] = count

    clipped_count = 0
    for ngram, count in cand_counts.items():
        if ngram in max_ref_counts:
            clipped_count += min(count, max_ref_counts[ngram])

    precision = clipped_count / len(sentence_ngrams)

    c = len(sentence)
    ref_lens = [len(ref) for ref in references]
    best_ref_len = min(ref_lens,
                       key=lambda ref_len: (abs(ref_len - c), ref_len))
    bp = 1 if c > best_ref_len else math.exp(1 - best_ref_len / c)

    return bp * precision
