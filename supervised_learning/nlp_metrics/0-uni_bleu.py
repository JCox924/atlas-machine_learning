#!/usr/bin/env python3
"""
This module contains function(s):
    uni_bleu(references, sentence)
"""

import math
from collections import Counter


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a candidate sentence.

    The score is computed as the product
    of the brevity penalty and the unigram
    precision (clipped count).

    Args:
        references (list[list[str]]): A list of
            reference translations, where each
            reference is represented as a list of words.
        sentence (list[str]): The candidate sentence as a list of words.

    Returns:
        float: The unigram BLEU score.
    """
    # Count candidate unigrams
    cand_counts = Counter(sentence)

    max_ref_counts = {}
    for ref in references:
        ref_counts = Counter(ref)
        for word, count in ref_counts.items():
            if word in max_ref_counts:
                max_ref_counts[word] = max(max_ref_counts[word], count)
            else:
                max_ref_counts[word] = count

    clipped_count = 0
    for word, count in cand_counts.items():
        if word in max_ref_counts:
            clipped_count += min(count, max_ref_counts[word])

    # Calculate unigram precision
    total_candidate_unigrams = len(sentence)
    if total_candidate_unigrams == 0:
        precision = 0.0
    else:
        precision = clipped_count / total_candidate_unigrams

    c = len(sentence)
    ref_lens = [len(ref) for ref in references]
    best_ref_len = min(ref_lens,
                       key=lambda ref_len: (abs(ref_len - c), ref_len))
    bp = 1 if c > best_ref_len else math.exp(1 - best_ref_len / c)

    return bp * precision

