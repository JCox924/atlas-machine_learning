#!/usr/bin/env python3
"""
This module contains function(s):
    cummulative_bleu(references, sentence, n):
"""
import math
from collections import Counter


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a candidate sentence.

    The score is computed as the product of the brevity penalty and the geometric
    mean of the n-gram precisions from 1-gram up to n-gram. All n-gram precisions
    are weighted evenly.

    Args:
        references (list[list[str]]): A list of reference translations, where each
            reference is represented as a list of words.
        sentence (list[str]): The candidate sentence as a list of words.
        n (int): The maximum n-gram order to use for evaluation.

    Returns:
        float: The cumulative n-gram BLEU score.
    """
    def get_ngrams(words, gram):
        """
        Generate a list of n-grams from a list of words.

        Args:
            words (list[str]): The list of words.
            gram (int): The size of the n-gram.

        Returns:
            list[tuple]: A list of n-grams represented as tuples.
        """
        return [tuple(words[i:i+gram]) for i in range(len(words) - gram + 1)]

    precisions = []
    for i in range(1, n + 1):
        candidate_ngrams = get_ngrams(sentence, i)
        cand_counts = Counter(candidate_ngrams)

        if len(candidate_ngrams) == 0:
            precisions.append(0)
            continue

        max_ref_counts = {}
        for ref in references:
            ref_ngrams = get_ngrams(ref, i)
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

        precisions.append(clipped_count / len(candidate_ngrams))

    for p in precisions:
        if p == 0:
            geo_mean = 0
            break
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / n)

    c = len(sentence)
    ref_lens = [len(ref) for ref in references]
    best_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))
    bp = 1 if c > best_ref_len else math.exp(1 - best_ref_len / c)

    return bp * geo_mean
