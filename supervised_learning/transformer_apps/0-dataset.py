#!/usr/bin/env python3
"""
0-dataset.py

Loads and prepares the TED‑HRLR Portuguese→English dataset
for machine translation. Creates sub-word tokenizers
based on pretrained BERT tokenizers, limited to 2**13 tokens.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import BertTokenizerFast


class Dataset:
    """
    Loads the 'ted_hrlr_translate/pt_to_en' dataset and
    builds Portuguese and English tokenizers.
    """

    def __init__(self):
        """
        - Loads train and validation splits (as_supervised=True).
        - Builds subword tokenizers from the *training* split,
          each capped at 2**13 tokens.
        """
        # Load the raw datasets
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        # Build tokenizers on the training data
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates and trains two BertTokenizerFast instances:
        - Portuguese:  pretrained on 'neuralmind/bert-base-portuguese-cased'
        - English:     pretrained on 'bert-base-uncased'
        Both are retrained on `data` up to vocab_size=2**13.

        Args:
            data (tf.data.Dataset): yields (pt, en) tf.Tensor pairs.

        Returns:
            tokenizer_pt (BertTokenizerFast),
            tokenizer_en (BertTokenizerFast)
        """
        # Load pretrained fast tokenizers
        tokenizer_pt = BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        # Generators over the raw text
        def pt_iter():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_iter():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        # Retrain on our dataset (max vocab size 8192)
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iter(),
            vocab_size=2**13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iter(),
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
