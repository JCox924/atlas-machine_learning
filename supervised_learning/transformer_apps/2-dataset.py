#!/usr/bin/env python3
"""
2-dataset.py

Loads and tokenizes the TED‑HRLR Portuguese→English dataset,
adding a TensorFlow wrapper for encoding and mapping it
over train/validation splits.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Loads the 'ted_hrlr_translate/pt_to_en' dataset, builds
    sub-word tokenizers, and provides tf_encode for mapping.
    """

    def __init__(self):
        """
        - Loads train and validation splits (as_supervised=True).
        - Builds tokenizers on the training set.
        - Maps tf_encode over train and validation datasets.
        """
        # Load raw datasets
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

        # Build tokenizers on training data
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        # Tokenize examples with tf wrapper
        self.data_train = self.data_train.map(
            self.tf_encode,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        self.data_valid = self.data_valid.map(
            self.tf_encode,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    def tokenize_dataset(self, data):
        """
        Creates and trains two BertTokenizerFast instances:
        - Portuguese: pretrained on 'neuralmind/bert-base-portuguese-cased'
        - English:    pretrained on 'bert-base-uncased'
        Both are retrained on `data` up to vocab_size=2**13.

        Args:
            data (tf.data.Dataset): yields (pt, en) tf.Tensor pairs.

        Returns:
            tokenizer_pt (BertTokenizerFast),
            tokenizer_en (BertTokenizerFast)
        """
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        def pt_iter():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_iter():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iter(),
            vocab_size=2**13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iter(),
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a single (pt, en) pair into integer tokens.

        - pt, en: tf.Tensor strings.
        - Adds a start token (id = vocab_size) and
          end token (id = vocab_size + 1).

        Returns:
            pt_tokens (list of int),
            en_tokens (list of int)
        """
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_ids = self.tokenizer_pt.encode(pt_text, add_special_tokens=False)
        en_ids = self.tokenizer_en.encode(en_text, add_special_tokens=False)

        start_pt = self.tokenizer_pt.vocab_size
        end_pt = start_pt + 1
        start_en = self.tokenizer_en.vocab_size
        end_en = start_en + 1

        pt_tokens = [start_pt] + pt_ids + [end_pt]
        en_tokens = [start_en] + en_ids + [end_en]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Wrapper around `encode` for tf.data.Dataset mapping.

        Args:
            pt (tf.Tensor): Portuguese sentence tensor.
            en (tf.Tensor): English sentence tensor.

        Returns:
            (tf.Tensor[int64], tf.Tensor[int64]): token sequences
            with dynamic shape ([None],).
        """
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens
