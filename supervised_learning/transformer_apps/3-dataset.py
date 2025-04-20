#!/usr/bin/env python3
"""
3-dataset.py

Loads, tokenizes, filters, batches, and prefetches the
TED‑HRLR Portuguese→English dataset for training/validation.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Builds sub-word tokenizers and sets up tf.data pipelines
    with filtering, caching, shuffling, padding, and prefetching.
    """

    def __init__(self, batch_size, max_len):
        """
        - batch_size:   size of each padded batch
        - max_len:      maximum tokens allowed per sentence

        After loading and tokenizing, applies to train:
          • filter lengths ≤ max_len
          • cache
          • shuffle(buffer_size=20000)
          • padded_batch(batch_size)
          • prefetch(AUTOTUNE)

        And to validation:
          • filter lengths ≤ max_len
          • padded_batch(batch_size)
        """
        self.batch_size = batch_size
        self.max_len = max_len

        # 1) Load raw splits
        raw_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        raw_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        # 2) Build tokenizers on raw training data
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(raw_train)

        # 3) Map tokenization into tf.Tensors
        train_tokens = raw_train.map(
            self.tf_encode,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        valid_tokens = raw_valid.map(
            self.tf_encode,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # 4) Training pipeline
        self.data_train = (
            train_tokens
            # filter out too-long sequences
            .filter(self._filter_by_max_len)
            .cache()
            .shuffle(buffer_size=20000)
            .padded_batch(
                batch_size,
                padded_shapes=([None], [None])
            )
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        # 5) Validation pipeline
        self.data_valid = (
            valid_tokens
            .filter(self._filter_by_max_len)
            .padded_batch(
                batch_size,
                padded_shapes=([None], [None])
            )
        )

    def tokenize_dataset(self, data):
        """
        From Task 0: trains two BertTokenizerFast on `data`
        (up to vocab_size=2**13) and returns them.
        """
        tok_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tok_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        def pt_iter():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_iter():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tok_pt = tok_pt.train_new_from_iterator(pt_iter(), vocab_size=2**13)
        tok_en = tok_en.train_new_from_iterator(en_iter(), vocab_size=2**13)
        return tok_pt, tok_en

    def encode(self, pt, en):
        """
        From Task 1: pure-Python encode to lists of ints,
        adding start/end tokens.
        """
        pt_str = pt.numpy().decode('utf-8')
        en_str = en.numpy().decode('utf-8')

        pt_ids = self.tokenizer_pt.encode(pt_str, add_special_tokens=False)
        en_ids = self.tokenizer_en.encode(en_str, add_special_tokens=False)

        s_pt = self.tokenizer_pt.vocab_size
        e_pt = s_pt + 1
        s_en = self.tokenizer_en.vocab_size
        e_en = s_en + 1

        return [s_pt] + pt_ids + [e_pt], [s_en] + en_ids + [e_en]

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper around `encode`, setting dynamic shape.
        """
        pt_t, en_t = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_t.set_shape([None])
        en_t.set_shape([None])
        return pt_t, en_t

    def _filter_by_max_len(self, pt, en):
        """
        Keep only pairs where both token sequences are ≤ max_len.
        """
        return tf.logical_and(
            tf.shape(pt)[0] <= self.max_len,
            tf.shape(en)[0] <= self.max_len
        )
