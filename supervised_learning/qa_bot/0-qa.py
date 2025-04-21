#!/usr/bin/env python3
"""
0-qa.py

Question-answering using BERT QA model from TensorFlow Hub
and BertTokenizer from transformers.
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

# Load once for efficiency
_TOKENIZER = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)
_MODEL = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')


def question_answer(question, reference):
    """
    Finds a span in `reference` that answers `question`.

    Args:
        question (str): the question to answer.
        reference (str): the text to search.

    Returns:
        str or None: the answer span, or None if no valid answer.
    """
    # Tokenize question and reference
    inputs = _TOKENIZER.encode_plus(
        question,
        reference,
        return_tensors='tf'
    )
    # Adapting keys for TF Hub model
    inputs = {
        'input_word_ids': inputs['input_ids'],
        'input_mask': inputs['attention_mask'],
        'input_type_ids': inputs['token_type_ids']
    }
    # Run model
    outputs = _MODEL(inputs)
    start_logits = outputs['start_logits'][0]
    end_logits = outputs['end_logits'][0]

    # Compute best start and end
    start_index = tf.argmax(start_logits).numpy().item()
    end_index = tf.argmax(end_logits).numpy().item()

    # If start > end, no valid answer
    if start_index > end_index:
        return None

    # Convert tokens to string
    token_ids = inputs['input_word_ids'][0].numpy()
    tokens = _TOKENIZER.convert_ids_to_tokens(token_ids)
    answer = _TOKENIZER.convert_tokens_to_string(
        tokens[start_index:end_index + 1]
    )

    return answer if answer.strip() else None
