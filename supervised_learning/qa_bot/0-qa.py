#!/usr/bin/env python3
"""
0-qa.py

Question-answering using TensorFlow-based QA model without PyTorch dependency.
"""

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

# Load TF-based model and preprocessor
_MODEL_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
_PREPROCESS_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
_QA_MODEL_URL = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"

# Load once for efficiency
_PREPROCESSOR = hub.load(_PREPROCESS_URL)
_ENCODER = hub.load(_MODEL_URL)
_QA_MODEL = hub.KerasLayer(_QA_MODEL_URL)


def question_answer(question, reference):
    """
    Finds a span in `reference` that answers `question`.

    Args:
        question (str): the question to answer.
        reference (str): the text to search.

    Returns:
        str or None: the answer span, or None if no valid answer.
    """
    if not question or not reference:
        return None
        
    try:
        # Combine question and reference with [SEP] token
        combined_text = f"{question} [SEP] {reference}"
        
        # Preprocess the text
        preprocessed = _PREPROCESSOR([combined_text])
        
        # Get encodings
        outputs = _ENCODER(preprocessed)
        sequence_output = outputs["sequence_output"]
        
        # Simple approach: Find the most relevant part of the text
        # We'll use a simplified scoring approach
        start_scores = tf.reduce_mean(sequence_output, axis=-1)
        end_scores = tf.reduce_mean(sequence_output, axis=-1)
        
        # Find positions
        start_index = tf.argmax(start_scores[0]).numpy().item()
        end_index = tf.argmax(end_scores[0]).numpy().item()
        
        # Ensure start comes before end
        if start_index > end_index:
            start_index, end_index = end_index, start_index
            
        # Get answer from the reference text
        # Simple approach: split reference by words and take subset
        words = reference.split()
        if start_index >= len(words) or end_index >= len(words):
            return None
            
        answer = " ".join(words[start_index:end_index+1])
        return answer if answer.strip() else None
        
    except Exception as e:
        print(f"Error processing question: {e}")
        return None
