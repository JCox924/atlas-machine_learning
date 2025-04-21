#!/usr/bin/env python3
"""
4-qa.py

Multi-document question-answering loop using semantic search
and the pretrained QA model.
"""

# Import necessary functions
question_answer_single = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    Interactive QA loop over multiple reference documents.

    Args:
        corpus_path (str): path to directory of reference documents.

    Behavior:
        - Prompts user with 'Q: '.
        - If user enters exit/quit/goodbye/bye (case-insensitive), prints 'A: Goodbye' and exits.
        - Otherwise, performs semantic search to pick the most relevant document,
          then applies the QA model to find an answer span.
        - Prints 'A: <answer>' on success or
          'A: Sorry, I do not understand your question.' if no answer.
    """
    while True:
        try:
            question = input("Q: ")
        except EOFError:
            break
        cmd = question.strip().lower()
        if cmd in ('exit', 'quit', 'goodbye', 'bye'):
            print("A: Goodbye")
            break

        # Retrieve the most relevant document
        ref_doc = semantic_search(corpus_path, question)
        # Get answer from single-document QA
        ans = question_answer_single(question, ref_doc)

        if ans:
            print(f"A: {ans}")
        else:
            print("A: Sorry, I do not understand your question.")
