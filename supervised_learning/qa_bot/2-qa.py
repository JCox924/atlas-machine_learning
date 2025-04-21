#!/usr/bin/env python3
"""
2-qa.py

Interactive question-answer loop that uses a single reference text
and the pretrained QA function to answer questions.
"""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Runs a loop prompting the user (Q:) and answering using the QA model.

    Args:
        reference (str): the reference text to search for answers.

    Behavior:
        - Prompts with 'Q: '.
        - If input is exit/quit/goodbye/bye (case-insensitive), prints 'A: Goodbye' and exits.
        - Otherwise, prints 'A: <answer>' if found, or 'A: Sorry, I do not understand your question.' if not.
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
        # Attempt to get an answer
        ans = question_answer(question, reference)
        if ans:
            print(f"A: {ans}")
        else:
            print("A: Sorry, I do not understand your question.")
