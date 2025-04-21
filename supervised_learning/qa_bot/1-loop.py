#!/usr/bin/env python3
"""
1-loop.py

Interactive loop that prompts the user with 'Q:' and
echos 'A:' as the response. Exits on exit/quit/goodbye/bye.
"""

def main():
    """Run the question-answering prompt loop."""
    while True:
        try:
            question = input("Q: ")
        except EOFError:
            break
        cmd = question.strip().lower()
        if cmd in ('exit', 'quit', 'goodbye', 'bye'):
            print("A: Goodbye")
            break
        print("A: ")


if __name__ == '__main__':
    main()
