# Natural Language Processing Metrics

This project implements the **BLEU** function that computes the cumulative n-gram BLEU score for a candidate sentence against a set of reference translations. The cumulative score is computed as the product of the brevity penalty and the geometric mean of n-gram precisions for orders 1 to n.

## Requirements

- **Python Version:** Python 3.9 (Ubuntu 20.04 LTS)
- **NumPy Version:** 1.25.2 (installed as required)
- **Style:** Code follows [pycodestyle](https://pycodestyle.pycqa.org/) guidelines.
- **Documentation:** All modules, functions, and classes include documentation.
- **Executable Files:** All files are executable and start with the shebang `#!/usr/bin/env python3`.
- **Restrictions:** The implementation does not use the `nltk` module.

## File Structure

- `0-uni_bleu.py`
- `1-ngram_bleu.py`
- `2-cumulative_bleu.py`: Contains the implementation of the `cumulative_bleu` function.

## Usage

Make sure the file `2-cumulative_bleu.py` is executable. You can run the example provided in the file with:

```bash
./2-cumulative_bleu.py
