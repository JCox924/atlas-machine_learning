# Error Analysis in Machine Learning


## Intro

Error Analysis is the process in which developers analysis the accuracy of a model
to find the distribution of errors across different subsets of data.

### Files

| #                   | Python                                         |Fortran|
|---------------------|------------------------------------------------|-------|
| 0- | [create_confusion.py](./0-create_confusion.py) ||
| 1- | [sensitivity.py](./1-sensitivity.py)           ||
| 2- | [precision.py](./2-precisiion.py)              ||
| 3- | [specificity.py](./3-specificity.py)           ||
| 4- | [f1_score.py](./4-f1_score.py)                 ||

## Confusion matrix
#### Let's take the coin toss example, think of why there is a 50/50 chance of guessing heads or tails.
There are only two outcomes and only two predictions you can make
### We can make a confusion matrix of the events:

| Guess| Actual: Heads | Tails |
|------|---------------|-------|
| Heads | True          | False |
| Tails | False         | True  |

