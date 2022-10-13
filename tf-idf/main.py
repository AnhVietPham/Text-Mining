"""
https://www.kaggle.com/code/yassinehamdaoui1/creating-tf-idf-model-from-scratch/notebook
"""

import pandas as pd
from IPython.display import display
import sklearn as sk
import math

first_sentence = "Data Science is the sexiest job of the 21st century"
second_sentence = "machine learning is the key for data science"

if __name__ == "__main__":
    first_sentence = first_sentence.split(" ")
    second_sentence = second_sentence.split(" ")
    print(first_sentence)
    print(second_sentence)
    total = set(first_sentence).union(set(second_sentence))
    print(total)

    wordDictA = dict.fromkeys(total, 0)
    wordDictB = dict.fromkeys(total, 0)

    for word in first_sentence:
        wordDictA[word] += 1

    for word in second_sentence:
        wordDictB[word] += 1

    print("=" * 50)
    print(wordDictA)
    print("=" * 50)
    print(wordDictB)

    df = pd.DataFrame([wordDictA, wordDictB])
    pd.set_option('display.max_columns', None)
    display(df)
    print(1 / 16)
