"""
https://www.kaggle.com/code/yassinehamdaoui1/creating-tf-idf-model-from-scratch/notebook
"""

import pandas as pd
from IPython.display import display
import sklearn as sk
import math

first_sentence = "Data Science is the sexiest job of the 21st century"
second_sentence = "machine learning is the key for data science"


def computeTF(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count / float(corpusCount)
    return (tfDict)


def computeIDF(docList):
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / (float(val) + 1))
    return (idfDict)


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return (tfidf)


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

    tfFirst = computeTF(wordDictA, first_sentence)
    tfSecond = computeTF(wordDictB, second_sentence)

    idfs = computeIDF([wordDictA, wordDictB])

    idfFirst = computeTFIDF(tfFirst, idfs)
    idfSecond = computeTFIDF(tfSecond, idfs)
