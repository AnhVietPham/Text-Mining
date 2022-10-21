import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import random
from keras.preprocessing.text import one_hot
from sklearn.preprocessing import OneHotEncoder
from torchsummaryX import summary

if __name__ == '__main__':
    docs = ["cat and mice are buddies",
            'mice lives in hole',
            'cat lives in house',
            'cat chases mice',
            'cat catches mice',
            'cat eats mice',
            'mice runs into hole',
            'cat says bad words',
            'cat and mice are pals',
            'cat and mice are chums',
            'mice stores food in hole',
            'cat stores food in house',
            'mice sleeps in hole',
            'cat sleeps in house']

    idx_2_word = {}
    word_2_idx = {}
    temp = []
    i = 1
    for doc in docs:
        for word in doc.split():
            if word not in temp:
                temp.append(word)
                idx_2_word[i] = word
                word_2_idx = i
                i += 1
    print(idx_2_word)
    print(word_2_idx)
