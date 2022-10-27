import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

text = (
    'Hello, how are you? I am Romeo.\n'
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'
    'Nice meet you too. How are you today?\n'
    'Great. My baseball team won the competition.\n'
    'Oh Congratulations, Juliet\n'
    'Thanks you Romeo'
)

if __name__ == "__main__":
    sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
    word_list = list(set(" ".join(sentences).split()))
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    print(sentences)
    print(word_list)
    print(word_dict)

    print("=" * 50)
    for i, w in enumerate(word_list):
        word_dict[w] = i + 4
    number_dict = {i: w for i, w in enumerate(word_dict)}
    vocab_size = len(word_dict)
    print(number_dict)
    print(vocab_size)

    print("=" * 50)
    token_list = list()
    for sentence in sentences:
        arr = [word_dict[s] for s in sentence.split()]
        token_list.append(arr)
    print(token_list)

    maxlen = 30  # maximum of length
    batch_size = 6
    max_pred = 5  # max tokens of prediction
    n_layers = 6  # number of Encoder of Encoder Layer
    n_heads = 12  # number of heads in Multi-Head Attention
    d_model = 768  # Embedding Size
    d_ff = 768 * 4  # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2
