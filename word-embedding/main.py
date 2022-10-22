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
from keras.utils import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.preprocessing import OneHotEncoder
from torchsummaryX import summary

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
            word_2_idx[word] = i
            i += 1


def one_hot_map(doc):
    x = []
    for word in doc.split():
        x.append(word_2_idx[word])
    return x


class WEMB(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(WEMB, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=1)

        self.l1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.l2 = nn.Linear(self.hidden_size, self.input_size, bias=False)

    def forward(self, x):
        out_bn = self.l1(x)
        out = self.l2(out_bn)
        out = self.softmax(out)
        return out, out_bn


if __name__ == '__main__':
    print(idx_2_word)
    print(word_2_idx)

    """
    Words to numbers
    """
    vocab_size = 25
    encoded_docs = [one_hot_map(d) for d in docs]
    print("=" * 50)
    print(encoded_docs)

    """
    Padding
    """
    max_len = 10
    padded_docs = pad_sequences(encoded_docs, maxlen=max_len, padding='post')
    print("=" * 50)
    print(padded_docs)

    """
    Creating Dataset tuples for training
    """

    training_data = np.empty((0, 2))
    window = 2
    print("=" * 50)
    print(training_data)
    for sentence in padded_docs:
        sent_len = len(sentence)
        for i, word in enumerate(sentence):
            w_context = []
            if sentence[i] != 0:
                w_target = sentence[i]
                for j in range(i - window, i + window + 1):
                    if j != i and j <= sent_len - 1 and j >= 0 and sentence[j] != 0:
                        w_context = sentence[j]
                        training_data = np.append(training_data, [[w_target, w_context]], axis=0)
    print(len(training_data))
    print(len(training_data.shape))
    print(training_data)

    print("=" * 50)
    print(np.array(range(30)))
    print(np.array(range(30)).reshape(-1, 1))
    enc = OneHotEncoder()
    enc.fit(np.array(range(30)).reshape(-1, 1))
    one_hot_label_x = enc.transform(training_data[:, 0].reshape(-1, 1)).toarray()
    print("One Hot Label X")
    print(one_hot_label_x.shape)
    print(one_hot_label_x)
    enc = OneHotEncoder()
    enc.fit(np.array(range(30)).reshape(-1, 1))
    one_hot_label_y = enc.transform(training_data[:, 1].reshape(-1, 1)).toarray()
    print("One Hot Label Y")
    print(one_hot_label_y.shape)
    print(one_hot_label_y)

    """
    From Numpy to Torch
    """
    onehot_label_x = torch.from_numpy(one_hot_label_x)
    onehot_label_y = torch.from_numpy(one_hot_label_y)
    print(onehot_label_x.shape, onehot_label_y.shape)

    input_size = 30
    hidden_size = 2
    lr = 0.01
    num_epochs = 10000

    model = WEMB(input_size, hidden_size)
    model.train(True)
    print("=" * 50)
    print(model)
    print()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, dampening=0, weight_decay=0,
                                nesterov=False)
    summary(model, torch.ones(1, 30))

    loss_val = []
    for epoch in range(num_epochs):
        for i in range(one_hot_label_y.shape[0]):
            inputs = onehot_label_x[i].float()
            labels = onehot_label_y[i].float()
            inputs = inputs.unsqueeze(0)
            labels = labels.unsqueeze(0)

            output, wemb = model(inputs)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_val.append(loss.item)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch[{epoch+1} / {num_epochs}], Loss: {loss.item():.4f}')

