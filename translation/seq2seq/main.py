import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os
import re
import random

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'S0S', 1: 'EOS'}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Normalize every sentence
def normalize_sentence(df, lang):
    sentence = df[lang].str.lower()
    sentence = sentence.str.replace('[^A-Za-z\s]+', '')
    sentence = sentence.str.normalize('NFD')
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
    return sentence


def read_sentence(df, lang1, lang2):
    sentence1 = normalize_sentence(df, lang1)
    sentence2 = normalize_sentence(df, lang2)
    return sentence1, sentence2


def read_file(loc, lang1, lang2, des):
    df = pd.read_csv(loc, delimiter='\t', header=None, names=[lang1, lang2, des])
    return df


def process_data(lang1, lang2):
    df = read_file('/Users/sendo_mac/Documents/avp/Text-Mining/translation/data/vie-eng/%s-%s.txt' % (lang1, lang2),
                   lang1, lang2, "des")
    print("Read %s sentence pairs" % len(df))
    sentence1, sentence2 = read_sentence(df, lang1, lang2)

    source = Lang()
    target = Lang()
    pairs = []
    for i in range(len(df)):
        if len(sentence1[i].split(' ')) < MAX_LENGTH and len(sentence2[i].split(' ')) < MAX_LENGTH:
            full = [sentence1[i], sentence2[i]]
            source.addSentence(sentence1[i])
            target.addSentence(sentence2[i])
            pairs.append(full)

    return source, target, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, self.embbed_dim)

        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)

    def forward(self, src):
        embedded = self.embedding(src).view(1, 1, -1)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, MAX_LENGTH=MAX_LENGTH):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        input_length = source.size(0)
        batch_size = target.shape[1]
        target_length = target.shape[0]
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(target_length, batch_size, vocab_size)

        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(source[i])

        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([SOS_token])

        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teach_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (target[t] if teach_force else topi)
            if (teach_force == False and input.item() == EOS_token):
                break
        return outputs


def clacModel(model, input_tensor, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    output = model(input_tensor, target_tensor)
    num_iter = output.size(0)

    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])
    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter
    return epoch_loss


def trainModel(model, source, target, pairs, num_iteration=20000):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0
    training_pairs = [tensorsFromPair(source, target, random.choice(pairs)) for i in range(num_iteration)]

    for iter in range(1, num_iteration + 1):
        training_pairs = training_pairs[iter - 1]
        input_tensor = training_pairs[0]
        target_tensor = training_pairs[1]

        loss = clacModel(model, input_tensor, target_tensor, optimizer, criterion)
        total_loss_iterations += loss

        if iter % 5000 == 0:
            avarge_loss = total_loss_iterations / 5000
            total_loss_iterations = 0
            print('%d %.4f' % (iter, avarge_loss))
    torch.save(model.state_dict(), 'avptraning.pt')
    return model


def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentences[0])
        output_tensor = tensorFromSentence(output_lang, sentences[1])

        decoded_words = []
        output = model(input_tensor, output_tensor)

        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)

            if topi[0].item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi[0].item()])
    return decoded_words


def evaluateRandomly(model, source, target, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('source {}'.format(pair[0]))
        print('target {}'.format(pair[1]))
        output_words = evaluate(model, source, target, pair)
        output_sentence = ' '.join(output_words)
        print('Predicted {}'.format(output_sentence))


if __name__ == '__main__':
    lang1 = 'eng'
    lang2 = 'vie'
    source, target, pairs = process_data(lang1, lang2)
    randomize = random.choice(pairs)
    print('Random sentence {}'.format(randomize))

    input_size = source.n_words
    out_size = target.n_words
    print('Input : {} Ouput : {}'.format(input_size, out_size))

    embed_size = 256
    hidden_size = 512
    num_layers = 1
    num_iteration = 100000

    encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
    decoder = Decoder(out_size, hidden_size, embed_size, num_layers)

    model = Seq2Seq(encoder, decoder)
    print(encoder)
    print("=" * 50)
    print(decoder)

    model = trainModel(model, source, target, pairs, num_iteration)
    evaluateRandomly(model, source, target, pairs)
