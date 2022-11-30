import random

import pandas as pd
import torch
from torch import optim, nn

from translation.Lang import Lang

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20


def normalize_sentence(df, lang):
    sentence = df[lang].str.lower()
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
