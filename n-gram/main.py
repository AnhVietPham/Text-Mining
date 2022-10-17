import string
import random
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('reuters')
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk import FreqDist


def remove_stopwords(x):
    y = []
    for pair in x:
        count = 0
        for word in pair:
            if word in removal_list:
                count = count or 0
            else:
                count = count or 1
        if (count == 1):
            y.append(pair)
    return (y)


if __name__ == "__main__":
    sents = reuters.sents()
    print("Sents ==> reuters.sents()")
    print(sents)
    print("=" * 50)
    print("=" * 50)
    print("Stop word ==> stopwords.words('english')")
    stop_words = set(stopwords.words('english'))
    print(stop_words)
    print("=" * 50)
    print("=" * 50)
    print("String punctuation ==> string.punctuation")
    string.punctuation = string.punctuation + '"' + '"' + '-' + '''+''' + '—'
    print(string.punctuation)
    removal_list = list(stop_words) + list(string.punctuation) + ['lt', 'rt']
    print("=" * 50)
    print("=" * 50)
    print("Removal List ==> removal_list")
    print(removal_list)
    print("=" * 50)
    print("=" * 50)
    unigram = []
    bigram = []
    trigram = []
    tokenized_text = []
    for sentence in sents:
        sentence = list(map(lambda x: x.lower(), sentence))
        for word in sentence:
            if word == '.':
                sentence.remove(word)
            else:
                unigram.append(word)
        tokenized_text.append(sentence)
        bigram.extend(list(ngrams(sentence, 2, pad_left=True, pad_right=True)))
        trigram.extend(list(ngrams(sentence, 3, pad_left=True, pad_right=True)))

    print("=" * 50)
    print("=" * 50)
    print("Tokenized Text ==> tokenized_text")
    print(len(tokenized_text))

    print("=" * 50)
    print("=" * 50)
    print("Unigram ==> unigram")
    print(len(unigram))

    print("=" * 50)
    print("=" * 50)
    print("Bigram ==> bigram")
    print(len(bigram))

    print("=" * 50)
    print("=" * 50)
    print("Trigram ==> trigram")
    print(len(trigram))

    print("=" * 50)
    print("=" * 50)
    unigram = remove_stopwords(unigram)
    bigram = remove_stopwords(bigram)
    trigram = remove_stopwords(trigram)

    print("=" * 50)
    print("=" * 50)
    print("After remove stopword")

    print("=" * 50)
    print("=" * 50)
    print("Unigram ==> unigram")
    print(len(unigram))

    print("=" * 50)
    print("=" * 50)
    print("Bigram ==> bigram")
    print(len(bigram))

    print("=" * 50)
    print("=" * 50)
    print("Trigram ==> trigram")
    print(len(trigram))



