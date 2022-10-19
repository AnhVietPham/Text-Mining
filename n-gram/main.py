import re
from urllib.request import urlopen

url_template = 'https://www.gutenberg.org/cache/epub/%s/pg%s.txt'

books = {'Pride and Prejudice': '1342',
         'Huckleberry Fin': '76',
         'Sherlock Holmes': '1661'}

book = books['Pride and Prejudice']


def get2GramSentence(word, n_gram, n=50):
    for i in range(n):
        print(word, end=" ")
        # for element in n_gram:
        #     if element[0][0] == word:
        # next(element[0][1], None)
        for index, element in enumerate(n_gram):
            if element[0][0] == word:
                word = element[0][1]
                break
        # word = next(element[0][1] for element in n_gram if element[0][0] == word)
        if not word:
            break


if __name__ == "__main__":
    url = "https://www.gutenberg.org/files/1342/1342-0.txt"
    with urlopen(url) as file:
        content = file.read().decode()
    print(f'Length: {len(content)}, content: {content[:50]}')

    print("=" * 50)

    words = re.split('[^A-za-z]+', content.lower())

    print(words[:50])

    print("=" * 50)
    gram1 = set(words)
    print(f'Length Gram 1: {len(gram1)}')
    print(gram1)
    gram1_iter = iter(gram1)
    print([gram1_iter.__next__() for i in range(20)])

    print("=" * 50)
    for i in range(len(words) - 10, len(words) - 1):
        print(f'{words[i]} {words[i + 1]}')

    print("=" * 50)
    word_pairs = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    print(word_pairs[:50])

    print("=" * 50)
    gram2 = set(word_pairs)
    print(f"Length word pairs: {len(word_pairs)}")
    print(f"Length gram2: {len(gram2)}")
    gram2_iter = iter(gram2)
    print([gram2_iter.__next__() for i in range(20)])

    print("=" * 50)
    print("=" * 20 + "Frequency" + "=" * 20)

    # Populate 1-gram dictionary
    gram_frequency_1 = dict()

    for word in words:
        if word in gram_frequency_1.keys():
            gram_frequency_1[word] += 1
        else:
            gram_frequency_1[word] = 1

    gram_frequency_1 = sorted(gram_frequency_1.items(), key=lambda item: -item[1])
    print(gram_frequency_1[:20])

    # Populate 2-gram dictionary
    print("=" * 50)
    gram_frequency_2 = dict()

    for i in range(len(words) - 1):
        key = (words[i], words[i + 1])
        if key in gram_frequency_2.keys():
            gram_frequency_2[key] += 1
        else:
            gram_frequency_2[key] = 1

    gram_frequency_2 = sorted(gram_frequency_2.items(), key=lambda item: -item[1])
    print(gram_frequency_2[:20])

    # Prediction
    print("=" * 50)
    start_word = words[int(len(words) / 4)]
    print(start_word)
    word = start_word
    print(f'Start word: {start_word}')

    print('2 gram sentence:')
    get2GramSentence(start_word, gram_frequency_2, 15)

    print()
    print("=" * 50)
    for i in ['and', 'he', 'she', 'when', 'join', 'never', 'i', 'how']:
        print()
        print(f'Start word: {i}')
        print("2 gram sentence:")
        get2GramSentence(i, gram_frequency_2, 15)
