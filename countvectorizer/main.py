from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


def create_vocab(data):
    unique_words = set()

    for each_sentence in data:
        for each_word in each_sentence.split(' '):
            if len(each_word) > 2:
                unique_words.add(each_word)

    vocab = {}
    for index, word in enumerate(sorted(list(unique_words))):
        vocab[word] = index

    return vocab


def counter_vectorizer(data):
    vocab = create_vocab(data)
    print(f'Vocab: {vocab}')

    print("===========================================")

    row, col, val = [], [], []

    for idx, sentence in enumerate(data):
        print(f'idx: {idx}, sentence: {sentence}')
        count_word = dict(Counter(sentence.split(' ')))
        print(f'Count Word: {count_word}')

        print(f'{count_word.items()}')

        for word, count in count_word.items():
            if len(word) > 2:
                col_index = vocab.get(word)
                print(col_index)
                if col_index >= 0:
                    row.append(idx)
                    col.append(col_index)
                    val.append(count)

    print("==================================")
    print(f'row: {row}')
    print(f'col: {col}')
    print(f'val: {val}')

    print((csr_matrix((val, (row, col)), shape=(len(data), len(vocab)))))
    print((csr_matrix((val, (row, col)), shape=(len(data), len(vocab)))).toarray())

    return (csr_matrix((val, (row, col)), shape=(len(data), len(vocab)))).toarray()


if __name__ == "__main__":
    data = ["Doubt Doubt thou the stars are fire ;",
            "Doubt that the sun doth move ;",
            "Doubt truth to be a liar ;",
            "But never doubt I love ."]

    vocab = create_vocab(data)
    my_counter_vectorizer_output = counter_vectorizer(data)

    vectorizer = CountVectorizer()

    a = vectorizer.fit(data)

    sklearn_counter_vectorizer_output = vectorizer.fit_transform(data).toarray()

    print("======================")
    print("This is Counter Vectorizer from sklearn: ")
    print(sklearn_counter_vectorizer_output)
    print(f"Shape: {sklearn_counter_vectorizer_output.shape}")
    print("\n")
    print("======================")
    print("This is My Counter Vectorizer: ")
    print(my_counter_vectorizer_output)
    print(f"Shape: {my_counter_vectorizer_output.shape}")
    print("\n")
    print("======================")
    print(f"Check: {sklearn_counter_vectorizer_output == my_counter_vectorizer_output}")
