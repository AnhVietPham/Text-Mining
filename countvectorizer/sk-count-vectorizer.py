from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from IPython.display import display

if __name__ == '__main__':
    document = ["One Geek helps Two Geeks",
                "Two Geeks help Four Geeks",
                "Each Geek helps many other Geeks at GeeksforGeeks"]
    vectorizer = CountVectorizer()
    vectorizer.fit(document)

    print(f"Vocabulary: {vectorizer.vocabulary_}")
    vector = vectorizer.transform(document)

    print("Encoded Document: ")
    print(vector.toarray())

    print("=" * 50)
    print("=" * 50)
    df = pd.read_csv('https://raw.githubusercontent.com/flyandlure/datasets/master/gonutrition.csv')
    display(df)

    print("=" * 50)
    print("=" * 50)
    text = df['product_description']
    display(text.head())

    print("=" * 50)
    print("=" * 50)
    unigram_model = CountVectorizer(ngram_range=(1, 1))
    unigram_matrix = unigram_model.fit_transform(text).toarray()
    df_uni_gram_output = pd.DataFrame(data=unigram_matrix, columns=unigram_model.get_feature_names_out())
    display(df_uni_gram_output.T.tail(5))
    print(df_uni_gram_output.shape)

    print("=" * 50)
    print("=" * 50)
    two_gram_model = CountVectorizer(ngram_range=(2, 2))
    two_gram_matrix = two_gram_model.fit_transform(text).toarray()
    df_two_gram_output = pd.DataFrame(data=two_gram_matrix, columns=two_gram_model.get_feature_names_out())
    display(df_two_gram_output.T.tail(5))
    print(df_two_gram_output.shape)

