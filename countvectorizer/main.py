if __name__ == "__main__":
    data = ["Doubt thou the stars are fire;",
            "Doubt that the sun doth move;",
            "Doubt truth to be a liar;",
            "But never doubt I love."]

    unique_words = set()

    for each_sentence in data:
        print(each_sentence)
        for each_word in each_sentence.split(' '):
            print(each_word)
