from nltk.translate.bleu_score import sentence_bleu

if __name__ == '__main__':
    reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
    MT_translation = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']
    score = sentence_bleu(reference, MT_translation)
    print(f'Score: {score}')
