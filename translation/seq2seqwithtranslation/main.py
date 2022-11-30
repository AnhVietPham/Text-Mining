import random

from translation.Encoder import Encoder
from translation.seq2seq import Seq2Seq
from translation.seq2seqwithtranslation.attndecoder import AttnDecoderRNN
from translation.uitls import process_data, evaluateRandomly

if __name__ == "__main__":
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
    decoder = AttnDecoderRNN(out_size, hidden_size, embed_size, num_layers)

    model = Seq2Seq(encoder, decoder)
    print(encoder)
    print("=" * 50)
    print(decoder)

    # model = trainModel(model, source, target, pairs, num_iteration)
    # model.load_state_dict(
    #     torch.load("/Users/sendo_mac/Documents/avp/Text-Mining/translation/seq2seq/model/avptraning.pt"))
    evaluateRandomly(model, source, target, pairs)
