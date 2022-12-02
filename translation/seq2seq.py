import random

import torch
from torch import nn
from translation.uitls import MAX_LENGTH, SOS_token, EOS_token


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, is_attention=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_attention = is_attention

    def forward(self, source, target, teacher_forcing_ratio=0.5, ):
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
            if self.is_attention:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teach_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (target[t] if teach_force else topi)
            if teach_force == False and input.item() == EOS_token:
                break
        return outputs
