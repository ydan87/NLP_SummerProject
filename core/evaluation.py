import torch
import random

from core.lang import EOS_token, SOS_token
from core.tensor_utils import tensor_from_sentence
from core.visualizations import show_attention


def evaluate(device, encoder, decoder, input_tokenizer, output_tokenizer, sentence, max_length):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(device, input_tokenizer, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token[0]]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token[0]:
                decoded_words.append(f'<{EOS_token[1]}>')
                break
            else:
                decoded_words.append(output_tokenizer.untokenize(topi.item()))

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(device, encoder, decoder, input_tokenizer, output_tokenizer, test_pairs, max_len, n=10):
    for i in range(n):
        pair = random.choice(test_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(device, encoder, decoder, input_tokenizer, output_tokenizer, pair[0], max_len)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluate_and_show_attention(device, encoder, decoder, input_tokenizer, output_tokenizer, max_len, input_sentence):
    output_words, attentions = \
        evaluate(device, encoder, decoder, input_tokenizer, output_tokenizer, input_sentence, max_len)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    print(len(output_words))
    show_attention(input_sentence, output_words, attentions)
