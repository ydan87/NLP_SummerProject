import os
import pandas as pd
import torch

from core.evaluation import evaluate_randomly, evaluate_and_show_attention
from core.string_utils import normalize_string
from core.tokenizers import Tokenizer
from models.attention_decoder_rnn import AttentionDecoderRNN
from models.encoder_rnn import EncoderRNN
from core.training import train_iters


def run_model():
    dataset_folder = 'dataset/number_word_std'

    dev_file_path = os.path.join(dataset_folder, 'number_word_std.dev.json')
    test_file_path = os.path.join(dataset_folder, 'number_word_std.test.json')

    math_train = pd.read_json(dev_file_path)
    math_test = pd.read_json(test_file_path)

    math_train.equations = math_train.equations.apply(lambda x: '; '.join(x))
    math_test.equations = math_test.equations.apply(lambda x: '; '.join(x))

    max_length = max(math_train.text.apply(lambda x: len(x.split(' '))).max(),
                     math_test.text.apply(lambda x: len(x.split(' '))).max())

    math_train.equations = math_train.equations.apply(lambda x: normalize_string(x))
    math_test.equations = math_test.equations.apply(lambda x: normalize_string(x))

    all_data = pd.concat([math_train, math_test])

    input_tokenizer = Tokenizer('text').build(all_data)
    output_tokenizer = Tokenizer('equations').build(all_data)

    train_pairs = [list(x) for x in math_train[['text', 'equations']].to_records(index=False)]
    test_pairs = [list(x) for x in math_test[['text', 'equations']].to_records(index=False)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model hyper-parameters
    hidden_size = 256
    dropout_p = 0.2

    encoder1 = EncoderRNN(input_tokenizer.lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttentionDecoderRNN(hidden_size, output_tokenizer.lang.n_words,
                                        dropout_p=dropout_p, max_length=max_length).to(device)

    # Todo: return to real number of iterations and prints
    train_iters(device, encoder1, attn_decoder1, input_tokenizer, output_tokenizer, 1000,
                train_pairs, max_length, print_every=200)

    evaluate_randomly(device, encoder1, attn_decoder1,
                      input_tokenizer, output_tokenizer, test_pairs, max_length)

    sample_question = "the sum of the digits of a 2-digit number is 7. The tens digit is one less than 3 times the " \
                      "units digit. Find the number. "
    evaluate_and_show_attention(device, encoder1, attn_decoder1,
                                input_tokenizer, output_tokenizer, max_length,
                                sample_question)


if __name__ == '__main__':
    run_model()