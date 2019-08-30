import os
import pandas as pd
import torch

from core.evaluation import evaluate_randomly, evaluate_and_show_attention
from core.lang import Lang
from core.string_utils import normalize_string
from models.attention_decoder_rnn import AttentionDecoderRNN
from models.encoder_rnn import EncoderRNN
from core.training import train_iters


def run_model():
    dataset_folder = 'dataset/number_word_std'

    dev_file_path = os.path.join(dataset_folder, 'number_word_std.dev.json')
    test_file_path = os.path.join(dataset_folder, 'number_word_std.test.json')

    math_train = pd.read_json(dev_file_path)
    math_test = pd.read_json(test_file_path)

    tokenizer = BaseTokenizer()
    tokenizer.setup(math_train, math_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 256

    encoder1 = EncoderRNN(tokenizer.input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttentionDecoderRNN(hidden_size, tokenizer.output_lang.n_words, dropout_p=0.2, max_length=tokenizer.max_length).to(device)

    train_iters(device, encoder1, attn_decoder1, tokenizer.input_lang, tokenizer.output_lang, 1000,
                tokenizer.train_pairs, tokenizer.max_length, print_every=200)

    evaluate_randomly(device, encoder1, attn_decoder1, tokenizer.input_lang, tokenizer.output_lang, tokenizer.test_pairs, tokenizer.max_length)
    evaluate_and_show_attention(device, encoder1, attn_decoder1, tokenizer.input_lang, tokenizer.output_lang, tokenizer.max_length,
                                "the sum of the digits of a 2-digit number is 7. The tens digit is one less than 3 times the units digit. Find the number.")


class BaseTokenizer:
    def __init__(self):
        self.max_length = None

        self.train_pairs = None
        self.test_pairs = None

        self.num_words = 0

        self.input_lang = Lang('text')
        self.output_lang = Lang('equations')

    def setup(self, train, test):
        self._clean(train, test)

        self.train_pairs = [list(x) for x in train[['text', 'equations']].to_records(index=False)]
        self.test_pairs = [list(x) for x in test[['text', 'equations']].to_records(index=False)]

        for pairs in [self.train_pairs, self.test_pairs]:
            for pair in pairs:
                self.input_lang.add_sentence(pair[0])
                self.output_lang.add_sentence(pair[1])

    def _clean(self, train, test):
        train.equations = train.equations.apply(lambda x: '; '.join(x))
        test.equations = test.equations.apply(lambda x: '; '.join(x))

        self.max_length = max(train.text.apply(lambda x: len(x.split(' '))).max(),
                              test.text.apply(lambda x: len(x.split(' '))).max())

        train.equations = train.equations.apply(lambda x: normalize_string(x))
        test.equations = test.equations.apply(lambda x: normalize_string(x))


if __name__ == '__main__':
    run_model()