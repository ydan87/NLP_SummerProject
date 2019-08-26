import os
import pandas as pd
import torch
from core.lang import Lang
from core.string_utils import normalize_string
from models.attention_decoder_rnn import AttentionDecoderRNN
from models.encoder_rnn import EncoderRNN
from baseline_model import train_iters, evaluate_randomly, evaluate_and_show_attention


dataset_folder = 'dataset/number_word_std'

dev_file_path = os.path.join(dataset_folder, 'number_word_std.dev.json')
test_file_path = os.path.join(dataset_folder, 'number_word_std.test.json')

math_train = pd.read_json(dev_file_path)
math_train.equations = math_train.equations.apply(lambda x: '; '.join(x))

math_test = pd.read_json(test_file_path)
math_test.equations = math_test.equations.apply(lambda x: '; '.join(x))

MAX_LENGTH = max(math_train.text.apply(lambda x: len(x.split(' '))).max(),
                 math_test.text.apply(lambda x: len(x.split(' '))).max())

math_train.equations = math_train.equations.apply(lambda x: normalize_string(x))
math_test.equations = math_test.equations.apply(lambda x: normalize_string(x))

input_lang = Lang('text')
output_lang = Lang('equations')
train_pairs = [list(x) for x in math_train[['text', 'equations']].to_records(index=False)]
test_pairs = [list(x) for x in math_test[['text', 'equations']].to_records(index=False)]

for pairs in [train_pairs, test_pairs]:
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttentionDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.2, max_length=MAX_LENGTH).to(device)

train_iters(device, encoder1, attn_decoder1, input_lang, output_lang, 75000, train_pairs, MAX_LENGTH, print_every=5000)

evaluate_randomly(device, encoder1, attn_decoder1, input_lang, output_lang, test_pairs, MAX_LENGTH)
evaluate_and_show_attention(device, encoder1, attn_decoder1, input_lang, output_lang, MAX_LENGTH, "the sum of the digits of a 2-digit number is 7. The tens digit is one less than 3 times the units digit. Find the number.")
