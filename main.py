import os
import pandas as pd
import torch

from core.evaluation import evaluate, evaluate_and_show_attention
from core.string_utils import normalize_string, text2int, remove_punctuation, generalize
from core.tensor_utils import DEVICE
from core.tokenizers import Tokenizer
from models.attention_decoder_rnn import AttentionDecoderRNN
from models.encoder_rnn import EncoderRNN
from core.training import train


def run_model(clean_func, approach):
    """ Using the pre-processing approach, fetches the data, processes it, and trains the model """
    torch.manual_seed(42)

    dataset_folder = 'dataset/number_word_std'
    dev_file_path = os.path.join(dataset_folder, 'number_word_std.dev.json')
    test_file_path = os.path.join(dataset_folder, 'number_word_std.test.json')

    math_train = pd.read_json(dev_file_path)
    math_test = pd.read_json(test_file_path)

    clean_func(math_train)
    clean_func(math_test)

    max_length = 1 + max(math_train.text.apply(lambda x: len(x.split(' '))).max(), math_test.text.apply(lambda x: len(x.split(' '))).max())

    all_data = pd.concat([math_train, math_test])

    input_tokenizer = Tokenizer('text').build(all_data)
    output_tokenizer = Tokenizer('equations').build(all_data)

    train_pairs = [list(x) for x in math_train[['text', 'equations']].to_records(index=False)]
    test_pairs = [list(x) for x in math_test[['text', 'equations']].to_records(index=False)]

    # Model hyper-parameters
    hidden_size = 256
    dropout_p = 0.2

    # Our network components - Encoder RNN and Attention decoder RNN
    encoder1 = EncoderRNN(input_tokenizer.lang.n_words, hidden_size).to(DEVICE)
    attn_decoder1 = AttentionDecoderRNN(hidden_size, output_tokenizer.lang.n_words, dropout_p=dropout_p, max_length=max_length).to(DEVICE)

    # Run training for n_iters times (we chose 60000 after several trials)
    train(approach, encoder1, attn_decoder1, input_tokenizer, output_tokenizer, 60000, train_pairs, max_length, log_every=5000)

    # Randomly takes problems from the test set and evaluates them
    test_accuracy = \
        evaluate(encoder1, attn_decoder1, input_tokenizer, output_tokenizer, test_pairs, max_length, n=10, verbose=True)

    print('Test accuracy')
    for key, value in test_accuracy.items():
        print(f'{key}: {value}')

    # Another evaluation
    sample_question = math_test.sample(1).iloc[0]['text']
    evaluate_and_show_attention(approach, encoder1, attn_decoder1, input_tokenizer, output_tokenizer, max_length, sample_question)


def run_baseline_model():
    # baseline approach: provided as the "DataHack" challenge
    def clean(data):
        clean_equations = lambda s: normalize_string('; '.join(s))
        data.equations = data.equations.apply(clean_equations)

    run_model(clean, 'baseline')


def run_text_to_numbers_model():
    # "text to numbers" approach: remove any punctuation from the text as well as transforming
    # any number-in-word to an integer (e.g. "one apple" to "1 apple", "two bananas" to "2 bananas")
    def clean(data):
        clean_text = lambda s: text2int(remove_punctuation(s))
        data.text = data.text.apply(clean_text)

        clean_equations = lambda s: normalize_string('; '.join(s))
        data.equations = data.equations.apply(clean_equations)

    run_model(clean, 'text2int')


def run_generalized_text_model():
    # generalized approach: as text-to-numbers, but also generalizes the questions.
    # That is, if a question start with "Mike has 3 toys...", change it to "Mike has var1 toys".
    def clean(data):
        clean_text = lambda s: text2int(remove_punctuation(s))
        data.text = data.text.apply(clean_text)

        clean_equations = lambda s: normalize_string('; '.join(s))
        data.equations = data.equations.apply(clean_equations)

        data[['text', 'equations']] = data[['text', 'equations']].apply(
            lambda x: generalize(x['text'], x['equations']), axis=1, result_type='expand')

    run_model(clean, 'generalized')


if __name__ == '__main__':
    # We have tried several approaches in pre-processing that data before moving to the training phase.
    # The approaches are described across this file as well as in the report
    #    run_baseline_model()
    #    run_text_to_numbers_model()
    run_generalized_text_model()
