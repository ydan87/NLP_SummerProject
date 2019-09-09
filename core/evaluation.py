from collections import Counter

import torch
import random
import numpy as np

from core.lang import EOS_token, SOS_token
from core.tensor_utils import tensor_from_sentence, DEVICE
from core.visualizations import show_attention

EQUATION_LEVEL_ACC, QUESTION_LEVEL_ACC, EQUATION_STRUCTURE_LEVEL_ACC, QUESTION_STRUCTURE_LEVEL_ACC = range(4)


def accuracy(y_true, y_pred):
    metrics = Counter()

    num_questions = len(y_true)
    num_equations = 0

    for true_equations, pred_equations in zip(y_true, y_pred):
        prepare_equ = lambda x: [equ.strip() for equ in x.split(';')]
        true_equations = prepare_equ(true_equations)
        num_equations += len(true_equations)

        pred_equations = prepare_equ(pred_equations)

        is_question_level_accurate = True
        for t_equ, p_equ in zip(true_equations, pred_equations):
            t_words = t_equ.split(' ')
            p_words = p_equ.split(' ')

            is_equation_level_accurate = len(p_words) == len(p_words)
            is_equation_structure_level_accurate = is_equation_level_accurate

            for t_word, p_word in zip(t_words, p_words):
                if not is_equation_structure_level_accurate:
                    break

                if t_word == p_word:
                    continue

                is_equation_level_accurate = False

                if t_word.startswith('var') and p_word.startswith('var'):
                    is_equation_structure_level_accurate = False
                elif t_word.isdigit():
                    try:
                        assert eval(t_word) == eval(p_word)
                    except:
                        is_equation_structure_level_accurate = False
                elif t_word.isalnum() and p_word.isalnum():
                    is_equation_structure_level_accurate = False

            if is_equation_structure_level_accurate:
                metrics[EQUATION_STRUCTURE_LEVEL_ACC] += 1

            if is_equation_level_accurate:
                metrics[EQUATION_LEVEL_ACC] += 1
            else:
                is_question_level_accurate = False

        if is_question_level_accurate:
            if len(true_equations) == len(pred_equations):
                metrics[QUESTION_LEVEL_ACC] += 1

    acc = dict()
    acc['question_level'] = metrics[QUESTION_LEVEL_ACC] / num_questions
    acc['equation_level'] = metrics[EQUATION_LEVEL_ACC] / num_equations
    acc['equation_structure_level'] = metrics[EQUATION_STRUCTURE_LEVEL_ACC] / num_equations

    return acc


def evaluate_sample(encoder, decoder, input_tokenizer, output_tokenizer, sentence, max_length):
    """ Evaluation """
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_tokenizer, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token[0]]], device=DEVICE)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        di = 0
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token[0]:
                break
            else:
                decoded_words.append(output_tokenizer.untokenize(topi.item()))

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate(encoder, decoder, input_tokenizer, output_tokenizer, test_pairs, max_len, n=None, verbose=False):
    """ Randomized n questions from the test set and evaluates each of them """
    y_true = []
    y_pred = []

    if n is not None:
        data = [random.choice(test_pairs) for i in range(n)]
    else:
        data = test_pairs

    for question, equation in data:
        y_true.append(equation)
        output_words, attentions = evaluate_sample(encoder, decoder, input_tokenizer, output_tokenizer, question, max_len)
        output_sentence = ' '.join(output_words)
        y_pred.append(output_sentence)

        if verbose:
            print('>', question)
            print('=', equation)
            print('<', output_sentence)
            print('')

    return accuracy(y_true, y_pred)


def evaluate_and_show_attention(approach, encoder, decoder, input_tokenizer, output_tokenizer, max_len, input_sentence):
    """ Given a sample question, evaluates it and plot the attention of each parts """
    output_words, attentions = evaluate_sample(encoder, decoder, input_tokenizer, output_tokenizer, input_sentence, max_len)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    print(len(output_words))
    show_attention(approach, input_sentence, output_words, attentions)
