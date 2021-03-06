from __future__ import unicode_literals, print_function, division
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd

from core.evaluation import evaluate
from core.lang import EOS_token, SOS_token
from core.tensor_utils import tensors_from_pair, DEVICE
from core.time_utils import time_since
from core.visualizations import show_loss, show_accuracy

_TEACHER_FORCING_RATIO = 0.5


def train_sample(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    """ training procedure """
    # Performs one iteration of training gets the current loss
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token[0]]], device=DEVICE)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = random.random() < _TEACHER_FORCING_RATIO

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token[0]:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train(approach, encoder, decoder, input_tokenizer, output_tokenizer, n_iters, train_pairs, test_pairs, max_len, log_every=1000):
    # Training iterations
    start = time.time()

    encoder_optimizer = optim.Adam(encoder.parameters())  # , lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters())  # , lr=learning_rate)
    training_pairs = [tensors_from_pair(input_tokenizer, output_tokenizer, random.choice(train_pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    iterations = []
    loss_history = []
    train_accuracy_history = defaultdict(list)
    test_accuracy_history = defaultdict(list)

    losses = []
    for idx in range(0, n_iters):
        input_tensor, target_tensor = training_pairs[idx]

        loss = train_sample(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_len)
        losses.append(loss)

        if (idx % log_every == 0) or (idx == n_iters - 1):
            iterations.append(idx + 1)
            avg_loss = np.mean(losses)
            loss_history.append(avg_loss)

            train_accuracy = evaluate(encoder, decoder, input_tokenizer, output_tokenizer, train_pairs, max_len)

            for key, value in train_accuracy.items():
                train_accuracy_history[key].append(value)

            test_accuracy = evaluate(encoder, decoder, input_tokenizer, output_tokenizer, test_pairs, max_len)

            for key, value in test_accuracy.items():
                test_accuracy_history[key].append(value)

            print('%s (%d %d%%) %.4f' % (
                time_since(start, (idx+1) / n_iters),
                idx+1,
                (idx+1) / n_iters * 100,
                avg_loss))

            losses = []

    df_train = pd.DataFrame.from_dict(train_accuracy_history)
    df_train.index = iterations
    df_test = pd.DataFrame.from_dict(test_accuracy_history)
    df_test.index = iterations

    df = pd.merge(left=df_train, right=df_test, left_index=True, right_index=True, suffixes=['_train', '_test'])
    df.to_msgpack(f'results/{approach}.msg')

    show_loss(approach, iterations, loss_history)
    show_accuracy(approach, iterations, train_accuracy_history)
    show_accuracy(approach, iterations, test_accuracy_history, is_train=False)
