from __future__ import unicode_literals, print_function, division
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import numpy as np

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


def train(approach, encoder, decoder, input_tokenizer, output_tokenizer, n_iters, train_pairs, max_len, log_every=1000):
    # Training iterations
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters())  # , lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters())  # , lr=learning_rate)
    training_pairs = [tensors_from_pair(input_tokenizer, output_tokenizer, random.choice(train_pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    iterations = []
    losses = []
    train_accuracy = defaultdict(list)
    for idx in range(0, n_iters):
        input_tensor, target_tensor = training_pairs[idx]

        loss = train_sample(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_len)
        losses.append(loss)

        if (idx % log_every == 0) or (idx == n_iters - 1):
            iterations.append(idx + 1)
            avg_loss = np.mean(losses)
            plot_losses.append(avg_loss)
            print('%s (%d %d%%) %.4f' % (
                time_since(start, (idx+1) / n_iters),
                idx+1,
                (idx+1) / n_iters * 100,
                avg_loss))

            accuracy = evaluate(encoder, decoder, input_tokenizer, output_tokenizer, train_pairs, max_len)

            for key, value in accuracy.items():
                train_accuracy[key].append(value)

            losses = []

    show_loss(approach, iterations, plot_losses)
    show_accuracy(approach, iterations, train_accuracy)
