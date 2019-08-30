import torch

from core.lang import EOS_token


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(device, lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token[0])
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(device, input_lang, output_lang, pair):
    input_tensor = tensor_from_sentence(device, input_lang, pair[0])
    target_tensor = tensor_from_sentence(device, output_lang, pair[1])
    return (input_tensor, target_tensor)
