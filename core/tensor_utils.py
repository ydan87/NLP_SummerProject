import torch

from core.lang import EOS_token


def tensor_from_sentence(device, tokenizer, sentence):
    indexes = tokenizer.tokenize(sentence)
    indexes.append(EOS_token[0])
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(device, input_tokenizer, output_tokenizer, pair):
    input_tensor = tensor_from_sentence(device, input_tokenizer, pair[0])
    target_tensor = tensor_from_sentence(device, output_tokenizer, pair[1])
    return input_tensor, target_tensor
