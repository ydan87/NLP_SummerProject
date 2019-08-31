import torch

from core.lang import EOS_token


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_from_sentence(tokenizer, sentence):
    indexes = tokenizer.tokenize(sentence)
    indexes.append(EOS_token[0])
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensors_from_pair(input_tokenizer, output_tokenizer, pair):
    input_tensor = tensor_from_sentence(input_tokenizer, pair[0])
    target_tensor = tensor_from_sentence(output_tokenizer, pair[1])
    return input_tensor, target_tensor
