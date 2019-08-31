from core.lang import Lang


class Tokenizer:
    def __init__(self, name):
        self.name = name
        self.num_words = 0

        self.lang = Lang(name)

    def build(self, data):
        for sentence in data[self.name]:
            self.lang.add_sentence(sentence)

        return self

    def tokenize(self, text):
        return [self.lang.word2index[word] for word in text.split(' ')]

    def untokenize(self, indices):
        if hasattr(indices, '__iter__'):
            return ' '.join([self.lang.index2word[idx] for idx in indices])
        else:
            return self.lang.index2word[indices]

