SOS_token = 0, 'SOS'
EOS_token = 1, 'EOS'


class Lang:
    """ This class represents the vocabulary """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}

        for idx, value in [SOS_token, EOS_token]:
            self.index2word[idx] = value

        self.n_words = len(self.index2word)  # Count SOS and EOS

    def add_sentence(self, sentence):
        """ Adds sentence to vocabulary """
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """ Adds a single word to the vocabulary """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
