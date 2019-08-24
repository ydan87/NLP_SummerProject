import abc
from abc import ABC

from core.lang import Lang, EOS_token
from core.string_utils import normalize_string


class Tokenizer(ABC):
    def __init__(self, df_train, df_test, **options):
        self.options = options
        self.normalizer = options.get('normalizer', normalize_string)

        self._build(df_train, df_test)

    @abc.abstractproperty
    def input_size(self):
        pass

    @abc.abstractproperty
    def output_size(self):
        pass

    @abc.abstractmethod
    def _build(self, df_train, df_test):
        pass

    @abc.abstractmethod
    def tokenize(self, sample, is_target=False):
        pass

    @abc.abstractmethod
    def untokenize(self, indices, is_target=False):
        pass


class SimpleTokenizer(Tokenizer):
    def __init__(self, df_train, df_test):
        self.input_lang = Lang('text')
        self.output_lang = Lang('equations')

        super(SimpleTokenizer, self).__init__(df_train, df_test, normalize=True)

    @property
    def input_size(self):
        return self.input_lang.n_words

    @property
    def output_size(self):
        return self.output_lang.n_words

    def _build(self, df_train, df_test):
        if self.normalizer:
            df_train.equations = df_train.equations.apply(lambda x: self.normalizer(x))
            df_test.equations = df_test.equations.apply(lambda x: self.normalizer(x))

        train_pairs = [list(x) for x in df_train[['text', 'equations']].to_records(index=False)]
        test_pairs = [list(x) for x in df_test[['text', 'equations']].to_records(index=False)]

        for pairs in [train_pairs, test_pairs]:
            for pair in pairs:
                self.input_lang.add_sentence(pair[0])
                self.output_lang.add_sentence(pair[1])

    def tokenize(self, sample, is_target=False):
        lang = self.input_lang if is_target is False else self.output_lang

        indices = [lang.word2index[word] for word in sample.split(' ')]
        indices.append(EOS_token)

        return indices

    def untokenize(self, indices, is_target=False):
        pass


