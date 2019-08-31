import unicodedata
import re


def unicode_to_ascii(s):
    """Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    """Lowercase, trim, and remove non-letter characters"""
    s = unicode_to_ascii(s)

    # ensure each math symbol is it's own token
    s = "".join([c if c.isalnum() else " {} ".format(c) for c in s])

    # remove the unknowns since tools like wolfram are good at identifying them
    s = s[s.index(';') + 1:]

    # remove all extra whitespaces
    s = " ".join(s.split())

    return s


def remove_punctuation(s):
    """ Removes all punctuation tokens from text"""
    return re.subn(r"""[!.><:;',@#~{}\[\]\-_+=£$%^&()?]""", "", s, count=0, flags=0)[0]


def generalize(text, equation):
    """ Replaces all numeric values with general variables """
    word_var_mapping = dict()
    var_idx = 1

    words = []
    for s in text.split(' '):
        if s.isdigit():
            var = f'var{var_idx}'
            words.append(var)
            word_var_mapping[s] = var

            var_idx += 1
        else:
            words.append(s)

    general_text = ' '.join(words)

    words = []
    for s in equation.split(' '):
        if s in word_var_mapping:
            words.append(word_var_mapping[s])
        else:
            words.append(s)

    general_equation = ' '.join(words)

    return general_text, general_equation


def text2int(textnum, numwords={}):
    """  Replaces all numbers that are written as words with their numeric representation """
    if not numwords:
        units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    curstring = ""
    onnumber = False
    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    return curstring
