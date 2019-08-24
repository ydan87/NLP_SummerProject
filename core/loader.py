import numpy as np
import pandas as pd
import os


class DataType: TEST = TRAIN = 0


class DatasetType: NUMBER_WORD_STD = 0


def get_number_word_std(data_type):
    dataset_folder = 'dataset/number_word_std'

    t = 'test' if data_type == DataType.TEST else 'dev'

    file_path = os.path.join(dataset_folder, f'number_word_std.{t}.json')
    df = pd.read_json(file_path)
    df.equations = df.equations.apply(lambda x: '; '.join(x))
    return df


def get(dataset_type, data_type, num_samples=None):
    if dataset_type == DatasetType.NUMBER_WORD_STD:
        data = get_number_word_std(data_type)
    else:
        raise NotImplementedError()

    if num_samples is None:
        return data

    return data.sample(num_samples)
