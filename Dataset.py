from abc import abstractmethod, ABC
import numpy as np
import os
import random
import logging
from collections import namedtuple
logging.basicConfig(level=logging.INFO)
from PIL import Image


class Dataset(ABC):

    def __init__(self, index, labels):

        self.data = self._build_data(index=index, labels=labels)

    def _build_data(self, index, labels) -> list:
        Entry = namedtuple('entry', 'sample label')
        data = [Entry(sample=sample, label=label) for sample, label in zip(index, labels)]
        return data

    @abstractmethod
    def _build_splits(self, data:list, splits:dict) -> list:
        pass

    @abstractmethod
    def _batch_split(self, split:list, batch_size:int) -> list:
        pass

    @abstractmethod
    def _preprocess(self, batch, key):
        return batch

    @abstractmethod
    def _save(self, samples, labels, saver):
        pass

    @abstractmethod
    def _init_save(self, save_dir:str):
        pass

    def preprocess(self, save_dir:str, splits:dict, batch_size:int):

        data = self._build_splits(data=self.data,
                                splits=splits)

        saver = self._init_save(save_dir=save_dir)
        for split in data:
            batched_split = self._batch_split(split=split,
                                              batch_size=batch_size)
            for batch in batched_split:
                batch = self._preprocess(batch=batch,
                                         key=split.key)


class ImageDataset(Dataset):

    def __init__(self, index, labels):
        self.logger = logging.getLogger('ImageDataset')
        super(ImageDataset, self).__init__(index=index, labels=labels)

    def _build_splits(self, data, splits):

        self.logger.info('N splits {0}'.format(len(splits)))

        random.shuffle(data)

        split_index = []
        values = list(splits.values())
        for i in range(len(splits)):
            start_position = int(len(data)*sum(values[:i]))
            end_position = int(len(data)*sum(values[:i+1]))
            split_index.append(data[start_position:end_position])

        Split = namedtuple('Split', 'data key')
        splits = [Split(data=i, key=k) for i, k in zip(split_index, splits.keys())]

        return splits

    def _batch_split(self, split, batch_size):
        split_size = len(split.data)
        n_batches = round(split_size/batch_size + 0.5)
        batches = [split.data[i*batch_size:i*batch_size+batch_size] for i in range(n_batches)]
        self.logger.info('Split size: {0} entries, N batches: {1}'.format(split_size, n_batches))
        return batches

    def _preprocess(self, batch, key):

        for entry in batch:
            sample = entry.sample
            label = entry.label
            image = Image.open('/home/oem/PycharmProjects/CatsvsDogs/train/' + sample)
            image = image.convert('L')




        return batch

    def _save(self, samples, labels, saver):
        pass

    def _init_save(self, save_dir):
        pass


data = os.listdir('/home/oem/PycharmProjects/CatsvsDogs/train')
labels = [1 for _ in range(len(data))]
dataset = ImageDataset(index=data, labels=labels)
print(dataset.preprocess(save_dir='bla',
                         splits={'train':0.6, 'validation:':0.2, 'test': 0.2},
                         batch_size=256))

