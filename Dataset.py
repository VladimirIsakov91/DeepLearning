from abc import abstractmethod, ABC
import numpy as np
import os
import random
import logging
from collections import namedtuple
logging.basicConfig(level=logging.INFO)


class Dataset(ABC):

    def __init__(self, index, labels):

        self.data = self._build_data(index=index, labels=labels)

    def _build_data(self, index, labels) -> list:
        Entry = namedtuple('entry', 'data_point labels')
        data = [Entry(data_point, label) for data_point, label in zip(index, labels)]
        return data

    @abstractmethod
    def _split_data(self, data:list, splits:list) -> list:
        pass

    @abstractmethod
    def _batch_split(self, split:list, batch_size:int) -> list:
        pass

    @abstractmethod
    def _preprocess_batch(self, batch):
        return batch

    @abstractmethod
    def _transform(self, batch):
        return batch

    @abstractmethod
    def _save(self, samples, labels, saver):
        pass

    @abstractmethod
    def _init_save(self, save_dir:str):
        pass

    def preprocess(self, save_dir:str, splits:list, batch_size:int):

        data = self._split_data(data=self.data,
                                splits=splits)

        saver = self._init_save(save_dir=save_dir)
        for split in data:
            batched_split = self._batch_split(split=split,
                                              batch_size=batch_size)
            for batch in batched_split:
                batch = self._preprocess_batch(batch=batch)


class ImageDataset(Dataset):

    def __init__(self, index, labels):
        self.logger = logging.getLogger('ImageDataset')
        super(ImageDataset, self).__init__(index=index, labels=labels)

    def _split_data(self, data, splits):

        self.logger.info('N splits {0}'.format(len(splits)))

        random.shuffle(data)

        split_index = []
        for i in range(len(splits)):
            start_position = int(len(data)*sum(splits[:i]))
            end_position = int(len(data)*sum(splits[:i+1]))
            split_index.append(data[start_position:end_position])

        return split_index

    def _batch_split(self, split, batch_size):
        split_size = len(split)
        n_batches = round(split_size/batch_size + 0.5)
        batches = [split[i*batch_size:i*batch_size+batch_size] for i in range(n_batches)]
        self.logger.info('Split size: {0} entries, N batches: {1}'.format(split_size, n_batches))
        return batches

    def _preprocess_batch(self, batch):
        batch = super(ImageDataset, self)._preprocess_batch(batch=batch)
        return batch

    def _transform(self, batch):
        return batch

    def _save(self, samples, labels, saver):
        pass

    def _init_save(self, save_dir):
        pass


data = os.listdir('C:/Users/C61124/trainingSample/0')
labels = [1 for _ in range(len(data))]
dataset = ImageDataset(index=data, labels=labels)
print(dataset.preprocess(save_dir='bla',
                         splits=[0.6, 0.2, 0.2],
                         batch_size=8))

