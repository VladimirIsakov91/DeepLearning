from abc import abstractmethod, ABC
import numpy as np
import os
import random
import logging
logging.basicConfig(level=logging.INFO)


class Dataset(ABC):


    def __init__(self, path):
        self.index = self._build_index(path=path)

    @abstractmethod
    def _build_index(self, path:str) -> list:
        pass

    @abstractmethod
    def _build_labels(self, **kwargs) -> list:
        pass

    @abstractmethod
    def _split_data(self, index:list, labels:list, splits:list) -> list:
        pass

    @abstractmethod
    def _batch_split(self, samples:list, labels:list, batch_size:int) -> list:
        pass

    @abstractmethod
    def _preprocess_batch(self, samples, labels):
        return samples, labels

    @abstractmethod
    def _transform(self, sample, label: int):
        return sample, label

    @abstractmethod
    def _save(self, samples, labels, saver):
        pass

    @abstractmethod
    def _init_save(self, save_dir:str):
        pass

    def preprocess(self, save_dir:str, splits:list, batch_size:int):

        index = self.index
        labels = self._build_labels(index=index)
        index, labels = self._split_data(index=index,
                                         labels=labels,
                                         splits=splits)

        saver = self._init_save(save_dir=save_dir)
        for index_split, label_split in zip(index, labels):
            sample_batches, label_batches = self._batch_split(samples=index_split,
                                                              labels=label_split,
                                                              batch_size=batch_size)
            for sample_batch, label_batch in zip(sample_batches, label_batches):
                samples, labels = self._preprocess_batch(samples=sample_batch,
                                                labels=label_batch)
                self._save(samples=samples,
                           labels=labels,
                           saver=saver)


class ImageDataset(Dataset):

    def __init__(self, path):
        self.logger = logging.getLogger('ImageDataset')
        super(ImageDataset, self).__init__(path=path)

    def _build_index(self, path):
        index = os.listdir(path)
        self.logger.info('Index size {0}'.format(len(index)))
        return index

    def _build_labels(self, index):
        labels = [[1, 0] if 'cat' in name else [0, 1] for name in self.index]
        self.logger.info('N classes {0}'.format(len(labels[0])))
        return labels

    def _split_data(self, index, labels, splits):

        self.logger.info('N splits {0}'.format(len(splits)))

        rng = list(zip(index, labels))
        random.shuffle(rng)
        index, labels = zip(*rng)

        split_index, split_labels = [], []
        for i in range(len(splits)):
            start_position = int(len(index)*sum(splits[:i]))
            end_position = int(len(index)*sum(splits[:i+1]))
            split_index.append(index[start_position:end_position])
            split_labels.append(labels[start_position:end_position])

        return split_index, split_labels

    def _batch_split(self, samples, labels, batch_size):
        split_size = len(samples)
        n_batches = round(split_size/batch_size + 0.5)
        sample_batches = [samples[i*batch_size:i*batch_size+batch_size] for i in range(n_batches)]
        labels_batches = [labels[i * batch_size:i * batch_size + batch_size] for i in range(n_batches)]
        self.logger.info('Split size: {0} entries, N batches: {1}'.format(split_size, n_batches))
        return sample_batches, labels_batches

    def _preprocess_batch(self, samples, labels):
        samples, labels= super(ImageDataset, self)._preprocess_batch(samples=samples,
                                                                labels=labels)
        return samples, labels

    def _transform(self, sample, label):
        return sample, label

    def _save(self, samples, labels, saver):
        pass

    def _init_save(self, save_dir):
        pass


dataset = ImageDataset('/home/oem/PycharmProjects/CatsvsDogs/train')
print(dataset.preprocess(save_dir='bla',
                         splits=[0.6, 0.2, 0.2],
                         batch_size=256))

