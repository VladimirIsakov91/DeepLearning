import os
import random


class Entry:

    def __init__(self, sample, label):
        self._sample = sample
        self._label = label

    @property
    def label(self):
        return self._label

    @property
    def sample(self):
        return self._sample


class Split:

    def __init__(self, collection, key, batch_size=8):

        self._collection = collection
        self._key = key
        self._batch_size = batch_size

    @property
    def collection(self):
        return self._collection

    @property
    def key(self):
        return self._key

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def n_classes(self):
        return len(self._collection[0].label)

    @property
    def size(self):
        return len(self._collection)

    def __str__(self):
        return 'Split - Key: {0}, Size: {1}, N Classes {2}, Batch Size: {3}'\
            .format(self._key, self.size, self.n_classes, self._batch_size)

    def __iter__(self):
        return iter(self.collection)


class Dataset:

    def __init__(self):
        pass


class ImageDataset(Dataset):

    def __init__(self, source, labels, splits, shuffle):

        super(ImageDataset, self).__init__()

        self._data = None
        self._source = source
        self._labels = labels
        self._split_spec = splits

        self._Split = Split
        self._Entry = Entry

        self._prepare_data(shuffle=shuffle)

    @property
    def data(self):
        return self._data

    @property
    def source(self):
        return self._source

    @property
    def labels(self):
        return self._labels

    @property
    def split_spec(self):
        return self._split_spec

    @property
    def _index(self):
        return (self._source + '/' + i for i in os.listdir(self._source))

    @property
    def n_splits(self):
        return len(self._split_spec)

    def __str__(self):

        return

    def __iter__(self):
        return iter(self._data)

    def _build_entries(self):

        self._data = [self._Entry(sample=sample, label=label) for sample, label in zip(self._index, self._labels)]

    def _build_splits(self, shuffle=False):

        data = self._data

        sizes = [int(i*len(data)) for i in self._split_spec.values()]

        if shuffle is True:
            random.shuffle(data)

        split_index = ([next(iter(data)) for _ in range(size)] for size in sizes)

        splits = [self._Split(collection=i, key=k) for i, k in zip(split_index, self._split_spec.keys())]

        self._data = splits

    def _prepare_data(self, shuffle):

        self._build_entries()
        self._build_splits(shuffle=shuffle)




