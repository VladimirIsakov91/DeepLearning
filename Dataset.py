import os
import random


class Entry:

    def __init__(self, sample, label):
        self._sample = sample
        self._label = label

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, value):
        self._sample = value

class Spec:

    def __init__(self, key, amount, batch_size):

        self._key = key
        self._amount = amount
        self._batch_size = batch_size

    @property
    def key(self):
        return self._key

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def amount(self):
        return self._amount

    def __str__(self):
        return 'Spec - Key: {0}, Amount: {1}, Batch Size: {2}'\
            .format(self.key, self.amount, self.batch_size)

class Split:

    def __init__(self, collection, spec: Spec):

        self._collection = collection
        self._spec = spec

    @property
    def spec(self):
        return self._spec

    @property
    def collection(self):
        return self._collection

    @property
    def n_classes(self):
        return len(self._collection[0].label)

    @property
    def size(self):
        return len(self._collection)

    def __str__(self):
        return 'Split - Size: {0}, N Classes {1}'\
            .format(self.size, self.n_classes)

    def __iter__(self):
        return iter(self.collection)

class Dataset:

    def __init__(self):
        pass


class ImageDataset(Dataset):

    def __init__(self, source, labels, split_spec, shuffle):

        super(ImageDataset, self).__init__()

        self._Split = Split
        self._Entry = Entry
        self._Spec = Spec

        self._data = None
        self._source = source
        self._labels = labels
        self._split_spec = split_spec

        self._prepare_data(shuffle=shuffle)

    @property
    def specs(self):
        return [self._Spec(key=k, amount=v['amount'], batch_size=v['batch_size'])
                for k, v in self._split_spec.items()]

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

        sizes = [int(spec.amount*len(data)) for spec in self.specs]

        if shuffle is True:
            random.shuffle(data)

        data = iter(data)
        split_index = ([next(data) for _ in range(size)] for size in sizes)

        splits = [self._Split(collection=i, spec=spec) for i, spec in zip(split_index, self.specs)]

        self._data = splits

    def _prepare_data(self, shuffle):

        self._build_entries()
        self._build_splits(shuffle=shuffle)
