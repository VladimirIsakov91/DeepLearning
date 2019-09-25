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

        self.data = None
        self.source = source
        self.index = (source + '/' + i for i in os.listdir(source))
        self.labels = labels
        self.splits = splits

        self._split = Split
        self._entry = Entry

        self.n_splits = len(self.splits)

        self._prepare_data(shuffle=shuffle)

    def __str__(self):

        return

    def __iter__(self):
        return iter(self.data)

    def _build_entries(self):

        self.data = [self._entry(sample=sample, label=label) for sample, label in zip(self.index, self.labels)]

    def _build_splits(self, shuffle=False):

        data = self.data

        sizes = [int(i*len(data)) for i in self.splits.values()]

        if shuffle is True:
            random.shuffle(data)

        split_index = ([next(iter(data)) for _ in range(size)] for size in sizes)

        splits = [self._split(collection=i, key=k) for i, k in zip(split_index, self.splits.keys())]

        self.data = splits

    def _prepare_data(self, shuffle):

        self._build_entries()
        self._build_splits(shuffle=shuffle)




