import zarr
import numpy


class Saver:

    def save(self):
        raise NotImplementedError


class DataSaver(Saver):

    KEY_BATCHES = {}

    def __init__(self, directory, name):
        super(DataSaver, self).__init__()
        self.directory = directory
        self.name = name + '.zarr'
        self.path = self.directory + '/' +self.name

        self._init_dir()

    def _init_dir(self):

        zarr.open(self.path, 'w')

    def save(self):
        pass

    def _prepare_batch(self, batch):
        samples = numpy.stack([entry.sample for entry in batch], axis=0)
        samples = numpy.expand_dims(samples, -1)
        labels = numpy.stack([entry.label for entry in batch], axis=0)

        return samples, labels

    @staticmethod
    def _monitor(key):
        if key not in DataSaver.KEY_BATCHES:
            DataSaver.KEY_BATCHES[key] = 0
        else:
            DataSaver.KEY_BATCHES[key] += 1

    def save_batch(self, batch, split):

        samples, labels = self._prepare_batch(batch)
        self._monitor(split.spec.key)

        root = zarr.open(self.path, 'w+')

        save_samples = root.zeros(split.spec.key + '/' + 'batch{}'.format(DataSaver.KEY_BATCHES[split.spec.key]) + '/' + 'samples',
                   shape=samples.shape, chunks=samples.shape, dtype=numpy.float32)

        save_labels = root.zeros(split.spec.key + '/' + 'batch{}'.format(DataSaver.KEY_BATCHES[split.spec.key]) + '/' + 'labels',
                   shape=labels.shape, chunks=labels.shape, dtype=numpy.int32)

        save_samples[:], save_labels[:] = samples, labels

