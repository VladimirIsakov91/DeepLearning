import zarr


class Saver:

    def save(self):
        raise NotImplementedError


class DataSaver(Saver):

    def __init__(self, directory, name):
        super(DataSaver, self).__init__()
        self.directory = directory
        self.name = name + '.zarr'
        self.path = self.directory + self.name

    def _init_dir(self):

        zarr.open(self.path, 'w')

    def _init_split(self, split, transformer):

        samples_shape = list(transformer.image_size)
        samples_shape.insert(0, split.size)
        samples_shape = tuple(transformer.image_size)
        labels_shape = tuple([split.size, split.n_classes])
        #batch_size = tuple([split.batch_size, samples_shape])

        root = zarr.open(self.path, 'w+')
        root.zeros(split.key + '/' + 'samples', shape=samples_shape, chunks=(), dtype=())
        root.zeros(split.key + '/' + 'labels', shape=labels_shape, chunks=(), dtype=())

    def save(self):
        pass

    def save_batch(self):
        pass