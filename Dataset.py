import os
import random
from recordclass import recordclass
from collections import namedtuple
from PIL import Image, ImageFilter, ImageOps, ImageChops
import funcy


class ImageDataset:

    def __init__(self, index, labels, batch_size, splits, shuffle):

        self.data = None
        self.index = index
        self.labels = labels

        self.batch_size = batch_size
        self.splits = splits

        self.n_splits = len(self.splits)
        self.n_entries = len(index)

        self._prepare_data(shuffle=shuffle)

    def __str__(self):

        return 'ImageDataset\nN Samples: {0}\nN Splits: {1}\nBatch Size: {2}'\
            .format(self.n_entries, self.n_splits, self.batch_size)

    def _build_entries(self):

        Entry = namedtuple('entry', 'sample label')
        self.data = [Entry(sample=sample, label=label) for sample, label in zip(self.index, self.labels)]

    def _build_splits(self, shuffle=False):

        data = self.data

        sizes = (int(i*len(data)) for i in self.splits.values())

        if shuffle is True:
            random.shuffle(data)

        split_index = ([next(iter(data)) for _ in range(size)] for size in sizes)

        Split = recordclass('Split', ['collection', 'key'])
        splits = [Split(collection=i, key=k) for i, k in zip(split_index, self.splits.keys())]

        self.data = splits

    def _build_batches(self):

        for split in self.data:
            split.collection = funcy.chunks(self.batch_size, split.collection)

    def _prepare_data(self, shuffle):

        self._build_entries()
        self._build_splits(shuffle=shuffle)
        self._build_batches()


class ImageTransformer:

    def __init__(self):
        self._transformations = []

    @property
    def transformations(self):
        return self._transformations

    @transformations.setter
    def transformations(self, value):
        raise NotImplementedError

    def add_unsharp_masking(self, radius=2, scale=1):

        def gaussian_blur(img, rad=radius, sc=scale):
            gauss_img = img.filter(ImageFilter.GaussianBlur(radius=rad))
            return ImageChops.add(img, ImageChops.subtract(img, gauss_img, scale=sc))

        self._transformations.append(gaussian_blur)

    def add_histogram_equalization(self):
        self._transformations.append(ImageOps.equalize)

    def add_median_filter(self, size=3):

        def median_filter(img):
            return img.filter(ImageFilter.MedianFilter(size=size))

        self._transformations.append(median_filter)

class Saver:
    pass

class DataHandler:

    def __init__(self, transformer: ImageTransformer, dataset: ImageDataset, saver: Saver):

        self.transformer = transformer
        self.dataset = dataset
        self.saver = saver

    def transform(self):

        for entry in self.dataset.data:

            sample = entry.sample
            label = entry.label
            origin = Image.open(sample)

            for transformation in self.transformer.transformations:
                res = transformation(origin)
                res.show()



data = os.listdir('C:/Users/C61124/trainingSample/0')

labels = [1 for _ in range(len(data))]

dataset = ImageDataset(index=data,
                       labels=labels,
                       batch_size=8,
                       splits={'train': 0.6,
                               'validation:': 0.2,
                               'test': 0.2},
                       shuffle=True)

transformer = ImageTransformer()

transformer.add_unsharp_masking()
transformer.add_histogram_equalization()
transformer.add_median_filter()

saver = Saver()
handler = DataHandler(dataset=dataset,
                      transformer=transformer,
                      saver=saver)

#cat = Image.open('C:/Users/C61124/ComputerVision/cat.jpg')
#handler.transform()


