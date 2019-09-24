import os
import random
from recordclass import recordclass
from collections import namedtuple
from PIL import Image, ImageFilter, ImageOps, ImageChops
import funcy
import numpy


class ImageDataset:

    def __init__(self, source, index, labels, batch_size, splits, shuffle):

        self.data = None
        self.source = source
        self.index = (self.source + '/' + name for name in index)
        self.labels = labels
        self.batch_size = batch_size
        self.splits = splits

        self.n_splits = len(self.splits)
        self.n_entries = len(index)

        self._prepare_data(shuffle=shuffle)

    @classmethod
    def _init_entry(cls, sample, label):
        Entry = namedtuple('Entry', ['sample', 'label'])
        return Entry(sample, label)

    def __str__(self):

        return 'ImageDataset\nN Samples: {0}\nN Splits: {1}\nBatch Size: {2}'\
            .format(self.n_entries, self.n_splits, self.batch_size)

    def _build_entries(self):

        self.data = [ImageDataset._init_entry(sample=sample, label=label) for sample, label in zip(self.index, self.labels)]

    def _build_splits(self, shuffle=False):

        data = self.data

        sizes = (int(i*len(data)) for i in self.splits.values())

        if shuffle is True:
            random.shuffle(data)

        split_index = ([next(iter(data)) for _ in range(size)] for size in sizes)

        Split = recordclass('Split', ['collection', 'name'])
        splits = [Split(collection=i, name=k) for i, k in zip(split_index, self.splits.keys())]

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

    def __str__(self):

        info = []
        separator = '\n'

        for transformation in self.transformations:
            info.append('Transformation - Name: {0}, Origin: {1}'.format(transformation.name, transformation.origin))

        info = separator.join(info)
        return info

    @property
    def transformations(self):
        return self._transformations

    @transformations.setter
    def transformations(self, value):
        raise NotImplementedError

    @classmethod
    def _init_transformation(cls, func, name, origin):
        Transformation = namedtuple('Transformation', ['func', 'name', 'origin'])
        return Transformation(func, name, origin)

    def add_unsharp_masking(self, radius=2, scale=1, origin=False):

        name='unsharp_masking'

        def unsharp_masking(img, rad=radius, sc=scale):
            gauss_img = img.filter(ImageFilter.GaussianBlur(radius=rad))
            return ImageChops.add(img, ImageChops.subtract(img, gauss_img, scale=sc))

        transformation = ImageTransformer._init_transformation(unsharp_masking, name, origin)
        self._transformations.append(transformation)

    def add_histogram_equalization(self, origin=False):

        name = 'histogram_equalization'

        transformation = ImageTransformer._init_transformation(ImageOps.equalize, name, origin)
        self._transformations.append(transformation)

    def add_median_filter(self, size=3, origin=False):

        name = 'median_filter'

        def median_filter(img):
            return img.filter(ImageFilter.MedianFilter(size=size))

        transformation = ImageTransformer._init_transformation(median_filter, name, origin)
        self._transformations.append(transformation)

    def add_resize(self, size, origin=False):

        name = 'resize'

        def resize(img, s = size):
            return img.resize(s)

        transformation = ImageTransformer._init_transformation(resize, name, origin)
        self._transformations.append(transformation)


    def add_grayscale(self, origin=False):

        name = 'grayscale'

        def grayscale(img):
            return img.convert('L')

        transformation = ImageTransformer._init_transformation(grayscale, name, origin)
        self._transformations.append(transformation)

    def add_rotation(self, angle, origin=False):

        name = 'rotation'

        def rotate(img):
            return img.rotate(angle)

        transformation = ImageTransformer._init_transformation(rotate, name, origin)
        self._transformations.append(transformation)


class Saver:
    pass


class DataHandler:

    def __init__(self, transformer: ImageTransformer, dataset: ImageDataset, saver: Saver):

        self.transformer = transformer
        self.dataset = dataset
        self.saver = saver

    def transform(self):

        for split in self.dataset.data:
            print(split.name)
            for batch in split.collection:

                transformed_batch = []

                for entry in batch:

                    origin = Image.open(entry.sample)

                    transformed = []

                    for transformation in self.transformer.transformations:
                        img = transformation.func(origin)
                        if transformation.origin is True:
                            origin = img
                        else:
                            transformed.append(img)

                    transformed_batch.append(origin)
                    transformed_batch.extend(transformed)

                transformed_batch = [DataHandler.img2aray(img) for img in transformed_batch]
                batch = numpy.stack(transformed_batch, axis=0)
                print(batch.shape)

    @staticmethod
    def img2aray(img):
        return numpy.asarray(img, dtype=numpy.uint8)


data = os.listdir('/home/oem/PycharmProjects/CatsvsDogs/images')

labels = [1 for _ in range(len(data))]

dataset = ImageDataset(source='/home/oem/PycharmProjects/CatsvsDogs/images',
                        index=data,
                       labels=labels,
                       batch_size=2,
                       splits={'train': 0.6,
                               'test': 0.4},
                       shuffle=True)

transformer = ImageTransformer()

transformer.add_resize((64,64),origin=True)
transformer.add_grayscale(origin=True)
transformer.add_unsharp_masking(origin=True)
transformer.add_histogram_equalization(origin=True)
transformer.add_median_filter()
transformer.add_rotation(60)

saver = Saver()
handler = DataHandler(dataset=dataset,
                      transformer=transformer,
                      saver=saver)

print(dataset)
print(transformer)
handler.transform()
#cat = Image.open('C:/Users/C61124/ComputerVision/cat.jpg')
#handler.transform()


