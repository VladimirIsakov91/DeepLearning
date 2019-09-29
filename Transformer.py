from PIL import Image, ImageFilter, ImageOps, ImageChops, ImageEnhance
from abc import abstractmethod, ABC
import numpy
from Dataset import Entry, Split


class Transformation:

    def __init__(self, func, name, origin, keys):
        self._func = func
        self._name = name
        self._origin = origin
        self._keys = keys

    @property
    def keys(self):
        return self._keys

    @property
    def func(self):
        return self._func

    @property
    def name(self):
        return self._name

    @property
    def origin(self):
        return self._origin

    def apply(self, entry):
        return self._func(entry)


class Transformer:

    def __init__(self):
        self._transformations = []
        self._Transformation = Transformation
        self._Entry = Entry

    def __str__(self):

        info = []
        separator = '\n'

        for transformation in self.transformations:
            info.append('Transformation - Name: {0}, Origin: {1}'.format(transformation.name, transformation.origin))

        info = separator.join(info)
        return info

    def __iter__(self):
        return iter(self._transformations)

    @property
    def transformations(self):
        return self._transformations

    @property
    def n_transformations(self):
        return len(self._transformations)

    def transform(self, origin):
        raise NotImplementedError


class ImageTransformer(Transformer):

    def __init__(self, image_size=None):
        super(ImageTransformer, self).__init__()
        self._image_size = image_size
        self._split_transformations = None

    @property
    def image_size(self):
        return self._image_size

    @staticmethod
    def _sample2object(sample):
        return Image.open(sample)

    @staticmethod
    def _img2aray(img):
        return numpy.asarray(img, dtype=numpy.uint8)

    def _expand_labels(self, label, n_copies):
        return numpy.array([label for _ in range(n_copies)])

    def build_transformations(self, split: Split):

        self._split_transformations = [transformation for transformation in self._transformations if split.spec.key in transformation.keys]

    def transform(self, entry: Entry):

        sample = self._sample2object(entry.sample)

        transformed = []

        for transformation in self._split_transformations:
            img = transformation.apply(sample)
            if transformation.origin is True:
                sample = img
            else:
                transformed.append(img)

        transformed.append(sample)
        labels = self._expand_labels(entry.label, len(transformed))

        entries = [Entry(sample=self._img2aray(img), label=label) for img, label in zip(transformed, labels)]

        return entries

    def add_unsharp_masking(self, keys, radius=2, scale=1, origin=False):

        name ='unsharp_masking'

        def unsharp_masking(img, rad=radius, sc=scale):
            gauss_img = img.filter(ImageFilter.GaussianBlur(radius=rad))
            return ImageChops.add(img, ImageChops.subtract(img, gauss_img, scale=sc))

        transformation = self._Transformation(unsharp_masking, name, origin, keys)
        self._transformations.append(transformation)

    def add_histogram_equalization(self, keys, origin=False):

        name = 'histogram_equalization'

        transformation = self._Transformation(ImageOps.equalize, name, origin, keys)
        self._transformations.append(transformation)

    def add_median_filter(self, keys, size=3, origin=False):

        name = 'median_filter'

        def median_filter(img):
            return img.filter(ImageFilter.MedianFilter(size=size))

        transformation = self._Transformation(median_filter, name, origin, keys)
        self._transformations.append(transformation)

    def add_resize(self, size, keys, origin=False):

        self._image_size = size

        name = 'resize'

        def resize(img, s=size):
            return img.resize(s)

        transformation = self._Transformation(resize, name, origin, keys)
        self._transformations.append(transformation)

    def add_grayscale(self, keys, origin=False):

        name = 'grayscale'

        def grayscale(img):
            return img.convert('L')

        transformation = self._Transformation(grayscale, name, origin, keys)
        self._transformations.append(transformation)

    def add_rotation(self, angles, keys, origin=False):

        name = 'rotation'

        for angle in angles:

            def rotate(img, angl=angle):
                return img.rotate(angl)

            transformation = self._Transformation(rotate, name, origin, keys)
            self._transformations.append(transformation)

    def add_brightening(self, brightnesses, keys, origin=False):

        name = 'brightening'

        for brightness in brightnesses:

            def brighten(img, br=brightness):
                return ImageEnhance.Brightness(img).enhance(br)

            transformation = self._Transformation(brighten, name, origin, keys)
            self._transformations.append(transformation)

    def add_contrast(self, contrasts, keys, origin=False):

        name = 'contrast'

        for contrast in contrasts:

            def contrasting(img, con=contrast):
                return ImageEnhance.Contrast(img).enhance(con)

            transformation = self._Transformation(contrasting, name, origin, keys)
            self._transformations.append(transformation)

    def add_sharpening(self, sharpenings, keys, origin=False):

        name = 'sharpen'

        for sharpening in sharpenings:

            def sharpen(img, sh=sharpening):
                return ImageEnhance.Contrast(img).enhance(sh)

            transformation = self._Transformation(sharpen, name, origin, keys)
            self._transformations.append(transformation)