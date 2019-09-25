from collections import namedtuple
from PIL import Image, ImageFilter, ImageOps, ImageChops, ImageEnhance
from abc import abstractmethod, ABC
import numpy


class Transformer:

    def __init__(self):
        self._transformations = []

    def __str__(self):

        info = []
        separator = '\n'

        for transformation in self.transformations:
            info.append('Transformation - Name: {0}, Origin: {1}'.format(transformation.name, transformation.origin))

        info = separator.join(info)
        return info

    def __iter__(self):
        return iter(self._transformations)

    @classmethod
    def _init_transformation(cls, func, name, origin):
        Transformation = namedtuple('Transformation', ['func', 'name', 'origin'])
        return Transformation(func, name, origin)

    @property
    def transformations(self):
        return self._transformations

    @transformations.setter
    def transformations(self, value):
        raise NotImplementedError

    @staticmethod
    def sample2object(sample):
        raise NotImplementedError


class ImageTransformer(Transformer):

    def __init__(self):
        super(ImageTransformer, self).__init__()
        self.image_size = None

    @staticmethod
    def sample2object(sample):
        return Image.open(sample)

    @staticmethod
    def img2aray(img):
        return numpy.asarray(img, dtype=numpy.uint8)

    def add_unsharp_masking(self, radius=2, scale=1, origin=False):

        name ='unsharp_masking'

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

        image_size = list(size)
        image_size.append(1)

        self.image_size = tuple(image_size)

        name = 'resize'

        def resize(img, s=size):
            return img.resize(s)

        transformation = ImageTransformer._init_transformation(resize, name, origin)
        self._transformations.append(transformation)

    def add_grayscale(self, origin=False):

        name = 'grayscale'

        def grayscale(img):
            return img.convert('L')

        transformation = ImageTransformer._init_transformation(grayscale, name, origin)
        self._transformations.append(transformation)

    def add_rotation(self, angles, origin=False):

        name = 'rotation'

        for angle in angles:

            def rotate(img, angl=angle):
                return img.rotate(angl)

            transformation = ImageTransformer._init_transformation(rotate, name, origin)
            self._transformations.append(transformation)

    def add_brightening(self, brightnesses, origin=False):

        name = 'brightening'

        for brightness in brightnesses:

            def brighten(img, br=brightness):
                return ImageEnhance.Brightness(img).enhance(br)

            transformation = ImageTransformer._init_transformation(brighten, name, origin)
            self._transformations.append(transformation)

    def add_contrast(self, contrasts, origin=False):

        name = 'contrast'

        for contrast in contrasts:

            def contrasting(img, con=contrast):
                return ImageEnhance.Contrast(img).enhance(con)

            transformation = ImageTransformer._init_transformation(contrasting, name, origin)
            self._transformations.append(transformation)

    def add_sharpening(self, sharpenings, origin=False):

        name = 'sharpen'

        for sharpening in sharpenings:

            def sharpen(img, sh=sharpening):
                return ImageEnhance.Contrast(img).enhance(sh)

            transformation = ImageTransformer._init_transformation(sharpen, name, origin)
            self._transformations.append(transformation)