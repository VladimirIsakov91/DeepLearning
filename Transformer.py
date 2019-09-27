from PIL import Image, ImageFilter, ImageOps, ImageChops, ImageEnhance
from abc import abstractmethod, ABC
import numpy


class Transformation:

    def __init__(self, func, name, origin):
        self._func = func
        self._name = name
        self._origin = origin

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

    @staticmethod
    def sample2object(sample):
        raise NotImplementedError

    def transform(self, origin):
        raise NotImplementedError


class ImageTransformer(Transformer):

    def __init__(self):
        super(ImageTransformer, self).__init__()
        self._image_size = None

    @property
    def image_size(self):
        return self._image_size

    @staticmethod
    def sample2object(sample):
        return Image.open(sample)

    @staticmethod
    def img2aray(img):
        return numpy.asarray(img, dtype=numpy.uint8)

    def expand_labels(self, label):
        return [label for _ in range(self.n_transformations)]

    def transform(self, origin):

        transformed = []

        for transformation in self._transformations:
            img = transformation.apply(origin)
            if transformation.origin is True:
                origin = img
            else:
                transformed.append(img)

        transformed.append(origin)

        return transformed

    def add_unsharp_masking(self, radius=2, scale=1, origin=False):

        name ='unsharp_masking'

        def unsharp_masking(img, rad=radius, sc=scale):
            gauss_img = img.filter(ImageFilter.GaussianBlur(radius=rad))
            return ImageChops.add(img, ImageChops.subtract(img, gauss_img, scale=sc))

        transformation = self._Transformation(unsharp_masking, name, origin)
        self._transformations.append(transformation)

    def add_histogram_equalization(self, origin=False):

        name = 'histogram_equalization'

        transformation = self._Transformation(ImageOps.equalize, name, origin)
        self._transformations.append(transformation)

    def add_median_filter(self, size=3, origin=False):

        name = 'median_filter'

        def median_filter(img):
            return img.filter(ImageFilter.MedianFilter(size=size))

        transformation = self._Transformation(median_filter, name, origin)
        self._transformations.append(transformation)

    def add_resize(self, size, origin=False):

        image_size = list(size)
        image_size.append(1)

        self._image_size = tuple(image_size)

        name = 'resize'

        def resize(img, s=size):
            return img.resize(s)

        transformation = self._Transformation(resize, name, origin)
        self._transformations.append(transformation)

    def add_grayscale(self, origin=False):

        name = 'grayscale'

        def grayscale(img):
            return img.convert('L')

        transformation = self._Transformation(grayscale, name, origin)
        self._transformations.append(transformation)

    def add_rotation(self, angles, origin=False):

        name = 'rotation'

        for angle in angles:

            def rotate(img, angl=angle):
                return img.rotate(angl)

            transformation = self._Transformation(rotate, name, origin)
            self._transformations.append(transformation)

    def add_brightening(self, brightnesses, origin=False):

        name = 'brightening'

        for brightness in brightnesses:

            def brighten(img, br=brightness):
                return ImageEnhance.Brightness(img).enhance(br)

            transformation = self._Transformation(brighten, name, origin)
            self._transformations.append(transformation)

    def add_contrast(self, contrasts, origin=False):

        name = 'contrast'

        for contrast in contrasts:

            def contrasting(img, con=contrast):
                return ImageEnhance.Contrast(img).enhance(con)

            transformation = self._Transformation(contrasting, name, origin)
            self._transformations.append(transformation)

    def add_sharpening(self, sharpenings, origin=False):

        name = 'sharpen'

        for sharpening in sharpenings:

            def sharpen(img, sh=sharpening):
                return ImageEnhance.Contrast(img).enhance(sh)

            transformation = self._Transformation(sharpen, name, origin)
            self._transformations.append(transformation)