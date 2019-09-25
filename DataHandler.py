from Transformer import Transformer, ImageTransformer
from Dataset import Dataset, ImageDataset
from Saver import DataSaver
import numpy


class DataHandler:

    def __init__(self, transformer: ImageTransformer, dataset: ImageDataset, saver: DataSaver):

        self.transformer = transformer
        self.dataset = dataset
        self.saver = saver

        #self.saver._init_dir()

    def transform(self):

        for split in self.dataset:

            print(split)
            #self.saver._init_split(split=split,
            #                       transformer=self.transformer)

            sample_batch = []
            label_batch = []

            for entry in split:

                origin = self.transformer.sample2object(entry.sample)

                transformed = []

                for transformation in self.transformer:
                    img = transformation.func(origin)
                    if transformation.origin is True:
                        origin = img
                    else:
                        transformed.append(img)

                sample_batch.append(origin)
                sample_batch.extend(transformed)
                labels = numpy.array([entry.label for _ in range(len(sample_batch))])
                label_batch.append(labels)

                if len(sample_batch) >= split.batch_size:
                    sample_batch = [self.transformer.img2aray(img) for img in sample_batch]
                    batch = numpy.stack(sample_batch, axis=0)
                    batch = numpy.expand_dims(batch, -1)
                    label_batch = numpy.concatenate(label_batch)
                    print(batch.shape, label_batch.shape)

                    sample_batch = []
                    label_batch = []

