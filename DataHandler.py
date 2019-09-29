from Transformer import Transformer, ImageTransformer
from Dataset import Dataset, ImageDataset
from Saver import DataSaver


class DataHandler:

    def __init__(self, transformer: ImageTransformer, dataset: ImageDataset, saver: DataSaver):

        self.transformer = transformer
        self.dataset = dataset
        self.saver = saver


    def run(self):

        for split in self.dataset:

            print(split)
            print(split.spec)

            batch = []

            for entry in split:

                self.transformer.build_transformations(split)
                entries = self.transformer.transform(entry)
                batch.extend(entries)

                if len(batch) >= split.spec.batch_size:

                    self.saver.save_batch(batch=batch,
                                          split=split)

                    batch = []