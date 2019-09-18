from abc import abstractmethod, ABC
import numpy as np


class Dataset(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def _build_index(self, path):
        pass

    @abstractmethod
    def _split_data(self, index) -> list:
        pass

    @abstractmethod
    def _batch_split(self, split):
        pass

    @abstractmethod
    def _preprocess_batch(self, batch):

        output = [self._transform(entry) for entry in batch]
        output = np.concatenate(output)

        return output

    @abstractmethod
    def _transform(self, entry):
        return entry

    @abstractmethod
    def _save(self, output, saver):
        pass

    @abstractmethod
    def _init_save(self, save_dir):
        pass

    def preprocess(self, path, save_dir):

        index = self._build_index(path=path)
        splits = self._split_data(index=index)
        saver = self._init_save(save_dir=save_dir)
        for split in splits:
            batches = self._batch_split(split=split)
            for batch in batches:
                output = self._preprocess_batch(batch=batch)
                self._save(output=output,
                           saver=saver)


class ImageDataset(Dataset):

    def __init__(self):
        super(ImageDataset, self).__init__()

    def _build_index(self, path):
        pass

    def _split_data(self, index) -> list:
        pass

    def _batch_split(self, split):
        pass

    def _preprocess_batch(self, batch):
        pass

    def _transform(self, entry):
        return entry

    def _save(self, output, saver):
        pass

    def _init_save(self, save_dir):
        pass


dataset = ImageDataset()
print(dataset)

