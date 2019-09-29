import os
from Transformer import ImageTransformer
from Saver import DataSaver
from Dataset import ImageDataset
from DataHandler import DataHandler

data = os.listdir('/home/oem/train')

labels = [[1, 0] if 'cat' in i else [0, 1] for i in data]

dataset = ImageDataset(source='/home/oem/train',
                       labels=labels,
                       split_spec={'train': {'amount': 0.6, 'transform': True, 'batch_size':32},
                               'validation': {'amount': 0.2, 'transform': False, 'batch_size':32},
                               'test': {'amount': 0.2, 'transform': False, 'batch_size':32}},
                       shuffle=True)

transformer = ImageTransformer()

transformer.add_resize((256, 256), origin=True, keys=['train', 'validation', 'test'])
transformer.add_grayscale(origin=True, keys=['train', 'validation', 'test'])
transformer.add_unsharp_masking(origin=True, keys=['train', 'validation', 'test'])
transformer.add_histogram_equalization(origin=True, keys=['train', 'validation', 'test'])
transformer.add_median_filter(keys=['train'])
transformer.add_rotation([45, 60, 90], keys=['train'])
transformer.add_contrast([1.5], keys=['train'])
transformer.add_brightening([0.5], keys=['train'])
transformer.add_sharpening([2.0], keys=['train'])

saver = DataSaver('/home/oem/PycharmProjects/DeepLearning', 'CatsvsDogs')
handler = DataHandler(dataset=dataset,
                      transformer=transformer,
                      saver=saver)

handler.run()
