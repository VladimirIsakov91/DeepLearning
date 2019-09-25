import os
from Transformer import ImageTransformer
from Saver import DataSaver
from Dataset import ImageDataset
from DataHandler import DataHandler

data = os.listdir('C:/Users/C61124/trainingSample/0')

labels = [[1, 0] for _ in range(len(data))]

dataset = ImageDataset(source='C:/Users/C61124/trainingSample/0',
                       labels=labels,
                       splits={'train': 0.6,
                               'test': 0.4},
                       shuffle=True)

transformer = ImageTransformer()

transformer.add_resize((64, 64), origin=True)
transformer.add_grayscale(origin=True)
transformer.add_unsharp_masking(origin=True)
transformer.add_histogram_equalization(origin=True)
transformer.add_median_filter()
transformer.add_rotation([45, 60, 90])
transformer.add_contrast([1.5])
transformer.add_brightening([0.5])
transformer.add_sharpening([2.0])

saver = DataSaver('bla', 'ble')
handler = DataHandler(dataset=dataset,
                      transformer=transformer,
                      saver=saver)

print(transformer)
#cat = Image.open('C:/Users/C61124/ComputerVision/cat.jpg')
handler.transform()