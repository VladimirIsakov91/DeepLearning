import tensorflow as tf
import pprint


class Network():

    def __init__(self, config: dict):
        self._config = config
        self._model = self._build()

    def _build(self):

        model = []
        layers = self._config['build']
        dense = False

        for layer in layers:
            if layer['name'] == 'conv':

                model.append(tf.keras.layers.Conv2D(filters=layer['n_kernels'],
                                                    kernel_size=layer['kernel_size'],
                                                    strides=layer['strides'],
                                                    padding=layer['padding'],
                                                    activation=layer['activation']))
            elif layer['name'] == 'pool':
                model.append(tf.keras.layers.MaxPooling2D(layer['pool_size']))

            elif layer['name'] == 'dense':

                if dense is False:
                    model.append(tf.keras.layers.Dense(input_dim=layer['input_dim'],
                                                        units=layer['n_units'],
                                                        activation=layer['activation']))
                    dense = True

                else:
                    model.append(tf.keras.layers.Dense(units=layer['n_units'],
                                                        activation=layer['activation']))
        return model

    def get_config(self):
        for value in self._config['build']:
            print(value)

    def forward(self, inp):
        for layer in self._model:
            inp = layer(inp)
        return inp

    def train(self):

        run = self._config['run']

    def predict(self):
        pass

    def save(self):
        pass


net = Network({'build': [
{'name': 'conv', 'n_kernels': 32, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu'},
{'name': 'dense', 'input_dim': 1024, 'n_units': 100, 'activation': 'relu'}]})
