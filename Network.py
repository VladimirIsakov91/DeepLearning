import tensorflow as tf
import numpy as np

class Network:

    def __init__(self, config: dict):
        self._config = config
        self._model = self._build_model()

    def _build_model(self):

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
                model.append(tf.keras.layers.MaxPooling2D(layer['pool_size'],
                                                          layer['stride'],
                                                          layer['padding']))

            elif layer['name'] == 'batch_normalization':
                model.append(tf.keras.layers.BatchNormalization())

            elif layer['name'] == 'dropout':
                model.append(tf.keras.layers.Dropout(layer['rate']))

            elif layer['name'] == 'dense':

                if dense is False:
                    model.append(tf.keras.layers.Flatten())
                    model.append(tf.keras.layers.Dense(input_dim=layer['input_dim'],
                                                        units=layer['n_units'],
                                                        activation=layer['activation'],
                                                        kernel_regularizer=tf.keras.regularizers.l2(layer['regularizer'])))
                    dense = True

                else:
                    model.append(tf.keras.layers.Dense(units=layer['n_units'],
                                                        activation=layer['activation'],
                                                        kernel_regularizer=tf.keras.regularizers.l2(layer['regularizer'])))
        return model

    def get_config(self):
        for value in self._config['build']:
            print(value)

    def forward(self, inp):
        for layer in self._model:
            inp = layer(inp)
            #print(inp.shape)
        return inp

    def _get_loss(self, run):

        if run['loss'] == 'mse':
            return tf.losses.mean_squared_error
        elif run['loss'] == 'cross_entropy':
            return tf.losses.softmax_cross_entropy
        else:
            raise NotImplementedError

    def _get_optim(self, run):

        if run['train'] == 'adam':
            return tf.train.AdamOptimizer
        elif run['loss'] == 'gradient_descent':
            return tf.train.GradientDescentOptimizer
        else:
            raise NotImplementedError

    def _get_learning_rate(self, run, s):
        if run['lr_decay'] == 'exp':
            return tf.train.exponential_decay(run['learning_rate'],
                                              s,
                                              run['decay_steps'],
                                              run['decay_rate'],
                                              staircase=True)
        else:
            raise NotImplementedError
    #@tf.function
    def train(self, x, y, s):

        run = self._config['run']
        out = self.forward(x)
        loss = self._get_loss(run)(y, out)
        loss = tf.reduce_mean(loss)
        learning_rate = self._get_learning_rate(run, s)
        optimizer = self._get_optim(run)(learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op, loss, optimizer


    def predict(self, x, y):
        out = self.forward(x)
        out = tf.math.argmax(out, axis=1)
        y = tf.math.argmax(y, axis=1)
        compare = tf.where(tf.equal(out, y))
        n_samples = tf.shape(y)[0]
        n_correct = tf.shape(compare)[0]
        accuracy = tf.math.divide(n_correct, n_samples)
        return accuracy

    def save(self):
        pass
