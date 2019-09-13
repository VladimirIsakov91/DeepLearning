import tensorflow as tf


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
        elif run['loss'] == 'cross_entropy':
            return tf.train.GradientDescentOptimizer
        else:
            raise NotImplementedError

    def _get_placeholders(self, run):

        height, width = run['image_size']
        xp = tf.placeholder("float", [None, height, width])
        yp = tf.placeholder("float", [None, 1])

        return xp, yp

    def train(self, x, y, session):

        run = self._config['run']
        xp, yp = self._get_placeholders(run)
        out = self.forward(xp)
        loss = self._get_loss(run)(yp, out)
        optimizer = self._get_optim(run)(run['learning_rate'])
        train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        session.run(init)

        for epoch in range(run['epochs']):

            avg_cost = 0.
            n_batches = int(run['image_size'][0]/run['batch_size'])

            for i in range(n_batches):

                x_batch = features[n_batches*batch_size: n_batches*batch_size + batch_size, :]
                y_batch = labels[n_batches*batch_size: n_batches*batch_size + batch_size, :]
                summary, _, cost = session.run([merge, train_op, loss_op], feed_dict={xp: x_batch, yp: y_batch})
                avg_cost += cost
                summary_writer.add_summary(summary, epoch+1)
            avg_cost /= n_batches

            print('Epoch {0} Loss: {1}'.format(epoch+1, avg_cost))



    def predict(self):
        pass

    def save(self):
        pass


net = Network({'build': [
{'name': 'conv', 'n_kernels': 32, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu'},
{'name': 'dense', 'input_dim': 1024, 'n_units': 100, 'activation': 'relu'}]})
