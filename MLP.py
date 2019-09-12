import tensorflow as tf
import tensorflow.keras.layers as tkl
import numpy as np
from DL.DNN import DNN


class MLP(DNN):

    __slots__ = '_n_features', '_n_hidden', '_activation', '_model'

    def __init__(self, n_features, n_hidden, activation):
        #self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        self._n_features = n_features
        self._n_hidden = n_hidden
        self._activation = activation

        self._model = self._build_network()

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_hidden(self):
        return self._n_hidden

    @property
    def activation(self):
        return self._activation

    @n_features.setter
    def n_features(self, value):
        pass

    @n_hidden.setter
    def n_hidden(self, value):
        pass

    @activation.setter
    def activation(self, value):
        pass

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        pass

    def _build_network(self):

        layers = []

        layers.append(tkl.Dense(self.n_hidden[0], input_dim=self.n_features))

        for i in range(1, len(self.n_hidden)-1):

            layers.append(tkl.Dense(self.n_hidden[i], activation=self.activation))

        layers.append(tkl.Dense(self.n_hidden[-1]))

        return layers

    def _forward_pass(self, data):

        for i in range(len(self.model)):
            data = self.model[i](data)

        return data

    @staticmethod
    def _get_data(data_path, n_features):

        data = np.genfromtxt(fname=data_path,
                                     delimiter=';',
                                     skip_header=1,
                                     dtype=np.float32)

        features = data[:, :n_features]
        labels = data[:, n_features:]

        return features, labels

    def train(self, session, data_path, epochs, learning_rate, batch_size, optimizer, logs_path):

        features, labels = self._get_data(data_path, self.n_features)
        features = features[:4000, :]
        labels=labels[:4000, :]

        xp = tf.placeholder("float", [None, self.n_features])
        yp = tf.placeholder("float", [None, 1])

        predictions = self._forward_pass(data=xp)

        loss_op = tf.reduce_mean(tf.losses.mean_squared_error(labels=yp,
                                                              predictions=predictions))
        tf.summary.scalar('loss', loss_op)
        merge = tf.summary.merge_all()

        if optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = optimizer.minimize(loss_op)

        init = tf.global_variables_initializer()
        session.run(init)

        summary_writer = tf.summary.FileWriter(logdir=logs_path,
                                                       graph=session.graph)
        for epoch in range(epochs):

            avg_cost = 0.
            n_batches = int(features.shape[0]/batch_size)

            for i in range(n_batches):

                x_batch = features[n_batches*batch_size: n_batches*batch_size + batch_size, :]
                y_batch = labels[n_batches*batch_size: n_batches*batch_size + batch_size, :]
                summary, _, cost = session.run([merge, train_op, loss_op], feed_dict={xp: x_batch, yp: y_batch})
                avg_cost += cost
                summary_writer.add_summary(summary, epoch+1)
            avg_cost /= n_batches

            print('Epoch {0} Loss: {1}'.format(epoch+1, avg_cost))

        summary_writer.close()

    def predict(self, session, data_path):

        features, labels = self._get_data(data_path, self.n_features)
        features = features[4000:, :]
        labels=labels[4000:, :]

        xp = tf.placeholder("float", [None, self.n_features])
        out = self._forward_pass(xp)

        pred = session.run([out], feed_dict={xp: features})

        return pred
