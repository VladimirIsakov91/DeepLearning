from Parser import Parser
import tensorflow as tf
from Network import Network
import zarr
import numpy as np
import mlflow
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Runner')


class Runner:

    #PARSER = Parser()
    #ARGS = PARSER.get_args()

    @classmethod
    def run(cls, config):

        if __name__ == '__main__':

            run = config['run']
            global_step = 0
            height, width = run['size']
            xp = tf.placeholder(tf.float32, [None, height, width, 1])
            yp = tf.placeholder(tf.float32, [None, 2])
            sp = tf.placeholder(tf.float32, shape=())
            dnn = Network(config=config)
            #dnn.get_config()
            train_op, loss, optimizer = dnn.train(x=xp, y=yp, s=sp)
            accuracy = dnn.eval(xp, yp, )

            init = tf.global_variables_initializer()

            with tf.Session() as session:

                session.run(init)
                #summary_writer = tf.summary.FileWriter(logdir=run['logs_path'],
                #                                       graph=session.graph)
                with mlflow.start_run():

                    for epoch in range(run['epochs']):

                        global_step += 1
                        data = zarr.open(run['path'], 'r')

                        features = data['train']['data']
                        labels = data['train']['labels']
                        batch_size = run['batch_size']
                        avg_cost = 0.
                        total_accuracy = 0.
                        total_lr = 0
                        n_batches = round((features.shape[0] / batch_size) + 0.5)

                        for i in range(n_batches):
                            x_batch = features[i * batch_size: i * batch_size + batch_size, :, :, :]
                            y_batch = labels[i * batch_size: i* batch_size + batch_size, :]

                            _, cost, lr = session.run([train_op, loss, optimizer._lr], feed_dict={xp: x_batch,
                                                                               yp: y_batch,
                                                                               sp: global_step})
                            avg_cost += cost
                            total_lr = lr
                            logger.info('Processing Batch {0}/{1}'.format(i, n_batches))
                            #print('Processing Batch {0}/{1}'.format(i, n_batches))


                        test_features = data['test']['data']
                        test_labels = data['test']['labels']
                        test_batches = round((test_features.shape[0] / batch_size) + 0.5)


                        for k in range(test_batches):

                            x_batch = test_features[k * batch_size: k * batch_size + batch_size, :, :, :]
                            y_batch = test_labels[k * batch_size: k* batch_size + batch_size, :]

                            batch_accuracy = session.run([accuracy], feed_dict={xp: x_batch, yp: y_batch})
                            total_accuracy += batch_accuracy[0]

                        avg_cost = round(avg_cost/n_batches, 2)
                        total_accuracy = round(total_accuracy / test_batches, 2)

                        mlflow.log_metric(key="Cost", value=avg_cost, step=epoch)
                        mlflow.log_metric(key="Accuracy", value=total_accuracy, step=epoch)
                        #tf.summary.scalar('Loss', avg_cost)
                        #tf.summary.scalar('Accuracy', total_accuracy)
                        #merge = tf.summary.merge_all()
#
                        #summary = session.run(merge)
#
                        #summary_writer.add_summary(summary, epoch + 1)

                        logger.info('Epoch {0}: Loss: {1}, Accuracy{2}'.format(epoch + 1, avg_cost, total_accuracy))
                        print('Epoch {0}: Loss: {1}, Accuracy {2}, Learning Rate {3}'
                        .format(epoch + 1, avg_cost, total_accuracy, total_lr))

                    tf.saved_model.save(dnn, './model')
                    print('END')



config = {'build': [
{'name': 'conv', 'n_kernels': 16, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'elu'},
{'name': 'pool', 'pool_size': (2,2), 'stride': 2, 'padding': 'valid'},
{'name': 'conv', 'n_kernels': 32, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'elu'},
{'name': 'pool', 'pool_size': (2,2), 'stride': 2, 'padding': 'valid'},
{'name': 'conv', 'n_kernels': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'elu'},
{'name': 'pool', 'pool_size': (2,2), 'stride': 2, 'padding': 'valid'},
{'name': 'dense', 'input_dim': 16384,'n_units': 100, 'activation': 'elu', 'regularizer': 0.01},
#{'name': 'dropout', 'rate': 0.2},
{'name': 'batch_normalization'},
#{'name': 'dense', 'n_units': 20, 'activation': 'elu', 'regularizer': 0.01},
{'name': 'dense', 'n_units': 2, 'activation': None, 'regularizer': None}],
'run': {'size': (64 , 64),
        'loss': 'cross_entropy',
        'train': 'adam',
        'lr_decay': 'exp',
        'learning_rate': 0.001,
        'decay_steps': 5,
        'decay_rate': 0.5,
        'batch_size': 256,
        'epochs': 1,
        'n_classes': 2,
        'path': '/home/oem/PycharmProjects/CatsvsDogs/catsvsdogs.zarr',
        'save': './model'}}

dnn = Runner.run(config=config)
