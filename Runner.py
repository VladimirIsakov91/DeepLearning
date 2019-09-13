from Parser import Parser


class Runner:

    PARSER = Parser()
    ARGS = PARSER.get_args()

    @classmethod
    def run(cls, model, params):

        if model == 'MLP':

            from MLP import MLP
            import tensorflow as tf

            if __name__ == '__main__':

                with tf.Session() as session:

                    dnn = MLP(n_features=params['n_features'],
                      n_hidden=params['n_hidden'],
                      activation=params['activation'])

                    dnn.train(session=session,
                         data_path=params['data_path'],
                         epochs=params['epochs'],
                         learning_rate=params['learning_rate'],
                         batch_size=params['batch_size'],
                         optimizer=params['optimizer'],
                         logs_path=params['logs_path'])

                    dnn.predict(session=session,
                                data_path=params['data_path'])


Runner.run('MLP', params={'n_features': 11,
                          'n_hidden': [100, 60, 20, 1],
                          'activation': 'relu',
                          'data_path': '/home/oem/PythonProjects/AWS_Sagemaker_estimators/winequality-white.csv',
                          'epochs': 100,
                          'learning_rate': 0.01,
                          'batch_size': 256,
                          'optimizer': 'adam',
                          'logs_path': '/home/oem/PythonProjects/AWS_dnn'})
