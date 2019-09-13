import argparse
import os


class Parser:

    @classmethod
    def get_args(cls):

        parser = argparse.ArgumentParser()

        parser.add_argument("--model",
                            required=True,
                            help="DNN type",
                            type=str)

        parser.add_argument("--n_features",
                            required=True,
                            help="number of trainable features",
                            type=int)

        parser.add_argument("--n_hidden",
                            required=True,
                            help="list which contains number of neurons for each hidden layer",
                            type=int,
                            nargs='+')

        parser.add_argument("--activation",
                            required=True,
                            help="layer activation function",
                            type=str)

        parser.add_argument("--train_path",
                            help="path to data for training",
                            type=str,
                            default=os.environ.get('SM_CHANNEL_TRAIN'))

        parser.add_argument("--test_path",
                            help="path to data for testing",
                            type=str,
                            default=os.environ.get('SM_CHANNEL_TEST'))

        parser.add_argument("--epochs",
                            required=True,
                            help="number of epochs for training",
                            type=int)

        parser.add_argument("--learning_rate",
                            required=True,
                            help="learning rate",
                            type=float)

        parser.add_argument("--batch_size",
                            required=True,
                            help="batch size",
                            type=int)

        parser.add_argument("--optimizer",
                            required=True,
                            help="optimizer type for learning",
                            type=str)

        parser.add_argument("--logs_path",
                            help="path for tensorboard logs",
                            type=str,
                            default=os.environ.get('SM_MODEL_DIR'))

        parser.add_argument("--delimiter",
                            required=True,
                            help="file delimiter for parsing",
                            type=str)

        parser.add_argument("--skip_header",
                            required=True,
                            help="lines to skip in file",
                            type=int)

        args = vars(parser.parse_args())

        return args

