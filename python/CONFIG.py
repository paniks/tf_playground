import tensorflow as tf
import numpy as np

DATA_PATH = './data/features.h5'
MODEL_NAME = './data/mlp_python_model'

N_FEATURES = 9     # number of features
N_CLASSES = 3      # number of classes

TRAINING_PARAMETERS = {
    'train_size':       0.7,     # size of training set
    'training_rate':    0.01,    # "step" of gradient descent
    'training_epochs':  25,      # how many times entire training dataset is passed forward neural net
    'batch_size':       200      # number of samples that will be propagated through the network.

    # nice explanation: https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
}


NETWORK_PARAMETERS = {
    'hidden_1':  32,     # number of perceptrons in hidden layer 1
    'hidden_2':  16,     # number of perceptrons in hidden layer 2
    'hidden_3':  8,      # number of perceptrons in hidden layer 3
}

# Store layers weight
WEIGHTS = {
    'w1':  tf.Variable(tf.random_normal([N_FEATURES, NETWORK_PARAMETERS['hidden_1']],
                                        mean=0, stddev=1 / np.sqrt(N_FEATURES)), name='weights1'),
    'w2':  tf.Variable(tf.random_normal([NETWORK_PARAMETERS['hidden_1'], NETWORK_PARAMETERS['hidden_2']],
                                        mean=0, stddev=1 / np.sqrt(N_FEATURES)), name='weights2'),
    'w3':  tf.Variable(tf.random_normal([NETWORK_PARAMETERS['hidden_2'], NETWORK_PARAMETERS['hidden_3']],
                                        mean=0, stddev=1 / np.sqrt(N_FEATURES)), name='weights3'),
    'out': tf.Variable(tf.random_normal([NETWORK_PARAMETERS['hidden_3'], N_CLASSES],
                                        mean=0, stddev=1 / np.sqrt(N_FEATURES)), name='weightsOut'),
}

# Store layers biases
BIASES = {
    'b1':  tf.Variable(tf.truncated_normal([NETWORK_PARAMETERS['hidden_1']],
                                           mean=0, stddev=1 / np.sqrt(N_FEATURES)), name='biases1'),
    'b2':  tf.Variable(tf.truncated_normal([NETWORK_PARAMETERS['hidden_2']],
                                           mean=0, stddev=1 / np.sqrt(N_FEATURES)), name='biases2'),
    'b3':  tf.Variable(tf.truncated_normal([NETWORK_PARAMETERS['hidden_3']],
                                           mean=0, stddev=1 / np.sqrt(N_FEATURES)), name='biases3'),
    'out': tf.Variable(tf.truncated_normal([N_CLASSES],
                                           mean=0, stddev=1 / np.sqrt(N_FEATURES)), name='biasesOut'),
}


LABELS_TRANSLATOR = {
    'N': 0,
    'V': 1,
    'S': 2
}
