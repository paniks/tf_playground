import tensorflow as tf
import numpy as np
import h5py as h5
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale

from python.CONFIG import *


def load_data(path=''):
    if path == '':
        raise IOError('Path required')
    if not path.endswith('.h5'):
        raise IOError('Only HDF5 files')
    f = h5.File(path)
    return f


def translate_labels(labels: np.ndarray, one_hot=False):
    for key, value in LABELS_TRANSLATOR.items():
        labels[labels == key] = value

    if one_hot:
        labels = labels.astype(int)
        labels = np.eye(N_CLASSES)[labels]
    return labels


def mlp(X):
    h1 = tf.nn.tanh((tf.matmul(X, WEIGHTS['w1'])+BIASES['b1']), name='hiddenLayer1')
    h2 = tf.nn.tanh((tf.matmul(h1, WEIGHTS['w2'])+BIASES['b2']), name='hiddenLayer2')
    h3 = tf.nn.tanh((tf.matmul(h2, WEIGHTS['w3'])+BIASES['b3']), name='hiddenLayer3')
    out = tf.nn.softmax((tf.matmul(h3, WEIGHTS['out']) + BIASES['out']), name='OutputLayer')
    return out


# OPEN AND PREPARE DATA
h5file = load_data(DATA_PATH)
features = h5file['features'][::]
labels = h5file['labels'][::]
ids = h5file['IDs'][::]
h5file.close()

features = scale(features)
labels = translate_labels(labels, True)
train_set, test_set, train_classes, test_classes = train_test_split(features, labels, test_size=0.3, random_state=12345)

# PLACEHOLDERS
# placeholders are type of arrays/vectors/variables that we prepare for learning sets
X = tf.placeholder("float", [None, N_FEATURES], name='features')    # for features batches
Y = tf.placeholder("float", [None, N_CLASSES], name='labels')       # for labels batches
keep_prob = tf.placeholder(tf.float32)                              # for: https://stackoverflow.com/questions/35545798/keep-prob-in-tensorflow-mnist-tutorial

# BUILD MODEL
clf = mlp(X)

# COST FUNCTION
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(clf), reduction_indices=[1]))

# OPTIMIZER
train_step = tf.train.GradientDescentOptimizer(TRAINING_PARAMETERS['training_rate']).minimize(cross_entropy)

# compare predicted value from network with the expected value/target
correct_prediction = tf.equal(tf.argmax(clf, 1), tf.argmax(Y, 1))
# accuracy determination
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

# initialization of all variables
initial = tf.global_variables_initializer()

# add ops to save and restore all the variables.
saver = tf.train.Saver()

# creating a session
with tf.Session() as sess:
    print('===== training started =====')
    sess.run(initial)
    writer = tf.summary.FileWriter("./data_priv/writer")
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

    # training loop over the number of epoches
    for epoch in range(TRAINING_PARAMETERS['training_epochs']):
        for i in range(len(train_set)):

            start = i
            end = i+TRAINING_PARAMETERS['batch_size']
            x_batch = train_set[start:end]
            y_batch = train_classes[start:end]

            # feeding training data_priv/examples
            sess.run(train_step, feed_dict={X: x_batch, Y: y_batch, keep_prob: 0.5})
            i += TRAINING_PARAMETERS['batch_size']

        # feeding testing data_priv to determine model accuracy
        y_pred = sess.run(tf.argmax(clf, 1), feed_dict={X: test_set, keep_prob: 1.0})
        y_true = sess.run(tf.argmax(test_classes, 1))
        acc = sess.run(accuracy, feed_dict={X: test_set, Y: test_classes, keep_prob: 1.0})
        # print accuracy for each epoch
        print('epoch {}, accuracy {}'.format(epoch, acc))
        print('---------------')

    print('==== training ended ====')
    print('...')
    print('==== saving model ====')
    print('...')
    save_path = saver.save(sess, MODEL_NAME)
    print("Model saved in path: %s \n" % save_path)
    print('==== results ====')
    print(classification_report(y_true, y_pred))
