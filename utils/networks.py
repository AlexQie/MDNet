# The implementation of MDNet using TensorFlow
# MDNet: http://cvlab.postech.ac.kr/research/mdnet/
# Modified from the Tensorflow tutorial:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py

import tensorflow as tf
import numpy as np
import re
import scipy.io

FLAGS = tf.app.flags.FLAGS


#tf.app.flags.DEFINE_integer('batch_size', 1024,
#                            """Number of images to process in a batch.""")


PRETRAINED = './models/imagenet-vgg-m-conv1-3.mat'

IMAGE_SIZE = np.array([107, 107])
NUM_CLASSES = 2

TOWER_NAME = 'tower'

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
    )
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _variable_init_from_constant(name, shape, val, wd):
    """initialize from pretrained networks"""
    var = _variable_on_cpu(
        name,
        shape,
        tf.constant_initializer(val, dtype=tf.float32)
    )
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def inference(images, K=None, k=None):
    """
    Build the MDNet model
    """
    # Load conv1, conv2, conv3 from the pre-trained VGG-M network
    def load_vgg_m():
        model = scipy.io.loadmat(PRETRAINED)
        model = model['layers']
        conv1 = model[0, 0][0][0][0][0][0]
        conv2 = model[0, 4][0][0][0][0][0]
        conv3 = model[0, 8][0][0][0][0][0]

        return conv1, conv2, conv3
    conv1, conv2, conv3 = load_vgg_m()

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_init_from_constant(
            'weights',
            shape=[7, 7, 3, 96],
            val=conv1,
            wd=5e-5,
        )
        # VALID means no padding
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    norm1 = tf.nn.lrn(conv1, 5, bias=2, alpha=1e-4, beta=0.75, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_init_from_constant(
            'weights',
            shape=[5, 5, 96, 256],
            val=conv2,
            wd=5e-5,
        )
        conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    norm2 = tf.nn.lrn(conv2, 5, bias=2, alpha=1e-4, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_init_from_constant(
            'weights',
            shape=[3, 3, 256, 512],
            val=conv3,
            wd=5e-5,
        )
        conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv3)

    # fc4
    with tf.variable_scope('fc4') as scope:
        # reshape dim: 3 * 3 * 512 = 4608
        reshape = tf.reshape(conv3, [-1, 4608])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[dim, 512],
                                              stddev=1e-2,
                                              wd=5e-4)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        fc4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(fc4)
    drop4 = tf.nn.dropout(fc4, 0.5, name='drop4')

    # fc5
    with tf.variable_scope('fc5') as scope:
        weights = _variable_with_weight_decay('weights',
                                             shape=[512, 512],
                                             stddev=1e-2,
                                             wd=5e-4)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        fc5 = tf.nn.relu(tf.matmul(drop4, weights) + biases, name=scope.name)
        _activation_summary(fc5)
    drop5 = tf.nn.dropout(fc5, 0.5, name='drop5')

    # fc6
    with tf.variable_scope('fc6') as scope:
        if K is None:
            # only one branches for online tracking
            K = 1
            k = 0

        weights = _variable_with_weight_decay('weights',
                                               shape=[512, 2 * K],
                                               stddev=1e-2,
                                               wd=5e-4)

        # K branches for training
        biases = _variable_on_cpu('biases', [2 * K], tf.constant_initializer(0))
        branch_weight = tf.slice(weights, [0, 2 * k], [512, 2])
        branch_biases = tf.slice(biases, [2 * k], [2])

        fc6 = tf.nn.bias_add(tf.matmul(drop5, branch_weight), branch_biases, name=scope.name)
        #logits = tf.nn.softmax(fc6, name='logits')
        _activation_summary(fc6)

    return fc6

def loss(logits, labels):
    """Add l2loss to all the traninalbe varibles"""
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_average_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw) ', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_average_op

def train(total_loss):
    loss_averages_op = _add_loss_summaries(total_loss)
    learning_rate = tf.constant(0.0001, dtype=tf.float64)
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(learning_rate)
        grads = opt.compute_gradients(total_loss)
        # using different learning rate for conv layers and fc layers
        for i in range(len(grads)):
            grad, var = grads[i]
            if var.name.find('fc') != -1:
                grads[i] = (grad * 10, var)

    apply_gradient_op = opt.apply_gradients(grads)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var ,in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    return apply_gradient_op
