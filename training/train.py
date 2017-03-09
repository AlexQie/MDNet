import time

import tensorflow as tf
import scipy.io

import data_input
import networks

NUM_CYCLES = 100
BATCH_FRAMES = 8
POS_PER_FRAME = 32
NEG_PER_FRAME = 96
BATCH_SIZE = 107

LOG_DIR = '/home/qiechunguang/tmp'
PRETRAINED = '../models/imagenet-vgg-m-conv1-3.mat'

def placeholder_inputs():
    batch_images = BATCH_FRAMES * (POS_PER_FRAME + NEG_PER_FRAME)
    images_placeholder = tf.placeholder(
        tf.float32, shape=(batch_images, BATCH_SIZE, BATCH_SIZE, 3))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_images))

    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch()
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed
    }

    return feed_dict

def load_vgg_m():
    model = scipy.io.loadmat(PRETRAINED)
    model = model['layers']
    conv1 = model[0, 0][0][0][0][0][0]
    conv2 = model[0, 4][0][0][0][0][0]
    conv3 = model[0, 8][0][0][0][0][0]

    return conv1, conv2, conv3

def run_training():
    data_set = data_input.Input(NUM_CYCLES,
                                BATCH_FRAMES,
                                POS_PER_FRAME,
                                NEG_PER_FRAME,BATCH_SIZE)
    K = data_set.seq_num

    conv1, conv2, conv3 = load_vgg_m()
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs()

        logits = networks.inference(images_placeholder, conv1, conv2, conv3)
        loss = networks.loss(logits, labels_placeholder)
        train_op = networks.train(loss)
        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        sess.run(init)

        for step in range(NUM_CYCLES * K):
            start_time = time.time()

            feed_dict = fill_feed_dict(data_set,
                                       images_placeholder,
                                       labels_placeholder)
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time
            print('Step %d: loss = %.5f (%.3f sec)' % (step, loss_value, duration))
            if step % 100 == 0:

                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

def main(_):
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    run_training()

if __name__ == '__main__':
    tf.app.run(main=main)