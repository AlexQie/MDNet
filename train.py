import time
import os

import tensorflow as tf
import scipy.io

from utils import data_input, networks


NUM_CYCLES = 100
BATCH_FRAMES = 8
POS_PER_FRAME = 32
NEG_PER_FRAME = 96
BATCH_SIZE = 107

LOG_DIR = '/home/qiechunguang/tmp'


def placeholder_inputs():
    images_placeholder = tf.placeholder(tf.float32,
                                        shape=(None, 107, 107, 3), name='images')
    labels_placeholder = tf.placeholder(tf.int32, name='label')
    step_placeholder = tf.placeholder(tf.int32, name='step')

    return images_placeholder, labels_placeholder, step_placeholder

def fill_feed_dict(data_set, step, images_pl, labels_pl, step_pl):
    images_feed, labels_feed = data_set.next_batch()
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        step_pl: step
    }

    return feed_dict

def run_training():
    data_set = data_input.Input(NUM_CYCLES,
                                BATCH_FRAMES,
                                POS_PER_FRAME,
                                NEG_PER_FRAME,BATCH_SIZE)
    K = data_set.seq_num

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder, step_placeholder = placeholder_inputs()

        logits = networks.inference(images_placeholder, K, step_placeholder)
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

            feed_dict = fill_feed_dict(data_set, step % K,
                                       images_placeholder,
                                       labels_placeholder,
                                       step_placeholder)
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time
            print('Global step %d, step %d on seq %s : loss = %.5f (%.3f sec)' %
                  (step, step / K, data_set.seq_list[step % K].split('/')[-1], loss_value, duration))
            if (step + 1) % K == 0:
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

def main(_):
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    run_training()

if __name__ == '__main__':
    tf.app.run(main=main)
