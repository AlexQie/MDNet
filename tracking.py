import os

import tensorflow as tf
from scipy.misc import imread
import numpy as np
from utils.generate_samples import generate_samples
from utils.data_input import overlap_ratio, draw_rect
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from utils import networks, data_input


SCALE_FACTOR = 1.05

TRANS_RANGE = 0.6
SCALE_RANGE = 1

BATCHSIZE_TEST = 256
BBREG_N_SAMPLES = 1000
BATCH_SIZE = 107

NPOS_INIT = 500
NNEG_INIT = 5000
INIT_POS_OVERLAP_THRE = 0.7
INIT_NEG_OVERLAP_THRE = 0.5

NPOS_UPDATE = 50
NNEG_UPDATE = 200
UPDATE_POS_OVERLAP_THRE = 0.7
UPDATE_NEG_OVERLAP_THRE = 0.3

BATCH_SIZE_HNM = 256
BATCH_ACC_HNM = 4

BATCH_POS = 32
BATCH_NEG = 96
HNM_ITER = 30

N_SAMPLES = 256

N_FRAMES_LONG = 100
N_FRAMES_SHORT = 20

UPDATE_INTERVAL = 10

def extract_conv3_feature(sess, im, boxes, conv3_feature, images_tensor):
    print('Extract conv features')
    batch_im = data_input.generate_data(im, boxes, 107)
    bt_size = batch_im.shape[0]
    n_iter = bt_size // BATCHSIZE_TEST
    #shape of conv3 is 3 * 2 * 512
    batch = np.zeros((bt_size, 3, 3, 512))
    for i in range(n_iter):
        current_end = min((i + 1) * BATCHSIZE_TEST, bt_size)
        current_batch_im = batch_im[i * BATCHSIZE_TEST:current_end, :, :, :]
        batch[i * BATCHSIZE_TEST:current_end, :, :, :] = sess.run(conv3_feature, feed_dict = {images_tensor: current_batch_im})
    return batch

def load_model(sess):
    """Load the pre-trained networks"""
    saver = tf.train.import_meta_graph('/home/qiechunguang/data/model.ckpt-5799.meta')
    saver.restore(sess, '/home/qiechunguang/data/model.ckpt-5799')
    fc4_weight, fc4_biase, fc5_weight, fc5_biase = None, None, None, None
    varibles = tf.global_variables()
    for var in varibles:
        if var.name == 'fc4/weights:0':
            fc4_weight = var
        elif var.name == 'fc4/biases:0':
            fc4_biase = var
        elif var.name == 'fc5/weights:0':
            fc5_weight = var
        elif var.name == 'fc5/biases:0':
            fc5_biase = var
    fc4_weight, fc4_biase, fc5_weight, fc5_biase = sess.run([fc4_weight, fc4_biase, fc5_weight, fc5_biase])

    return fc4_weight, fc4_biase, fc5_weight, fc5_biase


def init_networks(sess):
    """Load the pre-trained networks, then replace the fc6"""
    fc4_weight, fc4_biase, fc5_weight, fc5_biase = load_model(sess)
    conv3_feature = tf.get_default_graph().get_tensor_by_name('conv3/conv3:0')
    image_tensor = tf.get_default_graph().get_tensor_by_name('images:0')
    # Add new fully connected layers for online training
    with tf.variable_scope('input'):
        conv3_input = tf.placeholder(
            tf.float32, shape=[None, 3, 3, 512], name='conv3_input'
        )
        label_input = tf.placeholder(tf.int32, name='label')

    with tf.variable_scope('online_fc4'):
        reshaped = tf.reshape(conv3_input, [-1, 4608])
        weights = networks._variable_init_from_constant('weights',
                                                        shape=[4608, 512],
                                                        val=fc4_weight,
                                                        wd=5e-4)
        biases = networks._variable_on_cpu('biases', shape=[512], initializer=tf.constant_initializer(fc4_biase))
        online_fc4 = tf.nn.relu(tf.matmul(reshaped, weights) + biases)
    drop4 = tf.nn.dropout(online_fc4, 0.5)

    with tf.variable_scope('online_fc5'):
        weights = networks._variable_init_from_constant('weights',
                                                        shape=[512, 512],
                                                        val=fc5_weight,
                                                        wd=5e-4)
        biases = networks._variable_on_cpu('biases', shape=[512], initializer=tf.constant_initializer(fc5_biase))
        online_fc5 = tf.nn.relu(tf.matmul(drop4, weights) + biases)
    drop5 = tf.nn.dropout(online_fc5, 0.5)

    with tf.variable_scope('online_fc6'):
        weights = networks._variable_with_weight_decay('weights',
                                                       shape=[512, 2],
                                                       stddev=1e-2,
                                                       wd=5e-4)

        biases = networks._variable_on_cpu('biases', shape=[2], initializer=tf.constant_initializer(0))
        online_fc6 = tf.nn.relu(tf.matmul(drop5, weights) + biases)
    logits = tf.nn.softmax(online_fc6, name='online_logits')

    with tf.variable_scope('online_cross_entropy'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_input, logits=logits
        )
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

    with tf.variable_scope('train'):
        opt = tf.train.AdamOptimizer(0.0001)

        grads = opt.compute_gradients(cross_entropy_mean)
        for i in range(len(grads)):
            grad, var = grads[i]
            if var.name.find('online_fc6') != -1:
                grads[i] = (grad * 10, var)
        apply_grad_op = opt.apply_gradients(grads)

    return image_tensor, conv3_feature, conv3_input, label_input, logits, apply_grad_op, cross_entropy_mean

def tracking(images_list, region):
    sess = tf.Session()

    num_frames = len(images_list)

    im = imread(images_list[0])
    result = np.zeros((num_frames, 4))

    # Bounding box regression
    pos_examples = generate_samples('uniform_aspect',
                                    im.shape,
                                    region,
                                    BBREG_N_SAMPLES * 10,
                                    SCALE_FACTOR, 0.3, 10)
    #draw_rect(im, pos_examples)
    r = overlap_ratio(pos_examples, region)
    pos_examples = pos_examples[r > 0.6, :]
    indices = np.random.choice(np.arange(pos_examples.shape[0]), BBREG_N_SAMPLES)
    pos_examples = pos_examples[indices, :]

    images_tensor, conv3_feature, classify_input, label_input, logits_op, train_step, cross_entropy_mean = init_networks(sess)

    # Draw positive and negative samples
    pos_examples = generate_samples('gaussian',
                                    im.shape,
                                    region,
                                    NPOS_INIT * 2,
                                    SCALE_FACTOR, 0.1, 5)
    r = overlap_ratio(pos_examples, region)
    pos_examples = pos_examples[r > INIT_POS_OVERLAP_THRE, :]
    pos_idx = np.random.choice(np.arange(pos_examples.shape[0]), NPOS_INIT)
    pos_examples = pos_examples[pos_idx, :]

    neg_example_uniform = generate_samples('uniform',
                                           im.shape,
                                           region,
                                           NNEG_INIT,
                                           SCALE_FACTOR, 1, 10)
    neg_example_whole = generate_samples('whole',
                                         im.shape,
                                         region,
                                         NNEG_INIT,
                                         SCALE_FACTOR)
    neg_examples = np.concatenate((neg_example_uniform, neg_example_whole), axis=0)
    r = overlap_ratio(neg_examples, region)
    neg_examples = neg_examples[r < INIT_NEG_OVERLAP_THRE, :]
    neg_idx = np.random.choice(np.arange(neg_examples.shape[0]), NNEG_INIT)
    neg_examples = neg_examples[neg_idx, :]

    examples = np.concatenate((pos_examples, neg_examples), axis=0)

    conv3_features = extract_conv3_feature(sess, im, examples, conv3_feature, images_tensor)
    labels = np.zeros((NNEG_INIT + NPOS_INIT), dtype=np.int32)
    labels[0:NPOS_INIT] = 1

    sess.run(tf.global_variables_initializer())

    mdnet_finetune_hnm(sess, conv3_features[labels==1, :, :, :], conv3_features[labels==0, :, :, :],
                       classify_input, label_input, logits_op, train_step)

    # Prepare training data for online update
    neg_examples = generate_samples('uniform', im.shape, region, NNEG_UPDATE*2, SCALE_FACTOR, 2, 5)
    r = overlap_ratio(neg_examples, region)
    neg_examples = neg_examples[r < INIT_NEG_OVERLAP_THRE, :]
    neg_idx = np.random.choice(np.arange(neg_examples.shape[0]), NNEG_UPDATE)
    neg_examples = neg_examples[neg_idx, :]
    examples = np.concatenate((pos_examples, neg_examples))
    conv3_features = extract_conv3_feature(sess, im, examples, conv3_feature, images_tensor)
    labels = np.zeros((conv3_features.shape[0]), dtype=np.int32)
    labels[0:pos_examples.shape[0]] = 1

    # data for online updating
    total_pos_data = [conv3_features[labels == 1, :, :, :]]
    total_neg_data = [conv3_features[labels == 0, :, :, :]]
    success_frame = [0]

    trans_range = TRANS_RANGE
    scale_range = SCALE_RANGE
    print(region)

    # plot
    #plt.ion()
    #fig = plt.figure(0)
    ax = plt.gca()
    x, y, w, h = region[0], region[1], region[2], region[3]
    rectangle = Rectangle((x, y), w, h, fill=False, edgecolor='g')
    show_tracking(im, region, ax)


    for i in range(1, num_frames):
        print('Processing frame', i)
        im = imread(images_list[i])
        samples = generate_samples('gaussian', im.shape, region, N_SAMPLES, SCALE_FACTOR, trans_range, scale_range)
        #draw_rect(im, samples)
        conv_feat = extract_conv3_feature(sess, im, samples, conv3_feature, images_tensor)
        logits = sess.run(logits_op, feed_dict={classify_input: conv_feat})
        logits = logits[:, 1]
        #print(logits)
        sorted_index = np.argsort(logits)[::-1]
        target_score = np.mean(logits[sorted_index[0:5]])
        target_location = np.mean(samples[sorted_index[0:5]], axis=0)
        #neg_batch = batch[np.argsort(score_hneg)[::-1][0:BATCH_NEG], :, :, :]
        result[i, :] = target_location
        #print(target_score)
        if target_score < 0.4:
            trans_range = min(1.5, 1.1 * TRANS_RANGE)
        else:
            trans_range = TRANS_RANGE

        if target_score >= 0.4:
            pos_examples = generate_samples('gaussian', im.shape, region, NPOS_UPDATE * 2, SCALE_FACTOR, 0.1, 5)
            r = overlap_ratio(pos_examples, region)
            pos_examples = pos_examples[r > UPDATE_POS_OVERLAP_THRE]
            pos_idx = np.random.choice(np.arange(pos_examples.shape[0]), NPOS_UPDATE)
            pos_examples = pos_examples[pos_idx, :]

            neg_examples = generate_samples('uniform', im.shape, region, NNEG_UPDATE * 2, SCALE_FACTOR, 2, 5)
            r = overlap_ratio(neg_examples, region)
            neg_examples = neg_examples[r < UPDATE_NEG_OVERLAP_THRE]
            neg_idx = np.random.choice(np.arange(neg_examples.shape[0]), NNEG_UPDATE)
            neg_examples = neg_examples[neg_idx]

            examples = np.concatenate((pos_examples, neg_examples))

            feat_conv = extract_conv3_feature(sess, im, examples, conv3_feature, images_tensor)

            #total_pos_data = np.concatenate((total_pos_data, feat_conv[0:NPOS_UPDATE]))
            #total_neg_data = np.concatenate((total_neg_data, feat_conv[NPOS_UPDATE:]))
            total_pos_data.append(feat_conv[0:NPOS_UPDATE])
            total_neg_data.append(feat_conv[NPOS_UPDATE:])

            success_frame.append(i)

            if len(success_frame) > N_FRAMES_LONG:
                total_pos_data = total_pos_data[1:]
            if len(success_frame) > N_FRAMES_SHORT:
                total_neg_data = total_neg_data[1:]

        print(len(success_frame), len(total_pos_data), len(total_neg_data))

        if (i % UPDATE_INTERVAL == 0 or target_score < 0.4) and i < num_frames - 1:
            print('training...')
            if target_score < 0.4:
                pos_idx = max(0, len(total_pos_data) - N_FRAMES_SHORT)
            else:
                pos_idx = max(0, len(total_pos_data) - N_FRAMES_LONG)
            neg_idx = max(0, len(total_pos_data) - N_FRAMES_LONG)
            pos_data = np.concatenate(total_pos_data[pos_idx:])
            print(neg_idx)
            neg_data = np.concatenate(total_neg_data[neg_idx:])

            mdnet_finetune_hnm(sess, pos_data, neg_data, classify_input, label_input, logits_op, train_step)

        # Display
        #print(result[i, :])
        show_tracking(im, result[i, :], ax)

def show_tracking(im, rect, ax):
    ax.cla()
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    rectangle = Rectangle((x, y), w, h, fill=False, edgecolor='g')
    ax.add_patch(rectangle)
    plt.imshow(im)
    plt.pause(0.01)
    plt.draw()

def mdnet_finetune_hnm(sess, pos_samples, neg_samples, classify_input, label_input, logits_op, train_step):
    """
    :argument
        pos_samples: positive bounding boxes extracted from images
        neg_samples: negative boudning boxes extracted from images
        logits_op: the Tensoflow Op to compute logits of each samples
    """
    print('mdnet_fineture_hnm...')
    n_pos, n_neg = pos_samples.shape[0], neg_samples.shape[0]
    train_pos_cnt, train_neg_cnt = 0, 0

    # Extract positive batches
    remain = BATCH_POS * HNM_ITER
    train_pos_list = np.random.permutation(n_pos)
    train_pos = np.array([], dtype=np.int32)
    while remain > 0:
        end_idx = min(n_pos, train_pos_cnt + remain)
        train_pos = np.concatenate((train_pos, train_pos_list[train_pos_cnt:end_idx]), axis=0)
        train_pos_cnt = min(n_pos, train_pos_cnt + remain)
        train_pos_cnt %= n_pos
        remain = BATCH_POS * HNM_ITER - train_pos.shape[0]

    # Extract negative batches
    remain = BATCH_SIZE_HNM * BATCH_ACC_HNM * HNM_ITER
    train_neg_list = np.random.permutation(n_neg)
    train_neg = np.array([], dtype=np.int32)
    while remain > 0:
        end_idx = min(n_neg, train_neg_cnt + remain)
        train_neg = np.concatenate((train_neg, train_neg_list[train_neg_cnt:end_idx]), axis=0)
        train_neg_cnt = min(n_neg, train_neg_cnt + remain)
        train_neg_cnt %= n_neg
        remain = BATCH_SIZE_HNM * BATCH_ACC_HNM * HNM_ITER - train_neg.shape[0]

    # Traning iteration
    batch_per_iter = BATCH_SIZE_HNM * BATCH_ACC_HNM
    for t in range(HNM_ITER):
        # Hard negative mining
        score_hneg = np.zeros(batch_per_iter)
        hneg_start = BATCH_SIZE_HNM * BATCH_ACC_HNM * t
        batch = neg_samples[train_neg[hneg_start:(t+1)*batch_per_iter], :, :, :]
        for h in range(BATCH_ACC_HNM):
            acc_batch = batch[h*BATCH_SIZE_HNM:(h+1)*BATCH_SIZE_HNM, :]
            neg_logits = sess.run(logits_op, feed_dict={classify_input: acc_batch})
            score_hneg[h*BATCH_SIZE_HNM:(h+1)*BATCH_SIZE_HNM] = neg_logits[:, 1]

        hneg_index = np.argsort(score_hneg)[::-1][0:BATCH_NEG]
        #print(score_hneg[hneg_index])
        #input()
        neg_batch = batch[hneg_index, :, :, :]
        pos_batch = pos_samples[train_pos[BATCH_POS*t:BATCH_POS*(t+1)], :, :, :]
        batches = np.concatenate((pos_batch, neg_batch), axis=0)
        labels = np.zeros(batches.shape[0], dtype=np.int32)
        labels[0:BATCH_POS] = 1
        sess.run(train_step, feed_dict={classify_input: batches, label_input: labels})


if __name__ == '__main__':
    im_path = '/home/qiechunguang/datasets/vot2013/bolt/'
    im_list = [os.path.join(im_path, img)
               for img in sorted(os.listdir(im_path)) if img[-3:] == 'jpg']
    gt = np.loadtxt(im_path + 'groundtruth.txt', delimiter=',')
    tracking(im_list, gt[0, :])
