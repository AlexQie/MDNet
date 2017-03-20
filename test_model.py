import tensorflow as tf
from scipy.misc import imread
from tracking import init_networks
from tracking import *
from utils.generate_samples import generate_samples

def test_pos_neg_sample_score():
    #impath = '/home/qiechunguang/Desktop/tracking/datasets/vot2013/bicycle/00000001.jpg'
    impath = '/home/qiechunguang/datasets/vot2013/cup/00000001.jpg'
    im = imread(impath)
    sess = tf.Session()
    images_tensor, conv3_feature, classify_input, label_input, logits_op, train_step, cross_entropy_mean = init_networks(sess)
    #np.loadtxt('/home/qiechunguang/datasets/vot2013/cup/groundtruth.txt')
    gt = np.loadtxt('/home/qiechunguang/datasets/vot2013/cup/groundtruth.txt', delimiter=',')
    region = gt[0, :]
    pos_examples = generate_samples('uniform_aspect', im.shape, gt[0, :], BBREG_N_SAMPLES * 10, SCALE_FACTOR, 0.3, 10)

    print(pos_examples.shape, region.shape)
    r = overlap_ratio(pos_examples, region)
    pos_examples = pos_examples[r > INIT_POS_OVERLAP_THRE, :]
    pos_idx = np.random.choice(np.arange(pos_examples.shape[0]), NPOS_INIT)
    pos_examples = pos_examples[pos_idx, :]
    pos_data = data_input.generate_data(im, pos_examples, 107)
    draw_rect(im, pos_examples)
    logits = tf.get_default_graph().get_tensor_by_name('fc6/logits:0')
    l = sess.run(logits, {'images:0':pos_data, 'step:0': 0})
    print(l)

if __name__  == '__main__':
    test_pos_neg_sample_score()