import os
import random
import copy

import numpy as np
from numpy.matlib import repmat
from scipy.misc import imread
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy import misc

DATASET_PATH = "/home/qiechunguang/datasets"
SEQ_LIST_PATH = "./seqList"

POS_PER_FRAME = 50
NEG_PER_FRAME = 200
SCALE_FACTOR = 1.05

#DEBUG = True
DEBUG = False

class Input():
    """A batch generate, generate batch by calling next_batch"""
    def __init__(self, num_cycles, batch_frames, pos_num, neg_num, batch_size):
        self._roidb = generate_roidb()
        self.seq_list = sorted(self._roidb)
        self.num_cycles = num_cycles
        self.batch_frames = batch_frames
        self.pos_num = pos_num
        self.neg_num = neg_num
        self.batch_size = batch_size
        # tic for global step
        self.num_batch = 0
        self.seq_num = len(self._roidb)
        self.total_batch_num = self.num_cycles * self.seq_num
        nFrames = num_cycles * batch_frames
        # Prepare batches for each iteration
        for seq in self.seq_list:
            new_data_list = []
            while len(new_data_list) < nFrames:
                temp_data_list = copy.deepcopy(self._roidb[seq])
                random.shuffle(temp_data_list)
                new_data_list += temp_data_list
            self._roidb[seq] = new_data_list[:nFrames]

    def _generate_data(self, data_list):
        batch_per_frame = self.pos_num + self.neg_num
        batches = np.zeros((self.batch_frames * batch_per_frame, self.batch_size, self.batch_size, 3))
        boxes = np.zeros((self.batch_frames * batch_per_frame, 4), dtype=np.int64)
        labels = np.zeros(self.batch_frames * batch_per_frame)
        for i in range(len(data_list)):
            im = misc.imread(data_list[i][0])
            # random select positive boxes and negative boxes
            positive_idx = np.random.choice(np.arange(POS_PER_FRAME), self.pos_num)
            negative_idx = np.random.choice(np.arange(NEG_PER_FRAME), self.neg_num)
            boxes[i*batch_per_frame:i*batch_per_frame+self.pos_num, :] = data_list[i][1][positive_idx, :]
            boxes[i*batch_per_frame+self.pos_num:(i+1)*batch_per_frame, :] = data_list[i][2][negative_idx, :]

            for batch_index in range(batch_per_frame):
                idx = i*batch_per_frame+batch_index
                mini_batch = im[boxes[idx][1]:boxes[idx][1]+boxes[idx][3], boxes[idx][0]:boxes[idx][0]+boxes[idx][2]]
                mini_batch = misc.imresize(mini_batch, (self.batch_size, self.batch_size))
                batches[idx, :, :, :] = mini_batch
            labels[i*batch_per_frame:i*batch_per_frame+self.pos_num] = 1
            labels[i*batch_per_frame+self.pos_num:(i+1)*batch_per_frame] = 0

        return batches, labels

    def next_batch(self):
        if self.num_batch == self.total_batch_num:
            raise ValueError("no next batches.", self.num_batch)
        seq_name = self.seq_list[self.num_batch % len(self.seq_list)]
        data_index = self.num_batch // self.num_cycles
        data_list = self._roidb[seq_name][data_index*self.batch_frames:(data_index+1)*self.batch_frames]
        self.num_batch += 1
        return self._generate_data(data_list)

def generate_data(im, boxes, batch_size):
    boxes = boxes.astype(np.int32)
    batch_n = boxes.shape[0]
    batches = np.zeros((batch_n, batch_size, batch_size, 3), dtype=np.float32)
    #im = misc.imread(im_path)
    for i in range(batch_n):
        mini_batch = im[boxes[i][1]:boxes[i][1]+boxes[i][3], boxes[i][0]:boxes[i][0]+boxes[i][2]]
        mini_batch = misc.imresize(mini_batch, (batch_size, batch_size))
        batches[i, :, :, :] = mini_batch

    return batches

def generate_roidb():
    sequence_list = []

    # load vot13
    with open(os.path.join(SEQ_LIST_PATH, "vot13-otb.txt")) as f:
        sequence_list += [os.path.join(DATASET_PATH, "vot2013", seq)
                          for seq in f.read().split("\n") if seq != ""]
    # load vot14
    with open(os.path.join(SEQ_LIST_PATH, "vot14-otb.txt")) as f:
        sequence_list += [os.path.join(DATASET_PATH, "vot2014", seq)
                          for seq in f.read().split("\n") if seq != ""]
    # load vot15
    with open(os.path.join(SEQ_LIST_PATH, "vot15-otb.txt")) as f:
        sequence_list += [os.path.join(DATASET_PATH, "vot2015", seq)
                          for seq in f.read().split("\n") if seq != ""]
    print(sequence_list)

    roidb = {}
    for seq in sequence_list:
        roidb[seq] = sample_rios(seq)

    return roidb

def sample_rios(seq):
    print("Loading", seq)
    image_list = [os.path.join(seq, img) for img in sorted(os.listdir(seq)) if img[-3:] == 'jpg']
    gt = np.loadtxt(os.path.join(seq, "groundtruth.txt"), delimiter=',')
    _roidb = []

    r, c = gt.shape
    if c >= 6:
        x, y = gt[:, np.arange(0, c, 2)], gt[:, np.arange(1, c, 2)]
        new_gt = np.zeros((gt.shape[0], 4))
        new_gt[:, 0] = x.min(axis=1)
        new_gt[:, 1] = y.min(axis=1)
        new_gt[:, 2] = x.max(axis=1) - x.min(axis=1)
        new_gt[:, 3] = y.max(axis=1) - y.min(axis=1)
        gt = new_gt
    im = imread(image_list[0])

    if len(image_list) > gt.shape[0]:
        image_list = image_list[:gt.shape[0]]

    # Draw samples based on the groundtruth bounding-box
    for i in range(len(image_list)):
        pos_sample = np.zeros((POS_PER_FRAME, 4))
        current_size = 0
        while current_size < pos_sample.shape[0]:
            pos = generate_samples(gt[i, :], POS_PER_FRAME, im.shape, SCALE_FACTOR, 0.1, 5, False)
            r = overlap_ratio(pos, gt[i, :])
            pos = pos[r >= 0.7, :]
            if pos.shape[0] == 0:
                continue
            pos = pos[np.random.choice(np.arange(pos.shape[0]), min(pos.shape[0], POS_PER_FRAME - current_size)), :]
            pos_sample[current_size:current_size + pos.shape[0], :] = pos
            current_size += pos.shape[0]
            #print(current_size)
        current_size = 0
        neg_sample = np.zeros((NEG_PER_FRAME, 4))
        while current_size < neg_sample.shape[0]:
            neg = generate_samples(gt[i, :], NEG_PER_FRAME, im.shape, SCALE_FACTOR, 2, 10, True)
            r = overlap_ratio(neg, gt[i, :])
            #print(r)
            #input()
            neg = neg[r <= 0.5, :]
            if neg.shape[0] == 0:
                continue
            neg = neg[np.random.choice(np.arange(neg.shape[0]), min(neg.shape[0], NEG_PER_FRAME - current_size)), :]
            neg_sample[current_size:current_size + neg.shape[0], :] = neg
            current_size += neg.shape[0]

        if DEBUG is True:
            draw_rect(im, pos_sample, neg_sample)
            input()
        _roidb.append((image_list[i], pos_sample, neg_sample))

    return _roidb

def draw_rect(im, rect_patches1, rect_patches2=None):
    pos_patches, neg_patches = [], []
    ax = plt.gca()
    for i in range(rect_patches1.shape[0]):
        x, y, w, h = rect_patches1[i, 0], rect_patches1[i, 1], rect_patches1[i, 2], rect_patches1[i, 3]
        pos_patches.append(Rectangle((x, y), w, h, fill=False, edgecolor='g'))
    ax.add_collection(PatchCollection(pos_patches, match_original=True))
    if rect_patches2 is not None:
        for i in range(rect_patches2.shape[0]):
            x, y, w, h = rect_patches2[i, 0], rect_patches2[i, 1], rect_patches2[i, 2], rect_patches2[i, 3]
            neg_patches.append(Rectangle((x, y), w, h, fill=False, edgecolor='r'))
        ax.add_collection(PatchCollection(neg_patches, match_original=True))
    plt.imshow(im)
    plt.show()

def generate_samples(bbox, n, image_sz, scale_factor, trans_range, scale_range, valid):
    """Generate positive or negative samples based on the given parameters"""
    h, w, _ = image_sz
    sample = np.zeros_like(bbox)
    sample[0] = bbox[0] + bbox[2] / 2
    sample[1] = bbox[1] + bbox[3] / 2
    sample[2] = bbox[2]
    sample[3] = bbox[3]
    samples = repmat(sample, n, 1)

    samples[:, 0:2] += trans_range * samples[:, 2:4] * (np.random.rand(n, 2) * 2 - 1)
    samples[:, 2:4] *= scale_factor ** ((np.random.rand(n, 1) * 2 - 1) * scale_range)

    samples[:, 2] = np.maximum(5, np.minimum(w - 5, samples[:, 2]))
    samples[:, 3] = np.maximum(5, np.minimum(h - 5, samples[:, 3]))

    samples[:, 0] -= samples[:, 2] / 2
    samples[:, 1] -= samples[:, 3] / 2

    if valid is True:
        samples[:, 0] = np.maximum(1, np.minimum(w - samples[:, 2], samples[:, 0]))
        samples[:, 1] = np.maximum(1, np.minimum(h - samples[:, 3], samples[:, 1]))
    else:
        samples[:, 0] = np.maximum(1 - samples[:, 0] / 2, np.minimum(w - samples[:, 2] / 2, samples[:, 0]))
        samples[:, 1] = np.maximum(1 - samples[:, 1] / 2, np.minimum(h - samples[:, 3] / 2, samples[:, 1]))

    samples = np.round(samples)

    return samples

def intsection_area(rect1, rect2):
    """compute overlap size of two rectangle"""
    rect1_x1, rect1_x2 = rect1[:, 0], rect1[:, 0] + rect1[:, 2]
    rect1_y1, rect1_y2 = rect1[:, 1], rect1[:, 1] + rect1[:, 3]
    rect2_x1, rect2_x2 = rect2[0], rect2[0] + rect2[2]
    rect2_y1, rect2_y2 = rect2[1], rect2[1] + rect2[3]
    int_x1 = np.minimum(np.maximum(rect1_x1, rect1_x2), np.maximum(rect2_x1, rect2_x2))
    int_x2 = np.maximum(np.minimum(rect1_x1, rect1_x2), np.minimum(rect2_x1, rect2_x2))
    int_y1 = np.minimum(np.maximum(rect1_y1, rect1_y2), np.maximum(rect2_y1, rect2_y2))
    int_y2 = np.maximum(np.minimum(rect1_y1, rect1_y2), np.minimum(rect2_y1, rect2_y2))

    intsect = (int_x1 - int_x2) * (int_y1 - int_y2)

    return intsect

def overlap_ratio(rect1, rect2):
    """compute overlap ratio of two rectangle"""
    inter_area = intsection_area(rect1, rect2)
    union_area = rect1[:, 2] * rect1[:, 3] + rect2[2] * rect2[3] - inter_area

    return inter_area / union_area

def test():
    rect1 = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 0.5, 0.5]])
    rect2 = np.array([0.5, 0.5, 1, 1])
    correct_sect = np.array([0.25, 1, 0])
    intsect = intsection_area(rect1, rect2)
    assert np.allclose(intsect, correct_sect), "intsection_area error"

if __name__ == "__main__":
    data = Input(100, 8, 32, 96, 117)

    batch, label = data.next_batch()

    for i in range(0, batch.shape[0], 20):
        print(label[i])
        plt.imshow(batch[i, :, :, :])
        plt.show()
        input()