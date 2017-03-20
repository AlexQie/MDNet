import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy.misc import imread
from utils.data_input import draw_rect

def generate_samples(gtype, imsize, bbox, num_samples, scale_factor=None, trans_factor=None, scale_range=None):
    """
    gtype:
    'gaussian'          generate samples from a Gaussian distribution centered at bb
                       -> positive samples, target candidates
    'uniform'           generate samples from a uniform distribution around bb
                       -> negative samples
    'uniform_aspect'    generate samples from a uniform distribution around bb with varying aspect ratios
                       -> training samples for bbox regression
    'whole'             generate samples from the whole image
                       -> negative samples at the initial frame
    """

    h, w, _ = imsize

    sample = np.zeros_like(bbox)
    sample[0] = bbox[0] + bbox[2] / 2
    sample[1] = bbox[1] + bbox[3] / 2
    sample[2] = bbox[2]
    sample[3] = bbox[3]

    samples = repmat(sample, num_samples, 1)

    if gtype == 'gaussian':
        trans_var = np.round(bbox[2:4].mean()) * np.maximum(-1, np.minimum(1, 0.5*np.random.randn(num_samples, 2)))
        samples[:, 0:2] += trans_factor * trans_var
        scale_var = scale_range * np.maximum(-1, np.minimum(1, 0.5*np.random.randn(num_samples, 1)))
        samples[:, 2:4] *= scale_factor ** scale_var
    elif gtype == 'uniform':
        trans_var = np.round(bbox[2:4].mean()) * (2*np.random.rand(num_samples, 2) - 1)
        samples[:, 0:2] += trans_factor * trans_var
        scale_var = scale_range * (np.random.rand(num_samples, 1) * 2 - 1)
        samples[:, 2:4] *= scale_factor ** scale_var
    elif gtype == 'uniform_aspect':
        trans_var = bbox[2:4] * (np.random.rand(num_samples, 2) * 2 - 1)
        samples[:, 0:2] += trans_factor * trans_var
        scale_var1 = scale_factor ** (np.random.rand(num_samples, 2) * 4 - 2)
        scale_var2 = scale_factor ** (scale_range * np.random.rand(num_samples, 1))
        samples[:, 2:4] *= scale_var1 * scale_var2
    elif gtype == 'whole':
        r = np.round(np.array([bbox[2]/2, bbox[3]/2, w-bbox[2]/2, h-bbox[2]/2]))
        stride = np.round(np.array([bbox[2] / 5, bbox[3] / 5]))
        dx, dy, ds = np.meshgrid(np.arange(r[0], r[2] + 1, stride[0]),
                                 np.arange(r[1], r[3] + 1, stride[1]),
                                 np.arange(-5, 6, 1))
        windows = np.zeros((np.size(dx), 4))
        windows[:, 0] = np.ravel(dx)
        windows[:, 1] = np.ravel(dy)
        windows[:, 2] = bbox[2] * scale_factor ** np.ravel(ds)
        windows[:, 3] = bbox[3] * scale_factor ** np.ravel(ds)
        current_sample = 0
        windows_num = windows.shape[0]
        while current_sample < samples.shape[0]:
            indices = np.random.choice(np.arange(windows_num), np.minimum(windows_num, num_samples - current_sample))
            samples[current_sample:current_sample+indices.shape[0]] = windows[indices, :]
            current_sample += indices.shape[0]
    else:
        raise ValueError('Invalid gtype in generate_sample function.')

    samples[:, 2] = np.maximum(10, np.minimum(w - 10, samples[:, 2]))
    samples[:, 3] = np.maximum(10, np.minimum(w - 10, samples[:, 3]))


    samples[:, 0] -= samples[:, 2] / 2
    samples[:, 1] -= samples[:, 3] / 2

    samples[:, 0] = np.maximum(samples[:, 2]/2, np.minimum(w-samples[:, 2]/2, samples[:, 0]))
    samples[:, 1] = np.maximum(samples[:, 3]/2, np.minimum(w-samples[:, 3]/2, samples[:, 1]))

    samples = np.round(samples)

    assert np.all([samples >= 0])
    return samples

if __name__ == "__main__":
    impath = '/home/qiechunguang/Desktop/tracking/datasets/vot2013/bicycle/00000001.jpg'
    im = imread(impath)
    gt = np.array([154.00,94.00,18.00,48.00])
    samples1 = generate_samples('uniform_aspect', im.shape, gt, 128, 1.05, 0.3, 10)
    draw_rect(im, samples1)
    #input()
    samples2 = generate_samples('gaussian', im.shape, gt, 128, 1.05, 0.1, 5)
    draw_rect(im, samples2)
    #input()
    samples3 = generate_samples('uniform', im.shape, gt, 128, 1.05, 1, 10)
    draw_rect(im, samples3)
    samples4 = generate_samples('whole', im.shape, gt, 128, 1.05)
    draw_rect(im, samples4)
