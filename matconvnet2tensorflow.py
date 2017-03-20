"""
Convert a MDNet MatConvNet .mat file to tensorflow .meta and .data file
"""

import scipy.io

matdata = scipy.io.loadmat('./matmodels/mdnet_vot-otb.mat')
print(matdata)
