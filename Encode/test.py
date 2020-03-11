import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from skimage.filters import threshold_otsu
import random
import numpy as np
import pickle
import keras

from keras.models import Model
from keras.models import load_model
from keras.layers.core import Lambda
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from skimage.measure import compare_ssim as ssim
from keras.utils import np_utils

from keras.layers import Input, Conv2D, concatenate, Dense, Dropout, add
# GaussianNoise, GaussianDropout
from keras.models import Model
import keras.backend as K
# from keras.utils import multi_gpu_model

import tensorflow as tf

import numpy
import scipy.ndimage
from scipy.ndimage import imread
from numpy.ma.core import exp
from scipy.constants.constants import pi
from model.discriminator import D1, D2
from training import get_batch, SSIM_LOSS

### test data
itr = get_batch(train=0)
test = next(itr)
img = test[0]['C']
msk = test[0]['W']
model = load_model('/home/CVL1/Shaobo/StegoGAN/GR.h5',custom_objects= {'SSIM_LOSS': SSIM_LOSS })
#G = load_model('/home/CVL1/Shaobo/StegoGAN/0_G.h5',custom_objects= {'SSIM_LOSS': SSIM_LOSS })

[M,W_prime] = model.predict([img,msk])

plt.figure(figsize=(60, 4))
n = 32
for i in range(n):
    ax = plt.subplot(4, n, i+1)
    plt.imshow(img[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(4, n, i+1 + n)
    plt.imshow(M[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(4, n, i +1 + 2*n)
    plt.imshow(msk[i].reshape(32, 32),cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(4, n, i +1 + 3*n)
    plt.imshow(W_prime[i].reshape(32, 32),cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
plt.savefig('/home/CVL1/Shaobo/Encode/10.jpg')
