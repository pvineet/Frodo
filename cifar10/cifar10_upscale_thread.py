
# coding: utf-8

# ### Problem

# - CIFAR10 is a dataset with 10 class
# - 5000 train and 1000 test images per class
# - Image size = 32x32
# 
# - Most CNNs have input size of around 224x224
# 
# Will try to scale up the input images using cubic interpolation
# Using multiprocessing to  scale parallely

import matplotlib.pyplot as plt
import scipy.ndimage
import numpy as np
import time
import h5py
from keras import models, layers
from keras.datasets import cifar10
from multiprocessing import Pool

NUM_CORES = 4

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


hf = h5py.File('/home/pvineet/Frodo/cifar10_scaled_2.h5', 'w')


def scale_image(in_image, factor=7):
    """
        params: in_image: input image
                factor: the factor by which we want to scale the image
        return: the scaled image
    """
    return scipy.ndimage.zoom(in_image, (2,2,1), order=3)

out_x_train = np.ndarray((len(x_train),64,64,3),np.uint8)
out_x_test = np.ndarray((len(x_test),64,64,3),np.uint8)

if __name__ == '__main__':
    # Scale training images
    with Pool(NUM_CORES) as p:
        start_time =  time.time()
        out_x_train = p.map(scale_image,x_train)
        end_time = time.time()
        print("Time taken scale train images = {0}".format(end_time-start_time))
        # plt.imshow(out_x_train[0])
        # plt.show()

    # Test images
    # Use Multiprocessing to scale on each core
    with Pool(NUM_CORES) as p:
        start_time =  time.time()
        out_x_test = p.map(scale_image,x_test)
        end_time = time.time()
        print("Time taken scale test images = {0}".format(end_time-start_time))
        # plt.imshow(out_x_test[0])
        # plt.show()

    # Storing scaled data set as HDF5 file
    hf.create_dataset('train_data', data=out_x_train)
    hf.create_dataset('train_labels', data=y_train)
    hf.create_dataset('test_data', data=out_x_test)
    hf.create_dataset('test_labels', data=y_test)
    hf.close()
