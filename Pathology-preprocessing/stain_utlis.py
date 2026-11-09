from __future__ import division

import os
import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 



def read_image(path):
    '''
    Read an image to RGB uint8 format
    :param path: path to the image
    :return
    '''
    
    im = cv.imread(path)
    if im is None:
        raise FileNotFoundError("Unable to read image: " + path)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im 


def show_color(C):
    '''
    Shows rows of C in colors (RGB)
    :param C:
    :return
    '''
    n =  C.shape[0]
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(np.ones((100, 100, 3)) * C[i, :].reshape(1, 1, 3))
        plt.axis("off")
    
