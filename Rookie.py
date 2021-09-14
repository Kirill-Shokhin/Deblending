#!/usr/bin/env python
# coding: utf-8

##### Rookie 1.1 #####
import numpy as np
import pandas as pd
import cv2
from astropy.io import fits
from scipy.optimize import curve_fit 


def get_hdu_and_data(filename):
    with fits.open(filename) as f:
        hdu = f[0].header # Метаданные, в них есть интересная инфа о наблюдении
        data = f[0].data  # Сама картинка
        return hdu, data



def crop_img(img, size=100):
    return img[img.shape[0]//2-size:img.shape[0]//2+size,
               img.shape[1]//2-size:img.shape[1]//2+size]



def crop_objects(objects, size=100, center=1024):
    x = objects.x
    y = objects.y
    ob = objects[(x > center-size)&(x < center+size)&(y > center-size)&(y < center+size)]
    x = ob.x - center + size
    y = ob.y - center + size
    return y, x  



def crop_apex_objects(objects, size=100, center=1024):
    x = objects.x
    y = 2*center - objects.y
    ob = objects[(x > center-size)&(x < center+size)&(y > center-size)&(y < center+size)]
    x = ob.x - center + size
    y = center - ob.y + size
    return x, y



### Algorithm device 
# Circle masks
def create_kernel(r, normed=True):
    kernel = np.zeros((2*r+1, 2*r+1))
    y, x = np.ogrid[:2*r+1, :2*r+1]
    dst = np.sqrt((x-r)**2 + (y-r)**2)
    mask = ((dst <= r)&(dst > r-1)).astype(int)
    return mask/np.sum(mask) if normed else mask



# Convolution and packing in tensor 
def convolutions(img, max_r=5, normed=True):
    stack = np.zeros((max_r, *img.shape))
    for r in range(max_r):
        stack[r] = cv2.filter2D(img.astype(float), -1, create_kernel(r, normed))
    return stack



# Criterions for radius of objects
def components(stack):
    diff = np.diff(stack, axis=0)
    r = np.argmax(diff>=0, axis=0)
    return -diff[0], r


# Combining components and detecting peaks 
from scipy.ndimage import maximum_filter
def detect_peaks(img, size=3):
    peaks = np.where(maximum_filter(img, size=size) == img, img, 0)
    return peaks

def take_objects(components, treshhold=500):
    diff_model, r_model = components
    peaks = detect_peaks(diff_model, size = 3)
    objects = pd.DataFrame(np.argwhere(peaks > treshhold), columns=['x', 'y'])
    objects['value'] = peaks[objects.x, objects.y]
    objects['r'] = r_model[objects.x, objects.y]
    return objects


def Rookie(img, r=15, tresh=500):
    return take_objects(components(convolutions(img, max_r = r)), treshhold = tresh)


def Gaussian(data_tuple, amplitude, xo, yo, sigma_x, sigma_y, floor):
    (x, y) = data_tuple                                                        
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = 1/(2*sigma_x**2)
    c = 1/(2*sigma_y**2)   
    g = floor + amplitude*np.exp( - (a*((x-xo)**2) + c*((y-yo)**2)))                                   
    return g.ravel()


def get_obj(obj, img):
    x,y, _, r = obj.astype(int)
    return img[x-r+1:x+r,y-r+1:y+r]


def fit_obj(id, objects, img):
    const = 2*np.sqrt(2*np.log(2))
    x,y, _, r = objects.loc[id].astype(int)
    obj = img[x-r+1:x+r,y-r+1:y+r]
    ra = np.arange(obj.shape[0])
    x_gr, y_gr = np.meshgrid(ra, ra)
    xx = np.vstack((x_gr.ravel(),y_gr.ravel()))
    yy = obj.ravel()
    x_max, y_max = np.unravel_index(np.argmax(obj), obj.shape)
    popt = curve_fit(Gaussian, xx, yy, p0=(np.max(obj), x_max, y_max, r/const, r/const, np.median(obj)))[0]
    popt[1] += x-r+1
    popt[2] += y-r+1
    popt[3] *= const
    popt[4] *= const
    print('A = {:.2f}, center = ({:.2f}, {:.2f}),\nfwhm = ({:.2f}, {:.2f}), floor = {:.2f}'.format(*popt))
    return popt