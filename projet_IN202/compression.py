import PIL
from PIL import Image
import numpy as np
import scipy as sp
import os
from math import log10, sqrt

def load(filename):
    toLoad= Image.open(filename)
    return np.asarray(toLoad)


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def dct2(a):
    return sp.fft.dct( sp.fft.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return sp.fft.idct( sp.fft.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')



def RGB_YCbCr():
    mat = load("test.png")
    new_mat = np.empty((mat.shape),dtype = np.uint8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):

            Y = min(mat[i,j,0] * 0.299
            + mat[i,j,1] * 0.587
            + mat[i,j,2] * 0.114
            , 255)
            
            Cb = mat[i,j,0] * -0.1687
            + mat[i,j,1] * -0.3313
            + mat[i,j,2] * 0.5
            + 128

            Cr = mat[i,j,0] * 0.5
            + mat[i,j,1] * -0.4187
            + mat[i,j,2] * -0.0813
            + 128

            new_mat[i,j] = (Y, Cb, Cr)
    return new_mat

def YCbCr_RGB():
    mat = load("test.png")
    new_mat = np.empty((mat.shape),dtype = np.uint8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):

            Y = mat[i,j,0] * 0.299
            + mat[i,j,1] * 0.587
            + mat[i,j,2] * 0.114
            
            Cb = mat[i,j,0] * -0.1687
            + mat[i,j,1] * -0.3313
            + mat[i,j,2] * 0.5
            + 128

            Cr = mat[i,j,0] * 0.5
            + mat[i,j,1] * -0.4187
            + mat[i,j,2] * -0.0813
            + 128

            new_mat[i,j] = (Y, Cb, Cr)
    return new_mat

print(RGB_YCbCr())



test = load("test.png")
Image.fromarray(test,'RGB').show()
