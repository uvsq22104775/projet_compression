import PIL
from PIL import Image
import numpy as np
import scipy as sp
import os
from math import log10, sqrt

def load(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    toLoad= Image.open(os.path.join(script_dir, filename))
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





def RGB_YCbCr(image):
    new_image = np.empty(image.shape, dtype = np.uint8)
    new_image[:,:,0] = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
    new_image[:,:,1] = -0.1687*image[:,:,0] - 0.3313*image[:,:,1] + 0.5*image[:,:,2] + 128
    new_image[:,:,2] = 0.5*image[:,:,0] - 0.4187*image[:,:,1] - 0.0813*image[:,:,2] + 128
    return new_image


def YCbCr_RGB(image):
    image = np.float64(image.copy())
    new_image = np.empty(image.shape, dtype = np.uint8)
    new_image[:,:,0] = np.clip(image[:,:,0] + 1.402*(image[:,:,2] - 128), 0, 255)
    new_image[:,:,1] = np.clip(image[:,:,0] - 0.34414*(image[:,:,1] - 128) - 0.71414*(image[:,:,2] - 128), 0, 255)
    new_image[:,:,2] = np.clip(image[:,:,0] + 1.772*(image[:,:,1] - 128), 0, 255)
    return new_image

def padding(image):
    new_image = np.empty(image.shape[0],image.shape[1],image.shape[2], dtype = np.uint8)
    
    return new_image



test = load("test.png")
# Image.fromarray(test, mode = "RGB").show()
# Image.fromarray(RGB_YCbCr(test), mode = "YCbCr").show()
# Image.fromarray(YCbCr_RGB(RGB_YCbCr(test)), mode = "RGB").show()
print(psnr(test, YCbCr_RGB(RGB_YCbCr(test))))