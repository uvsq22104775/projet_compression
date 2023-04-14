import PIL
from PIL import Image
import numpy as np
import scipy as sp
import os
from math import log10, sqrt


'''Question 4 à faire'''


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
    global orig_shape
    shape = list(image.shape)
    orig_shape = list(image.shape)
    shape[0] += 8 - shape[0] % 8 if shape[0] % 8 != 0 else 0
    shape[1] += 8 - shape[1] % 8 if shape[1] % 8 != 0 else 0
    new_image = np.empty(shape, dtype = np.uint8)
    for x in range (orig_shape[0]):
        for y in range (orig_shape[1]):
            new_image[x, y] = image[x, y]
    return new_image


def anti_padding(image):
    global orig_shape
    new_image = np.empty(orig_shape, dtype = np.uint8)
    for x in range (orig_shape[0]):
        for y in range (orig_shape[1]):
            new_image[x, y] = image[x, y]
    return new_image




def sous_echantillonage(mat):
    shape = list(mat.shape)
    shape[1] = (shape[1] - 1) // 2 + 1

    new_matY = np.empty((list(mat.shape)[0], list(mat.shape)[1]), dtype = np.uint8)
    new_matCb = np.empty((shape[0], shape[1]), dtype = np.uint8)
    new_matCr = np.empty((shape[0], shape[1]), dtype = np.uint8)

    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            new_matY[x, y] = mat[x, y, 0]

    for x in range(shape[0]):
        for y in range(shape[1]):
            new_matCb[x, y] = (mat[x, 2*y, 1] / 2) + (mat[x , 2*y + 1, 1] / 2)
            new_matCr[x, y] = (mat[x, 2*y, 2] / 2) + (mat[x , 2*y + 1, 2] / 2)
    return [new_matY, new_matCb, new_matCr]


def anti_sous_echantillonage(matYmatCbmatCr):
    matY, matCb, matCr = matYmatCbmatCr[0], matYmatCbmatCr[1], matYmatCbmatCr[2]
    new_mat = np.empty((matY.shape[0], matY.shape[1], 3), dtype = np.uint8)
    for x in range(new_mat.shape[0]):
        for y in range(new_mat.shape[1]):
            new_mat[x, y, 0] = matY[x, y]
            new_mat[x, y, 1] = matCb[x, y//2]
            new_mat[x, y, 2] = matCr[x, y//2]
    return new_mat














test = load("test.png")



'''affiche test.png en RGB'''
# Image.fromarray(test, mode = "RGB").show()


#           Q1
'''transforme test.png en YCbCr'''
# Image.fromarray(RGB_YCbCr(test), mode = "YCbCr").show()


#           Q2
'''transforme test.png en YCbCr puis de retour en RGB'''
# Image.fromarray(YCbCr_RGB(RGB_YCbCr(test)), mode = "RGB").show()

'''calcule la ressemblance entre l'image transformée ↑ avec l'image originale'''
# print(psnr(test, YCbCr_RGB(RGB_YCbCr(test))))


#           Q3
'''les dimensions de l'image deviennent des multiples de 8 (padding)'''
# Image.fromarray(padding(test), mode = "RGB").show()

'''enlève un padding déjà ajouté'''
# Image.fromarray(anti_padding(padding(test)), mode = "RGB").show()


#           Q4
'''l'image deviens 2x plus courte en largeur'''
sous_echantillonage(RGB_YCbCr(test))


#           Q5
'''l'image deviens 2x plus longue en largeur'''
Image.fromarray(anti_sous_echantillonage(sous_echantillonage(RGB_YCbCr(test))), mode = "YCbCr").show()