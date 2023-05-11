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


# Question 6 
#

def decoupage_matrice(mat):
    global orig_shape_2
    m_new_Y = np.empty(mat.shape, dtype = np.uint8)
    m_new_Cb = np.empty(mat.shape, dtype = np.uint8)
    m_new_Cr = np.empty(mat.shape, dtype = np.uint8)
    decoupage = 8
    orig_shape_2 = [mat.shape[0]//8, mat.shape[1]//8]
    liste3 = []
    for x in range(mat.shape[0]//8):
        for y in range(mat.shape[1]//8):
            liste2 = []
            for i in range(decoupage):               # on la decoupe en bloc 8x8
                liste1 = []
                for j in range(decoupage):
                    liste1.append(mat[i+8*x][j+8*y][0])
                liste2.append(liste1)
            liste3.append(liste2)
    m_new_Y = np.array(liste3)


    liste3 = []
    for x in range(mat.shape[0]//8):
        for y in range(mat.shape[1]//8):
            liste2 = []
            for i in range(decoupage):
                liste1 = []
                for j in range(decoupage):
                    liste1.append(mat[i+8*x][j+8*y][1])
                liste2.append(liste1)
            liste3.append(liste2)
    m_new_Cb = np.array(liste3)


    liste3 = []
    for x in range(mat.shape[0]//8):
        for y in range(mat.shape[1]//8):
            liste2 = []
            for i in range(decoupage):
                liste1 = []
                for j in range(decoupage):
                    liste1.append(mat[i+8*x][j+8*y][2])
                liste2.append(liste1)
            liste3.append(liste2)
    m_new_Cr = np.array(liste3)
    return [m_new_Y, m_new_Cb, m_new_Cr]


# Question 7
def reconstruction_image(m_new):
    global orig_shape_2
    decoupage = 8
    img_h = orig_shape_2[0] * decoupage
    img_w = orig_shape_2[1] * decoupage
    img = np.empty((img_h, img_w, 3), dtype=np.uint8)

    for x in range(orig_shape_2[0]):
        for y in range(orig_shape_2[1]):
            for i in range(decoupage):
                for j in range(decoupage):
                    img[i + x*decoupage][j + y*decoupage][0] = m_new[0][x*orig_shape_2[1] + y][i][j]
                    img[i + x*decoupage][j + y*decoupage][1] = m_new[1][x*orig_shape_2[1] + y][i][j]
                    img[i + x*decoupage][j + y*decoupage][2] = m_new[2][x*orig_shape_2[1] + y][i][j]

    return img

# Question 8

def filtrage(arr, threshold):
    arr = np.array(arr)
    mask = np.abs(arr) < threshold
    arr[mask] = 0
    return arr


# Question 12

def rle_compress(matrix):

    matrix = matrix.astype(int)
    # Flatten the matrix into a 1D array
    flattened = matrix.reshape(-1)

    # Initialize variables for RLE compression
    count = 0
    compressed = ""

    # Iterate over the flattened array
    for i in range(len(flattened)):
        if flattened[i] == 0:
            # If the value is 0, increment the count
            count += 1
        else:
            # If the value is not 0, add the RLE-compressed string to the output
            if count > 0:
                compressed += "#{},".format(count)
                count = 0
            compressed += str(flattened[i])+","

    # If there are any 0's at the end of the array, add the RLE-compressed string to the output
    if count > 0:
        compressed += "#{}".format(count)

    # Write the compressed string to a file
    with open("compressed.txt", "wb") as f:
        f.write(bytes(compressed, "utf-8"))


 




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
# sous_echantillonage(RGB_YCbCr(test))


#           Q5
'''l'image deviens 2x plus longue en largeur'''
# Image.fromarray(anti_sous_echantillonage(sous_echantillonage(RGB_YCbCr(test))), mode = "YCbCr").show()


#           Q6
'''l'image se divise en blocs de 8x8 dans une liste à 4 dimentions'''
# print(decoupage_matrice(test))


#           Q7
blocs = decoupage_matrice(RGB_YCbCr(padding(test)))
# print(dct2(blocs))
# Image.fromarray(reconstruction_image(idct2(dct2(blocs))), mode = "YCbCr").show()

#           Q8
print(filtrage(dct2(blocs),1.5))
Image.fromarray(reconstruction_image(idct2(filtrage(dct2(blocs),1.5))), mode = "YCbCr").show()