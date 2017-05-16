# __ coding: UTF-8 __

'''
convariance calculation
C = 1/N x Matrix 1 x Matrix 1T
'''
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
#img read and transformation

def imgread(path):
    img = Image.open(path)
    img = np.asarray(img)
    count = img.shape
    num = count[0]*count[1]
    img = np.reshape(img, (num,1))
    return img

def meanface(facematrix):
    meanface = facematrix.mean(1)
    return meanface
    
#convariance
def conv(a):
    b = np.transpose(a)
    multiply = np.matmul(a,b)
    count = a.shape
    num = count[0]*count[1]
    result = multiply / num
    return result

def eig(convariance):
    value, vector = np.linalg.eig(convariance)
    return vector

def eigface(vector,no):
    mainface = vector[:,:no]
    return mainface

###Main Function

###Img Read
i = 1
img = np.zeros((4608,20),dtype = np.float )

while i < 21:
    path = 'C:\\Users\\Orion_Peng\\Desktop\\Training\\' + str(i)+'.bmp'
    img[:,i-1:i] = imgread(path)
    i += 1

mean = meanface(img)

mean = np.reshape(mean,(4608,1))

demean1 = img - mean


demean = np.transpose(demean1)


convariance = conv(demean)


eigenvector = eig(convariance)

eigenfacematrix = eigface(eigenvector, 20)

eigenfacematrix = np.matmul(demean1,eigenfacematrix)

splitface = np.hsplit(eigenfacematrix, 20)

for item in splitface:
    eigenface = np.reshape(item,(72,64))
    plt.imshow(eigenface,cmap = 'gray')
    plt.show()

    
   



    
    
