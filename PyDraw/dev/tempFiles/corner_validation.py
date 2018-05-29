import cv2
import base64
import numpy as np
import time
import os
from dev.data_to_letters import DataToImage
from dev.data_to_letters import ImageNormaliser
from dev.data_to_letters import CharSynthesizer
from dev.data_to_letters import ImageAnalyser

def check_corner(roi, kernel):

    img = roi*kernel
    sum_kernel = np.sum(kernel)
    sum_img = int(np.sum(img)/255)

    if sum_img > (sum_kernel*0.8):
        return True
    else:
        return False

def check_corner2(roi, kernel):
    sum_kernel = np.sum(kernel)
    img = kernel - roi
    sum_img = int(np.sum(img))
    if sum_img > sum_kernel:
        sum_img = int(np.sum(img)/255)

    if abs(sum_img) < (sum_kernel*0.5):
        return True
    else :
        return False

def corner_fill(raw):
    height, width = raw.shape[0], raw.shape[1]
    ks = 15 #kernel_size
    left_kernel = def_corner(ks=ks)
    right_kernel = def_corner(ks=ks, side='right')
    temp = raw[:int(raw.shape[0]*0.5),:]
    ImageNormaliser.plotData(temp)

    if np.any(temp[:, :] == 200):
        print("Avem 200 in temp")
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            if i+ks < temp.shape[0]-1 and j+ks < temp.shape[1]-1:
                roi = temp[i:i+ks, j:j+ks]
                if check_corner2(roi, left_kernel):
                    raw[i:i+ks, j:j+ks] = np.ones((ks, ks))*200
                if check_corner2(roi, right_kernel):
                    raw[i:i+ks, j:j+ks] = np.ones((ks, ks))*200
    if np.any(temp[:, :] == 200):
        print(">>Avem 200 in temp")
    ImageNormaliser.plotData(raw)
    return raw

def def_corner(ks=10, side='left'):
    kernel = np.zeros((ks,ks))

    if(side == 'left'):
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                if i == 0 or j == 0:
                    kernel[i][j] = 1
        return kernel
    else:
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                if i == 0 or j == (kernel.shape[0]-1):
                    kernel[i][j] = 1
        return kernel

def test():
    a=5
    b = a

    b = 1

    print(str(a) + "  " + str(b))

if __name__ == "__main__":
    print(">> Strat")
    obj = DataToImage("tempFiles/fis.txt", "tempFiles/newImage.png")
    image = obj.image

    norm = ImageNormaliser(image)
    img = norm.image

    #corner validation
    img = corner_fill(img)
    # test()

    # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #
    # analyser = ImageAnalyser(img)
    # analyser.drawBounds()
    # char = CharSynthesizer(img, analyser.bounds)
    # letters = char.letters
    #
    # for elem in letters:
    #     ImageNormaliser.plotData(elem, 'hue')