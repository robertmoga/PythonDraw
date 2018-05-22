import cv2
import base64
import numpy as np
import time
import os
from data_to_letters import DataToImage
from data_to_letters import ImageNormaliser
from data_to_letters import CharSynthesizer
from data_to_letters import ImageAnalyser

def check_sides(img, kernel):
    # if img.shape[0] == kernel.shape[0] and kernel.shape[1] == kernel.shape[1]:
    if img.shape == kernel.shape:
        #produs
        #verificare


def corner_dilation(img):
    height, width = img.shape[0], img.shape[1]
    print(str(height) + "  "+ str(width))

    upper = img[:int(img.shape[0]*0.5),:]
    print(upper.shape)

    kernel = np.ones((10, 10))
    for i in range(upper.shape[0]):
        for j in range(upper.shape[1]):
            if i+20 < upper.shape[0] and j+20 < upper.shape[1]:
                roi = upper[i:i+20, j:j+20]
                check_sides(roi, kernel)
                print(roi.shape)
                break

    # ImageNormaliser.plotData(upper)

if __name__ == "__main__":
    obj = DataToImage("tempFiles/fis.txt", "tempFiles/newImage.png")
    image = obj.image

    norm = ImageNormaliser(image)
    img = norm.image

    #corner validation
    corner_dilation(img)

    # analyser = ImageAnalyser(img)
    # char = CharSynthesizer(img, analyser.bounds)
    # letters = char.letters
    #
    # for elem in letters:
    #     ImageNormaliser.plotData(elem, 'hue')