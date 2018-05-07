import cv2
import base64
import numpy as np
import threading
import matplotlib.pyplot as pt
import pandas as pd
from sklearn import ensemble
import time
import os


'''
    Reads data from file where the server is buffering it
    Normalises data to the desierd shape 
            - tresholding
            - resize : in order to reduce the are for computation
            - does morphological transformations
    Realises histogram-like measurments for whites on y axis
    Analyses the whites values
    Get the bounds between letters
    Normalise letters 
    Return letters 28x28   
'''



class image_normaliser():

    def __init__(self, base_64_string):
        img = turn_to_image(base_64_string)

    def turn_to_image(self, base64):
        pass

class data_to_image:

    def __init__(self, dataPath , imagePath):
        self._base64 = self.read_from_file(dataPath)
        self.imageName = imagePath
        self.base64_to_image()
        self._image = None

    @property
    def base64(self):
        return self.base640
    @base64.setter
    def base64(self, value):
        #adding a condition
        self._base64 = value

    @staticmethod
    def partition(strOne):
        if ',' in strOne:
            strOne = strOne.partition(",")[2]
            pad = len(strOne) % 4
            strOne += "=" * pad
            return strOne
        else:
            return strOne

    def read_from_file(self, dataPath):
        f = open(dataPath, 'r')
        strOne = f.read()
        f.close()
        # print(">> " +strOne)
        data = self.partition(strOne)
        return data

    def base64_to_image(self):
        base64_str = self.partition(self._base64)
        binary_str = base64.b64decode(base64_str)
        # Here comes the naming solution\
        self.imageName = "tempFiles/newImage.png"
        with open(self.imageName, "wb") as fh:
            fh.write(binary_str)

    def get_image(self):
        img = cv2.imread(self.imageName, cv2.IMREAD_GRAYSCALE)
        return img

    @property
    def image(self):
        self._image = self.get_image()
        return self._image

    @image.setter
    def image(self, value):
        if isinstance(value,str):
            self._image = value



def plotData(img, winname):
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
    cv2.imshow(winname, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    obj = data_to_image("tempFiles/fis.txt", "tempFiles/newImage.png")
    image = obj.image
    plotData(image, 'test')