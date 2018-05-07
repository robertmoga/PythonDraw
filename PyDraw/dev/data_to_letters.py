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
    Reads data from file where the server that is buffering it
    Normalises data to the desierd shape 
            - tresholding
            - resize : in order to reduce the area for computation : percent + contours
            - does morphological transformations
    Realises histogram-like measurments for whites on y axis
    Analyses the whites values
    Get the bounds between letters - Done
    Normalise letters 
    Return letters 28x28   
'''


class ImageNormaliser:
    def __init__(self, raw_img=None):
        self.raw_image = raw_img
        self._image = None

    @staticmethod
    def plotData(image, winname):

        cv2.namedWindow(winname)  # Create a named window
        cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
        cv2.imshow(winname, image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def thresholding(img):
        ret, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        return thresh

    @staticmethod
    def resize_specific_dim(img, val):
        resized_image = cv2.resize(img, (val, val))
        # res = cv2.resize(img, None, fx=1/10, fy=1/10, interpolation=cv2.INTER_CUBIC) #for scaling up
        return resized_image

    @staticmethod
    def resize_percent(img, val):
        new_width = int(img.shape[1] * val / 100)
        new_height = int(img.shape[0] * val / 100)
        dim = (new_width, new_height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        return resized

    @staticmethod
    def find_contours(img):
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        err = 20  # error for croping
        x, y, w, h = cv2.boundingRect(contours[0])
        new_img = img[y - err:y + h + err, x - err:x + w + err]
        img = new_img

        return img

    def apply_processing(self):
        img = self.raw_image
        img = self.thresholding(img)
        img = self.resize_percent(img, 50)
        img = self.find_contours(img)
        return img

    @property
    def image(self):
        self._image = self.apply_processing()
        return self._image

    @image.setter
    def image(self, value):
        if isinstance(value, numpy.ndarray):
            self._image = value


class DataToImage:

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

class ImageAnalyser:

    def __init__(self, raw_image=None):
        self.raw_image = raw_image
        self.image_morpho = self.morpho


    def execute_morpho(self):
        im2, contours, hierarchy = cv2.findContours(self.raw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.cvtColor(self.raw_image, cv2.COLOR_GRAY2RGB)
        cv2.fillPoly(img, contours, color=(255, 255, 255))
        #kernels for morphological operations
        kernel1 = np.ones((30, 2))
        kernel2 = np.ones((3, 3))

        dilation = cv2.dilate(img, kernel2, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel1)
        erosion = cv2.erode(closing, kernel2, iterations=2)

        return erosion

    @property
    def morpho(self):
        self.image_morpho = self.execute_morpho()
        return self.image_morpho

    @morpho.setter
    def morpho(self, value):
        if isinstance(value, numpy.ndarray):
            self.image_morpho = value

    def get_histogram_values(self):

        img = self.image_morpho

        height, width = img.shape[0], img.shape[1]
        white_values = list()

        im2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        for i in range(width):
            white_values.append(0)
            for j in range(height):
                # print(im2[i][j])
                if im2[j][i] == 255:
                    white_values[i] += 1

        # print(str(white_values))
        return white_values

    @staticmethod
    def get_boundaries_from_histogram(white_values):

        boundaries = list()
        index = -1

        def trigger(pos, vec):
            count = 0
            for i in range(pos, len(vec) - 1):
                if abs(vec[i] - vec[i + 1]) > 4:
                    return count
                count += 1

            return -1

        for i in range(len(white_values) - 1):
            if index > -1 and i < index:
                continue
            val1 = white_values[i]
            val2 = white_values[i + 1]
            if i > 1:
                val3 = white_values[i - 1]

            if (white_values[i] < 10 and white_values[i] > 0) and \
                    (white_values[i + 1] < 10 and white_values[i + 1] > 0):
                if abs(white_values[i] - white_values[i + 1]) < 10:

                    count = trigger(i, white_values)
                    if count != -1:
                        boundaries.append(int(count / 2 + i))
                        index = i + count

        return boundaries

    def drawBounds(self, bounds):
        img = self.raw_image
        height, width = img.shape[0], img.shape[1]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        for y in bounds:
            for x in range(height):
                img[x][y] = (0, 255, 0)

        return img

    def analyse_image(self):
        white_values = self.get_histogram_values()
        bounds = self.get_boundaries_from_histogram(white_values)

        return bounds

    def show_bounds(self):
        bounds = self.analyse_image()
        img = self.drawBounds(bounds)

        return img


if __name__ == "__main__":
    obj = DataToImage("tempFiles/fis1.txt", "tempFiles/newImage.png")
    image = obj.image

    norm = ImageNormaliser(image)
    img = norm.image

    analyser = ImageAnalyser(img)
    img = analyser.show_bounds()

    norm.plotData(img, winname="hue")