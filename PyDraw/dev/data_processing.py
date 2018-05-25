import cv2
import numpy as np
from data_to_letters import DataToImage
from data_to_letters import ImageNormaliser


def image_norm(img):
    img = ImageNormaliser.resize_percent(img, 40)
    img = ImageNormaliser.thresholding(img)

    return img

'''
    Status Image Analyser 
        - made to be inherited si specialised by build_lines and build_words
        - can add method for get char bounds
        - fully developed for oy
        - must implement compute projection for ox
'''
class ImageAnalyser:

    def __init__(self, raw_img):
        self.raw_image = raw_img


    #define lines
    def execute_morpho_on_oy(self):
        im2, contours, hierarchy = cv2.findContours(self.raw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.cvtColor(self.raw_image, cv2.COLOR_GRAY2RGB)
        cv2.fillPoly(img, contours, color=(255, 255, 255))

        #kernels for morphological operations
        kernel1 = np.ones((2, 30))
        kernel2 = np.ones((3, 3))

        dilation = cv2.dilate(img, kernel2, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel1)
        erosion = cv2.erode(closing, kernel2, iterations=2)

        img = cv2.cvtColor(erosion, cv2.COLOR_RGB2GRAY)

        return img

        # define words : UNTESTED
    def execute_morpho_on_ox(self):
        im2, contours, hierarchy = cv2.findContours(self.raw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.cvtColor(self.raw_image, cv2.COLOR_GRAY2RGB)
        cv2.fillPoly(img, contours, color=(255, 255, 255))

        # kernels for morphological operations
        kernel1 = np.ones((2, 30))
        kernel2 = np.ones((3, 3))

        dilation = cv2.dilate(img, kernel2, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel1)
        erosion = cv2.erode(closing, kernel2, iterations=2)

        return erosion

    # @property
    # def bounds(self):
    #     self._bounds = self.analyse_image()
    #     return self._bounds
    #
    # @bounds.setter
    # def bounds(self, value):
    #     if isinstance(value, list):
    #         self._bounds = value

    def compute_projection_oy(self, morpho_image):

        oy_values = list()
        for i in range(self.raw_image.shape[0]):
            oy_values.append(0)
            for j in range(self.raw_image.shape[1]):
                if morpho_image[i][j] > 0:
                    oy_values[i] += 1
        return oy_values

    def get_bounds(self, values):
        index = -1
        bounds = list()

        def trigger(values, pos):
            count = -1

            for i in range(pos, len(values)):
                if values[i] == 0:
                    return count
                count += 1
            return -1

        for i in range(len(values)):
            if index > -1 and i < index:
                continue

            if values[i] > 0:
                count = trigger(values, i)
                if count > 0:
                    index = count + i
                    bounds.append((i, count+i))

        return bounds


class LineBuilder(ImageAnalyser):
    hue = 0
    def __init__(self, raw_img):
        ImageAnalyser.__init__(self, raw_img=raw_img)
        self.raw_image = raw_img
        self.bounds = self.__build_bounds()
        # self.elements = self.build_elements()

    #private method
    def __build_bounds(self):
        img_morpho = self.execute_morpho_on_oy()
        oy_values = self.compute_projection_oy(img_morpho)
        bounds = self.get_bounds(oy_values)

        return bounds

    def build_elements(self):
        elements = list()

        for elem in self.bounds:

            new_elem = self.raw_image[elem[0]:elem[1], :]
            elements.append(new_elem)

        for img in elements:
            ImageNormaliser.plotData(img)

    def add_neagtive_space(self, elements):
        pass


def test(img):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.fillPoly(img, contours, color=(255, 255, 255))

    for cnt in contours:
        # print(">>" + str(cnt))
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = cv2.drawContours(img, [box], -1, (200, 0, 0), 2)

    ImageNormaliser.plotData(img)
    return img

if __name__ == "__main__":

    data_reader = DataToImage("tempFiles/fis.txt", "tempFiles/newImage.png")
    image = data_reader.image

    image = image_norm(image)

    line_builder = LineBuilder(image)
    line_builder.build_elements()

    # print(line_builder.bounds)

    # ImageNormaliser.plotData(img)

    # img = test(image)
    # ImageNormaliser.plotData(img)