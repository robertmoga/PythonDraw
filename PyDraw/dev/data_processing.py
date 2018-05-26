import cv2
import numpy as np
from data_to_letters import DataToImage
from data_to_letters import ImageNormaliser


def image_norm(img):
    img = ImageNormaliser.resize_percent(img, 40)
    img = ImageNormaliser.thresholding(img)

    return img

'''
    Element class is a container for image and type of it : line, word or char
'''
class Element:
    def __init__(self, img, type='undef'):
        self.image = img
        self.type = type
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        # aici s-ar putea pune predictul
        # (nu stiu daca e o idee prea buna sa incarcam modelul pentru un singur simbol dar disctuam)


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



    #define words
    def execute_morpho_on_ox(self):
        img = self.raw_image
        img = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.fillPoly(img, contours, color=(255, 255, 255))

        # kernels for morphological operations
        kernel1 = np.ones((30, 2))
        kernel2 = np.ones((3, 3))

        dilation = cv2.dilate(img, kernel2, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel1)
        erosion = cv2.erode(closing, kernel2, iterations=2)

        img = cv2.cvtColor(erosion, cv2.COLOR_RGB2GRAY)
        return img


    def compute_projection_oy(self, morpho_image):

        oy_values = list()
        for i in range(self.raw_image.shape[0]):
            oy_values.append(0)
            for j in range(self.raw_image.shape[1]):
                if morpho_image[i][j] > 0:
                    oy_values[i] += 1
        return oy_values

    def compute_projection_ox(self, morpho_image):
        oy_values = list()
        # print(morpho_image.shape)
        # print(morpho_image[328][47])
        for i in range(self.raw_image.shape[1]):
            oy_values.append(0)
            for j in range(self.raw_image.shape[0]):
                # print(str(i) + " , " + str(j))
                # if i == 88:
                #     print('hue')
                try:
                    if morpho_image[j][i] > 0:
                        pass
                        oy_values[i] += 1
                except Exception as e:
                    print(">> " + str(i) +" , " +str(j))
                    print(str(e))
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
        self.elements = self.build_elements()

    #private method
    def __build_bounds(self):
        img_morpho = self.execute_morpho_on_oy()
        oy_values = self.compute_projection_oy(img_morpho)
        bounds = self.get_bounds(oy_values)

        return bounds


    def build_elements(self, verbose=0):
        elements = list()
        height_negative_space = 20

        if verbose == 1:
            print(">> Height of added negative space : " + str(height_negative_space))

        for elem in self.bounds:

            new_elem = self.raw_image[elem[0]:elem[1], :]
            if verbose == 1:
                print(">> Current raw elem shape " + str(new_elem.shape))
            new_elem_np = self.__add_negative_space(new_elem, height_negative_space)
            new_elem = Element(img=new_elem_np, type='line')
            elements.append(new_elem)

        return elements

    def __add_negative_space(self, element, height):

        space = np.zeros((height, self.raw_image.shape[1]))
        new_elem = np.vstack((space, element))
        new_elem = np.vstack((new_elem, space))


        return new_elem


class WordBuilder(ImageAnalyser):
    def __init__(self, raw_img):
        ImageAnalyser.__init__(self, raw_img=raw_img)
        self.raw_image = raw_img
        self.bounds = self.__build_bounds()
        self.elements = self.build_elements()

    def __build_bounds(self):
        img_morpho = self.execute_morpho_on_ox()
        ox_values = self.compute_projection_ox(img_morpho)
        bounds = self.get_bounds(ox_values)

        return bounds

    @staticmethod
    def __merge_bounds(elements, max_distance=10, verbose=0):
        i = 0
        while i < len(elements) - 1:
            e1 = elements[i]
            if i + 1 < len(elements):
                e2 = elements[i + 1]

                if abs(e2[0] - e1[1]) < max_distance:
                    if verbose == 1:
                        print(">> Merging bounds : " + str(e1) + " and " + str(e2))
                    new_elem = (e1[0], e2[1])
                    elements[i] = new_elem
                    elements.pop(i + 1)
                    i -= 1
            i += 1
        return elements

    def build_elements(self, verbose=0):
        elements = list()
        width_negative_space = 10
        max_dist_allowed = 10

        if verbose == 1:
            print(">> Width of added negative space : " + str(width_negative_space))

        if len(self.bounds) > 1:
            self.bounds = self.__merge_bounds(self.bounds, max_dist_allowed, verbose)

        for elem in self.bounds:

            new_elem = self.raw_image[:, elem[0]:elem[1]]
            if verbose == 1:
                print(">> Current raw elem shape " + str(new_elem.shape))
            new_elem_np = self.__add_negative_space(new_elem, width_negative_space)
            new_elem = Element(img=new_elem_np, type='word')
            elements.append(new_elem)

        return elements

    def __add_negative_space(self, element, width):

        space = np.zeros((self.raw_image.shape[0], width))
        new_elem = np.hstack((space, element))
        new_elem = np.hstack((new_elem, space))
        return new_elem



def test():
    pass


def get_elems(img):

    line_builder = LineBuilder(img)
    lines = line_builder.elements

    for line in lines:
        word_builder = WordBuilder(line.image)
        words = word_builder.elements
        for elem in words:
            ImageNormaliser.plotData(elem.image)

if __name__ == "__main__":

    data_reader = DataToImage("tempFiles/fis.txt", "tempFiles/newImage.png")
    image = data_reader.image

    image = image_norm(image)

    get_elems(image)

    # test()

