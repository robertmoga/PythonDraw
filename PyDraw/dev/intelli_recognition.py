import cv2
import numpy as np
from .InputProcessing import InputFormatting
from .InputProcessing import DataToImage
from .data_to_letters import ImageNormaliser
# from .test_calibrare import DataToImage
import os
import re
from keras.models import load_model
import base64

# class DataToImage:
#     def __init__(self, dataPath, imagePath):
#         self._base64 = self.read_from_file(dataPath)
#         self.imageName = imagePath
#         self.base64_to_image()
#         self._image = None
#
#     @property
#     def base64(self):
#         return self._base64
#
#     @base64.setter
#     def base64(self, value):
#         if isinstance(value, str):
#             self._base64 = value
#
#     @staticmethod
#     def partition(strOne):
#         if ',' in strOne:
#             strOne = strOne.partition(",")[2]
#             pad = len(strOne) % 4
#             strOne += "=" * pad
#             return strOne
#         else:
#             return strOne
#
#     def read_from_file(self, dataPath):
#         f = open(dataPath, 'r')
#         strOne = f.read()
#         f.close()
#         # print(">> " +strOne)
#         data = self.partition(strOne)
#         return data
#
#     def base64_to_image(self):
#         base64_str = self.partition(self._base64)
#         binary_str = base64.b64decode(base64_str)
#         # Here comes the naming solution\
#         self.imageName = "tempFiles/newImage.png"
#         with open(self.imageName, "wb") as fh:
#             fh.write(binary_str)
#
#     def get_image(self):
#         img = cv2.imread(self.imageName, cv2.IMREAD_GRAYSCALE)
#         return img
#
#     @property
#     def image(self):
#         self._image = self.get_image()
#         return self._image
#
#     @image.setter
#     def image(self, value):
#         if isinstance(value, str):
#             self._image = value
#

def image_norm(img):
    # we are doing the default threshold in order to accept every color in the image
    # then we are resizing and afterwards we are thresholding with the thresh 200
    # in order to keep most of the greys that appeard during the interpolation in the image.
    img = ImageNormaliser.thresholding(img)
    img = ImageNormaliser.resize_percent(img, 50)
    img = ImageNormaliser.thresholding(img, thresh=200)
    # ImageNormaliser.plotData(img)

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

'''
class ImageAnalyser:

    def __init__(self, raw_img):
        self.raw_image = raw_img

    # define lines
    def execute_morpho_on_oy(self):
        img = self.raw_image
        img = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.fillPoly(img, contours, color=(255, 255, 255))
        # ImageNormaliser.plotData(img)
        #kernels for morphological operations
        kernel1 = np.ones((2, 30))
        kernel2 = np.ones((3, 3))

        dilation = cv2.dilate(img, kernel2, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel1)
        erosion = cv2.erode(closing, kernel2, iterations=2)

        img = cv2.cvtColor(erosion, cv2.COLOR_RGB2GRAY)
        return img

    # define words
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

    def get_bounds_variable(self, white_values):

        bounds = list()
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

            if (white_values[i] < 8 and white_values[i] > 0) and \
                    (white_values[i + 1] < 8 and white_values[i + 1] > 0):
                if abs(white_values[i] - white_values[i + 1]) < 5:

                    count = trigger(i, white_values)
                    if count != -1:
                        bounds.append(int(count / 2 + i))
                        index = i + count

        return bounds

    @staticmethod
    def compute_partial_projection(morpho_image, axis='ox'):
        positions = [0.25, 0.40, 0.5, 0.60, 0.75]
        values = list()
        if axis == 'ox':
            for pos in positions:
                values.append(0)
                j = int(morpho_image.shape[1] * pos)
                for i in range(morpho_image.shape[0]):
                    if morpho_image[i][j] == 0:
                        values[positions.index(pos)] += 1
                        # morpho_image[i][j] = 100
            # ImageNormaliser.plotData(morpho_image)
        else:
            pass

        sum = 0
        for val in values:
            sum += val
        return int(sum/len(values))

    @staticmethod
    def draw_bounds(image, bounds, axis='ox'):

        img = image.copy()

        if axis == 'ox':
            try:
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                for elem in bounds:
                    for i in range(img.shape[0]):
                        img[i][elem[0]] = (0, 255, 0)
                        img[i][elem[1]] = (0, 255, 0)
            except :
                for elem in bounds:
                    for i in range(img.shape[0]):
                        img[i][elem[0]] = 200
                        img[i][elem[1]] = 200
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for elem in bounds:
                for i in range(img.shape[1]):
                    img[elem[0]][i] = (0, 255, 0)
                    img[elem[1]][i] = (0, 255, 0)
        ImageNormaliser.plotData(img)


class LineBuilder(ImageAnalyser):
    def __init__(self, element):
        ImageAnalyser.__init__(self, raw_img=element.image)
        self.raw_image = element.image
        self.bounds = self.__build_bounds()
        self.elements = self.build_elements()

    #private method
    def __build_bounds(self):
        img_morpho = self.execute_morpho_on_oy()
        # ImageNormaliser.plotData(img_morpho)
        oy_values = self.compute_projection_oy(img_morpho)
        bounds = self.get_bounds(oy_values)
        # self.draw_bounds(self.raw_image, bounds, axis='oy')
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
    def __init__(self, element):
        ImageAnalyser.__init__(self, raw_img=element.image)
        self.raw_image = element.image
        self.bounds = self.__build_bounds()
        self.elements = self.build_elements()

    def __build_bounds(self):
        img_morpho = self.execute_morpho_on_ox()
        # ImageNormaliser.plotData(img_morpho)
        ox_values = self.compute_projection_ox(img_morpho)
        bounds = self.get_bounds(ox_values)
        # self.draw_bounds(self.raw_image, bounds)

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
        width_negative_space = 20
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


class CharBuilder(ImageAnalyser):

    def __init__(self, element):
        ImageAnalyser.__init__(self, element.image)
        self.raw_image = element.image
        self._mean_height = -1
        self.bounds = self.__build_bounds()
        self.elements = self.build_elements()

    def __build_bounds(self):
        img_morpho_ox = self.execute_morpho_on_ox()
        img_morpho_oy = self.execute_morpho_on_oy()
        ox_values = self.compute_projection_ox(img_morpho_ox)
        self._mean_height = self.compute_partial_projection(morpho_image=img_morpho_oy, axis='ox')
        bounds = self.get_bounds_variable(ox_values)
        bounds = self.__normalise_bounds(bounds)
        # self.draw_bounds_hue(bounds)
        print(">>> " + str(bounds))

        return bounds

    def __normalise_bounds(self, bounds):
        # elinate the bounds that are above the thresh
        i = 0
        while i < len(bounds)-1:
            if self.__check_bound(bounds[i]):
                bounds.remove(bounds[i])
                i -= 1
            i += 1
        #turn the list of values into list of tuples
        bounds = [0] + bounds + [self.raw_image.shape[1]]
        result = list()
        for i in range(len(bounds) -1):
            result.append((bounds[i], bounds[i+1]))
        return result

    def __check_bound(self, bound):
        thresh = int(self._mean_height * 0.6)
        for i in range(self.raw_image.shape[0]):
            if self.raw_image[i][bound] > 0:
                if i > thresh:
                    # print(i)
                    return False
                else:
                    return True

    def draw_bounds_hue(self, bounds):
        print(bounds)
        img = self.raw_image.copy()
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for elem in bounds:
            for i in range(img.shape[0]):
                img[i][elem[0]] = (0,255,0)
                # img[i][elem[1]-1] = (0,255,0)
        val = int(self._mean_height *0.6)
        for i in range(img.shape[1]):
            img[val][i] = (0,0,255)

        ImageNormaliser.plotData(img)

    def build_elements(self, verbose=0):
        elements = list()

        for elem in self.bounds:
            if elem == 0:
                continue
            else:
                new_elem = self.raw_image[:, elem[0]:elem[1]]
                # ImageNormaliser.plotData(new_elem)
                char_synth = CharSynthesizer(new_elem)
                new_elem = char_synth.normalise_char()
                if new_elem is not None:
                    # print(">>" + str(new_elem.shape))
                    # ImageNormaliser.plotData(new_elem)
                    ret, thresh = cv2.threshold(new_elem, 0, 255, cv2.THRESH_BINARY)
                    new_elem = thresh
                    new_elem = Element(img=new_elem, type='char')
                    elements.append(new_elem)

        return elements


class CharSynthesizer:
    def __init__(self, img, resize_dim=64):
        self.__img = img
        self.__resize_dim = resize_dim

    def normalise_char(self):
        img = self.crop_char(self.__img)
        img = ImageNormaliser.resize_specific_dim(img, self.__resize_dim)
        if np.count_nonzero(img > 0):
            # ImageNormaliser.plotData(img)
            return img
        return None

    def crop_char(self, img):
        img = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = img.shape[0], img.shape[1]
        #init the oy exterme coordinates for letter
        y_max = -1
        y_min = height + 1
        x_coords = list()
        y_coords = list()
        return_image = None

        for i in contours:
            for j in i:
                for elem in j: #for every point in countour
                    y = elem[1]
                    x_coords.append(elem[0])
                    y_coords.append(elem[1])
                    if y > y_max: #define oy extremities
                        y_max = y
                    elif y < y_min:
                        y_min = y

        if y_max > -1 and y_min < height+1:
            x_coords= np.array(x_coords)
            y_coords = np.array(y_coords)
            x_distrib = x_coords.sum()/len(x_coords) #compute mean of the x coords
            y_distrib = y_coords.sum()/len(y_coords) #compute mean of the y coords
            new_height = y_max - y_min
            image = img[y_min: y_max, :] #the letter touches all bounds of the image

            #square the rectangle
            try:
                if new_height > width:
                    return_image = self.fill_letter_ox(image, x_distrib) #get a square image

                    return_image = self.square_img(return_image)
                elif width > new_height:
                    return_image = self.fill_letter_oy(image, y_distrib, height)
                    # ImageNormaliser.plotData(return_image)
                    return_image = self.square_img(return_image)
                    pass
                else:
                    return_image = self.square_img(image)
                    pass
            except Exception as e:
                print("Exception in filling the letters : " + str(e))
        else:
            print(">> There is no char in this split")

        return return_image

    # methods are returning a square image with centred letter based on info distribution
    @staticmethod
    def fill_letter_ox(image,x_distrib):
        direction = None
        height, width = image.shape[0], image.shape[1]
        im = None
        #establish the direction for fill
        if x_distrib > width/2:
            direction = "right"
        else:
            direction = "left"
        try:
            im = np.array(image)
            val = height - width
            fill = np.zeros((height, val))

            try:
                # fill in by direction
                if direction == "left":
                    im = np.hstack((fill, im))
                else:
                    im = np.hstack((im, fill))
            except Exception as ex:
                print(">> Exception on applying hstack : " + str(ex))

        except Exception as e:
            print(">> Exception in building the fill " + str(e))

        # if im is not None:
        # ImageNormaliser.plotData(im, "hue")
        return im

    @staticmethod
    def fill_letter_oy(image, y_distrib, original_height):
        direction = None
        im = None
        height, width = image.shape[0], image.shape[1]
        # establish the direction for fill
        if y_distrib > original_height / 2:
            direction = "bottom" #trebuie inversate
        else:
            direction = "top"

        im = np.array(image)
        val = width - height

        #treat the case where we need to add more fill lines that the actual image height
        #this way we overcome the isolation of the image
        try:
            if val > height*0.7: #if the difference between width and height is big
                fill_height = int(val/2)
                if val % 2 == 0:
                    fill1 = np.zeros((fill_height, width))
                    fill2 = np.zeros((fill_height, width))
                    # print(">>> : " + str(fill1.shape) + "  " + str(fill2.shape))

                else:
                    fill1 = np.zeros((fill_height+1, width))
                    fill2 = np.zeros((fill_height, width))

                try:
                    im = np.vstack((fill1, im))
                    im = np.vstack((im, fill2))
                except Exception as ex:
                    print(">> Exception in applying vstack : " + str(ex))

            else :
                fill = np.zeros((val, width))
                # fill in by direction
                try:
                    if direction == "top":
                        im = np.vstack((fill, im))
                    else:
                        im = np.vstack((im, fill))
                except Exception as ex:
                    print(">> Exception in applying vstack : " + str(ex))

        except Exception as e:
            print(">> Exception in building fill" + str(e))

        # if im is not None:
        #     print(">> Char shape : " + str(im.shape))
            # ImageNormaliser.plotData(im, "hue")
        return im

    #method is computing information percent in order to eroad or not and adds
    #negative space
    def square_img(self,char):

        # filling the shapes in order to distinguish letters from simple lines
        # lines from the extremities of the word can represent segmentation errors
        img = char.copy()
        temp = char.copy()
        temp = temp * 255
        temp = cv2.convertScaleAbs(temp, alpha=(255.0 / 65535.0))
        temp = temp * 255
        im2, contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(temp, contours, color=255)


        #count the distribution of whites vs blacks
        # ImageNormaliser.plotData(temp)
        black = np.count_nonzero(temp == 0)
        white = np.count_nonzero(temp > 0)
        # print(">> Black value " + str(black) + "  White value " + str(white))
        computed_percent = -1
        if black > white:
            computed_percent = int((white * 100)/ (black+white)) #+ white
            print(">>>" + str(img.shape) + "   ->" + str(computed_percent))
        if computed_percent > -1:
            if computed_percent > 35:
                kernel = np.ones((2, 2))
                img = cv2.erode(img, kernel, iterations=1)
                val = int(img.shape[0] / 4)
                img = self.add_negative_space(img, val)
            elif computed_percent > 6 and computed_percent < 35:
                val = int(img.shape[0] / 4) #discutabil
                img = self.add_negative_space(img, val)
            else:
                # #FINDME
                img = np.zeros(img.shape)
                pass


            return img
        else:
            img = np.zeros(img.shape)
            return img

    @staticmethod
    def add_negative_space(img, val):
        side_len = img.shape[0]

        fill1 = np.zeros((val, side_len))  # intended for vstack
        fill2 = np.zeros(((side_len + val*2), val))  # intended for hstack

        img = np.vstack((fill1, img))
        img = np.vstack((img, fill1))
        img = np.hstack((fill2, img))
        img = np.hstack((img, fill2))

        return img


class OutputBuilder:
    def __init__(self, path):
        self.path = path
        self.raw_img = self._get_raw_image()
        # print(self.path)

    def build_images(self):

        img = self.raw_img
        # ImageNormaliser.plotData(img)

        initial_elem = Element(img)
        names_list = list()
        line_builder = LineBuilder(initial_elem)
        lines = line_builder.elements

        line_index = 1

        for line in lines:
            word_builder = WordBuilder(line)
            words = word_builder.elements
            l_name = '_'+str(line_index)
            line_path = self.path+"\\lines\\"+l_name+".png"
            cv2.imwrite(line_path, line.image)
            names_list.append(line_path)
            word_index = 1
            for word in words:
                w_name = l_name+'_' + str(word_index)
                word_path = self.path+"\\words\\"+w_name+".png"
                cv2.imwrite(word_path, word.image)
                names_list.append(word_path)
                char_builder = CharBuilder(word)
                chars = char_builder.elements
                char_index = 1
                for char in chars:
                    c_name = w_name + '_' + str(char_index)
                    char_path = self.path+"\\chars\\"+c_name+'.png'
                    cv2.imwrite(char_path, char.image)
                    names_list.append(char_path)
                    char_index += 1
                word_index += 1
            line_index += 1

        # for name in names_list:
        #     arr = re.split('\\\|\.', name)
        #     print(str(name) + "    " + str(arr[-2]))

    def _get_raw_image(self):
        data_reader = DataToImage(self.path)
        img = data_reader.image
        img = ImageNormaliser.resize_percent(img, 40)
        img = ImageNormaliser.thresholding(img)
        return img

#prima data se incearca cu clasificator si daca nu iese se face fara
    def build_output_dict(self, clf):
        if clf is not None:
            # clf = CharClassifier("dev/tempFiles/keras_pd2_v4.h5")
            result = dict()
            count = 0
            dirs = [self.path + "\\" + 'lines',
                    self.path + "\\" + 'words',
                    self.path + "\\" + 'chars']

            for curr_dir in dirs:
                for img in os.listdir(curr_dir):
                    splitedArray = re.split('\\\|\.', img)
                    temp = InputFormatting.png_to_base64(curr_dir + "\\" + img)

                    if curr_dir is dirs[2]:
                        current_char = cv2.imread(curr_dir + "\\" + img, 0)
                        # ImageNormaliser.plotData(current_char)
                        predict = clf.predict(current_char)
                        temp = predict + temp
                    result[splitedArray[-2]] = temp
                    count += 1

            return result

        else:

            result = dict()
            count = 0
            dirs = [self.path + "\\" + 'lines',
            self.path + "\\" + 'words',
            self.path + "\\" + 'chars']

            for curr_dir in dirs:
                for img in os.listdir(curr_dir):
                    splitedArray = re.split('\\\|\.', img)
                    temp = InputFormatting.png_to_base64(curr_dir + "\\" + img)
                    if curr_dir is dirs[2]:
                        current_char = cv2.imread(curr_dir+"\\"+img, 0)
                        # ImageNormaliser.plotData(current_char)
                        temp = '0' + temp

                    result[splitedArray[-2]] = temp
                    count += 1

            return result


class CharClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load_model(model_path)

    def predict(self, img):
        labels = {1: 97, 2: 98, 3: 99, 4: 100, 5: 101, 6: 102, 7: 103,
                  8: 104, 9: 105, 10: 106, 11: 108, 12: 109, 13: 110,
                  14: 111, 15: 112, 16: 113, 17: 114, 18: 115, 19: 116,
                  20: 117, 21: 118, 22: 122}
        img = img.reshape(1, 64, 64, 1)
        predictions = self.model.predict(img)
        print(predictions)
        predicted_label = np.argmax(predictions, axis=1)
        letter = labels[int(predicted_label[0]) + 1 ]
        print(">>Litera : " + chr(letter) + "   " + str(predicted_label))
        return chr(letter)

def test(img):
    line_builder = LineBuilder(img)
    lines = line_builder.elements

    word_builder = WordBuilder(lines[0].image)
    words = word_builder.elements

    char_builder = CharBuilder(words[0].image)
    chars = char_builder.bounds

    # print(chars)


def get_elems(img):
    print(">> In get_elems")
    initial_elem = Element(img)
    clf = CharClassifier("tempFiles/keras_pd2_v4.h5")
    line_builder = LineBuilder(initial_elem)
    lines = line_builder.elements

    for line in lines:
        # ImageNormaliser.plotData(line.image)
        word_builder = WordBuilder(line)
        words = word_builder.elements
        # ImageNormaliser.plotData(line.image)
        for word in words:
            # ImageNormaliser.plotData(word.image)

            char_builder = CharBuilder(word)
            # bounds = char_builder.bounds
            # char_builder.draw_bounds(bounds)
            chars = char_builder.elements
            for char in chars:
                clf.predict(char.image)
                ImageNormaliser.plotData(char.image)


def show_projection(im, values):
    img = np.zeros((im.shape[0], im.shape[1]))

    for i in range(len(values)):
        for j in range(values[i]):
            img[i][j] = 255

    ImageNormaliser.plotData(img)


if __name__ == "__main__":
    # print(">>Start")
    # data_reader = DataToImage("tempFiles/fis.txt", "tempFiles/newImage.png")
    # image = data_reader.image
    # ImageNormaliser.plotData(image)
    # # name = 'vala'
    # # path = str(os.getcwd()) + '\\temporary\\' + name
    # #
    # #
    # # ir = OutputBuilder(path)
    # # # ir.build_images()
    # # res = ir.build_output_dict()
    # #
    # # for k, v in res.items():
    # #     print(str(k) + "   " + str(v))
    #
    # image = image_norm(image)
    # get_elems(image)
    # # test(image)
    pass

