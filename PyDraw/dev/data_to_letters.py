from distutils.command.config import config

import cv2
import base64
import numpy as np
# import threading
# import matplotlib.pyplot as pt
# import pandas as pd
# from sklearn import ensemble
import time
import os


'''
    Reads data from file where the server that is buffering it
    Normalises data to the desierd shape 
            - thresholding
            - resize : in order to reduce the area for computation : percent + contours
            - does morphological transformations
    Realises histogram-like measurments for whites on y axis
    Analyses the whites values
    Get the bounds between letters - Done
    
    Normalise letters 
    Return letters 28x28   
'''

'''
    DataToImage class
        2 parameters : 
                dataPath -> the relative path of the buffer file
                imagePath -> relative path of the new image (+naming)

        - takes base64 data from buffer file, given as parameter at dataPath. 
                Did in : read_from_file() -> Returns a string
        - the string of data is stored in _base64.
        - turns the base64 into binary and writes into file with .png extension.
                Did in : base64_to_image() -> Does not return anything
        - we read with openCV from .png file and return a nparray-image via _image prop.
                Did in get_image() -> returns a image/nparray
'''


class DataToImage:
    def __init__(self, dataPath, imagePath):
        self._base64 = self.read_from_file(dataPath)
        self.imageName = imagePath
        self.base64_to_image()
        self._image = None

    @property
    def base64(self):
        return self._base64

    @base64.setter
    def base64(self, value):
        if isinstance(value, str):
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
        if isinstance(value, str):
            self._image = value


'''
    ImageNormaliser class
        1 parameter : raw_image -> a nparray/ image
        
        Its job is to prepare the image to the needed form and shape
        
        Methods : 
            - plotData() -> able to plot an image in a window : it will be moved in a separate class
            - thresholding() -> thresholds the image given as parameter : everything that is higher than 0
                                becomes 255
            - resize_percent() -> able to resize the image given as param at the percent of dimensions given
                                    as well as param
            - resize_specific_dim() -> takes one paramater for dimension and resizes as square
            - find_contours() -> takes image, finds countours and expands the area with a parameter, err, 
                                convenient set a 20px
            - apply_processing() -> applies the transformations above and returns an image ready for analysis
            
            User can retrive the needed image from 'image' property that applies apply_processing()
'''
class ImageNormaliser:
    def __init__(self, raw_img=None):
        self.raw_image = raw_img
        self._image = self.image

    @staticmethod
    def plotData(image, winname="win"):

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
        img = self.resize_percent(img, 50)
        img = self.thresholding(img)
        img = self.find_contours(img)
        return img

    @property
    def image(self):
        self._image = self.apply_processing()
        return self._image

    @image.setter
    def image(self, value):
        if isinstance(value, np.ndarray):
            self._image = value



'''
    ImageAnalyser class
        1 parameter -> raw_image : the image in the needed ready for analysing
        
        Methods : 
            - execute_morpho() -> dies morphological transformations on raw_image 
                                    via morpho propand stores the result into self.image_morpho
            - get_histogram_values() -> analyse the image_morpho and returns an array of white values computed on oY axis
            - get_boundaries_from_histogram() -> returns an array of the boundaries between letters as a series 
                                                    of Y coordinates, takes whites vector as a param
            - analyse_image() -> runs the two methods above and puts them together
            - draw_bounds() -> draws the bounds for demo or verification
            
            The main output of the class is the self.bounds array that will be used in determinig the letters and normalise them
                                             
'''
class ImageAnalyser:

    def __init__(self, raw_image=None):
        self.raw_image = raw_image
        self.image_morpho = self.morpho
        self._bounds = self.bounds


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
        if isinstance(value, np.ndarray):
            self.image_morpho = value

    @property
    def bounds(self):
        self._bounds = self.analyse_image()
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        if isinstance(value, list):
            self._bounds = value

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

    def drawBounds(self):
        img = self.raw_image
        height, width = img.shape[0], img.shape[1]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # self.__get_mean_height()

        for y in self.bounds:
            for x in range(height):
                img[x][y] = (0, 255, 0)

        h = int(self.raw_image.shape[0] * 0.4)
        print(str(self.raw_image.shape[0] ) + "   " + str(h))

        for x in range(width):
            img[h][x] = (0,255,0)
        norm = ImageNormaliser(self.raw_image)
        norm.plotData(img, winname="hue")

        return img

    def __check_bound(self, bound, height_allowance):
        pass
        # height = self.raw_image.shape[0] * height_allowance
        # for i in range(self.raw_image.shape[0]):
        #     if self.raw_image[i][bound] > 0:
        #         if i > height:
        #             print(i)
        #             return False
        #         else:
        #             return True
    '''
        Calculez mai bine inaltimea acceptata:
            Calculez in 6 puncte, reprezentate de width*procent,
            nivelul ce alb, si fac media lor,
            inmultesc media cu un procent pentru a ridica inaltimea.
            
            Asta va fi pragul peste acre accept cross fara bound
    '''

    def __get_mean_height(self):
        pass
        # img = self.raw_image
        # img = img[:, 10:-10]
        # positions = [0.25, 0.40, 0.5, 0.60, 0.75]
        # white_levels = list()
        # for pos in positions:
        #     white_levels.append(0)
        #     j = int(img.shape[1]*pos)
        #     for i in range(img.shape[0]):
        #         if img[i][j] == 0:
        #             white_levels[positions.index(pos)] += 1
        # print(white_levels)

    def analyse_image(self):
        # ImageNormaliser.plotData(self.image_morpho)
        white_values = self.get_histogram_values()
        bounds = self.get_boundaries_from_histogram(white_values)

        # i = 0
        # while i < len(bounds)-1:
        #     if self.__check_bound(bounds[i], 0.4):
        #         bounds.remove(bounds[i])
        #         i -= 1
        #     i += 1

        return bounds

'''
    CharSynthesizer class 
        2 parameters : the normalised image and the bounds array
        Methods : callable is only : 
                    - normalise_letters() that takes the bounds and image input and returns a list of
                        np.array letters 100x100 and can be accesed by property 'letters'
        Other methods : 
            - define_letters : split the normalised image by bounds and return a list of np.array of letters
            - crop_letter() : has a letter as input an computes the contour points distribution on
                x and y axis and then fills with negative space depending on the distribution and then
                makes the images square
            - fill_letter_ox, and fill_letter_oy fills the letter on axis depending on x or y distribution
            - square_img : computes the rate of whites reported to blacks and fills with a negative frame
                            depending on the result of distribution
            - add_negative_space : adding the needed space on the sides of the image 
'''

class CharSynthesizer():

    def __init__(self, img, bounds):
        self.image = img
        self.bounds = bounds
        self._letters = self.letters

    @property
    def letters(self):
        self._letters = self.normalise_letters()
        return self._letters

    @letters.setter
    def letters(self, value):
        if isinstance(value, list):
            self._letters = value


    def define_letters(self):
        height, width = self.image.shape[0], self.image.shape[1]
        bounds = self.bounds
        bounds = [0] + bounds + [width]

        letters = list()
        for elem in range(len(bounds)):
            char = list()
            if elem == 0:
                continue
            else:
                char = self.image[:, bounds[elem-1]:bounds[elem]]
                letters.append(char)

        return letters

    #singura metoda apelabila
    def normalise_letters(self):
        letters = self.define_letters() #le imparte
        return_list = list()
        # aici putem verifica procentul de alb din imagine si sa dam pop elementului respectiv
        for l in letters:
            try:

                img = self.crop_letter(l) # le face patrate si cu negative space
                img = ImageNormaliser.resize_specific_dim(img, 64)
                #aici putem forma obiecte de tip element : FINDME

                #check if the image has info or not
                #in the case that theimage had less then 15% info, all of it was deleted
                if np.count_nonzero(img > 0):
                    ImageNormaliser.plotData(img)
                    return_list.append(img)
            except Exception as e:
                print(">> Exception during the iteration through letters " + str(e))

        return return_list


    def crop_letter(self, img):
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

        if im is not None:
            print(">> Char shape : " + str(im.shape))
            # ImageNormaliser.plotData(im, "hue")
        return im

    #method is computing information percent in order to eroad or not and adds
    #negative space
    def square_img(self,char):

        img = char
        #count the distribution of whites vs blacks

        black = np.count_nonzero(img == 0)
        white = np.count_nonzero(img > 0)
        # print(">> Black value " + str(black) + "  White value " + str(white))

        computed_percent = -1
        if black > white:
            computed_percent = int((white * 100)/ black)

        if computed_percent > -1:
            if computed_percent > 35:
                kernel = np.ones((2, 2))
                img = cv2.erode(img, kernel, iterations=2)
                val = int(img.shape[0] / 4)
                img = self.add_negative_space(img, val)

            elif computed_percent > 15 and computed_percent < 35:
                val = int(img.shape[0] / 4) #discutabil
                img = self.add_negative_space(img, val)
            else:
                # fill the image with zeros, in case of having less than 15% info
                img = np.zeros(img.shape)

        # ImageNormaliser.plotData(img, "char")

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

def teste():

    a = np.ones((1, 10))
    b = np.ones((1,4))
    c = np.zeros((1,6))

    hue = np.hstack((a, c))
    hue = np.hstack((hue, b))

    # print(hue)

    val = np.count_nonzero(hue == 0)
    print(val)

'''
    VectorNormaliser class reduces all of the intermediate values between 0 and 255
    and also make the values int
'''
class VectorNormaliser:
    def __init__(self, images):
        self.images = images

    def normalise(self):

        for i in range(len(self.images)):

            img = self.images[i]
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            img = img.astype('int')
            self.images[i] = img

        return self.images

class OutputWriter:

    def __init__(self, images, output_path):
        self.output_file = output_path
        self.images = images



    def init_output_file(self):
        open(self.output_file, 'w').close()

    @staticmethod
    def build_line(img):
        letter = str()

        for line in img:
            for elem in line:
                try:
                    letter = letter + str(elem) + ","
                except Exception as e:
                    print("Exception in building a new text : " + str(e))

        # ImageNormaliser.plotData(img, "hue")
        return letter[0:-1]

    def write(self):
        with open(self.output_file, 'a') as f:
            for letter in self.images:
                normalised_letter = self.build_line(letter)
                print(normalised_letter)
                f.write(normalised_letter+'\n')
                # f.write('\n')
            f.close()


if __name__ == "__main__":

    obj = DataToImage("tempFiles/fis1.txt", "tempFiles/newImage.png")
    image = obj.image

    norm = ImageNormaliser(image)
    img = norm.image

    analyser = ImageAnalyser(img)
    print(">> " + str(analyser.bounds))
    im2 = analyser.drawBounds()
    ImageNormaliser.plotData(im2)
    char = CharSynthesizer(img, analyser.bounds)
    letters = char.letters

    for elem in letters:
        ImageNormaliser.plotData(elem, 'hue')
    #
    # out_file_path = 'F:\Python\learn_keras\letters_from_pd\output1.txt'
    #
    # out_norm = VectorNormaliser(letters)
    # letters = out_norm.normalise()
    #
    # ow = OutputWriter(letters, out_file_path)
    # ow.init_output_file()
    # ow.write()
    # teste()

