import base64
import numpy as np
import cv2
import os


'''
    InputFormatting
    f = open('dev/temporary/fis.txt', 'w')
'''


class InputFormatting:
    def __init__(self, directory_name, data):
        self.dir_path = directory_name
        self.buffer_path = self.dir_path + '\\buffer.txt'
        self._data = data
        self.build_directories()
        self.write_to_buffer()

    def write_to_buffer(self):
        try:
            f = open(self.buffer_path, 'w')
            f.write(self._data)
            f.close()
        except Exception as e:
            print(">> Error in writing in buffer the that was received from client  : " + str(e))

    def build_directories(self):
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        dirs = [self.dir_path + '\\lines',
                self.dir_path + '\\words',
                self.dir_path + '\\chars']

        for dir in dirs:
            if not os.path.exists(dir):
                os.mkdir(dir)

    @staticmethod
    def png_to_base64(path):

        with open(path, "rb") as imageFile:
            string = base64.b64encode(imageFile.read())
        part1 = 'data:image/png;base64, '
        base64_str = string.decode("utf-8")
        result = part1 + str(base64_str)
        # print(result[:10])
        return result


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
    def __init__(self, directory_name):
        self._dir_name = directory_name
        self._buffer_path = self._dir_name + '\\buffer.txt'
        self.imageName = self._dir_name + '\\raw_img.png'
        self._base64 = self.read_from_file()
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

    def read_from_file(self):
        f = open(self._buffer_path, 'r')
        str_one = f.read()
        f.close()
        # print(">> " +str_one)
        data = self.partition(str_one)
        return data

    def base64_to_image(self):
        base64_str = self.partition(self._base64)
        binary_str = base64.b64decode(base64_str)
        with open(self.imageName, "wb") as fh:
            fh.write(binary_str)

    def _get_image(self):
        img = cv2.imread(self.imageName, cv2.IMREAD_GRAYSCALE)
        return img

    @property
    def image(self):
        self._image = self._get_image()
        return self._image

    @image.setter
    def image(self, value):
        if isinstance(value, str):
            self._image = value


def build_name_current():
    print(">> Am apelat build name ")
    buff_path = "F:/_licenta/PythonDraw/PyDraw/dev/temporary/buff.txt"
    current_number = -1
    try:
        with open(buff_path, 'r') as file:
            current_number = file.read()
            file.close()
    except Exception as e:
        print(">> Exception occured during reading naming buffer : " + str(e))
    current_number = int(current_number)
    if current_number != -1:
        current_number += 1

        try:
            with open(buff_path, 'w') as file:
                file.write(str(current_number))
                file.close()
        except Exception as e:
            print(">> Exception occured during writing current num in buffer : " + str(e))

        new_name = 'dir'+str(current_number)
        return new_name
    return None

def test():
    print(os.getcwd())
    path = "D:\_licenta\PythonDraw\PyDraw\dev\\temporary\\vala\\raw_img.png"
    path2 = os.getcwd() + "\\temporary\\vala\\raw_img.png"

    with open(path2, "rb") as imageFile:
    # with open("tempFiles/newImage.png", "rb") as imageFile:
        string = base64.b64encode(imageFile.read())
    part1 = 'data:image/png;base64, '
    base64_str = string.decode("utf-8")
    result = part1 + str(base64_str)
    print(result[:10])
if __name__ == "__main__":
    # name = "xvx"
    # data = "ahaa"
    # input_formator = InputFormatting(name, data)
    # data_reader = DataToImage(name)
    test()

