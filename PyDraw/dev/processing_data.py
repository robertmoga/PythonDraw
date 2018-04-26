import cv2
import base64
import numpy as np
import threading
import matplotlib.pyplot as pt
import pandas as pd
from sklearn import ensemble
import time

'''
-> Reads from the file that the client uses as a buffer and saves the data as an image
-> Manage the naming
-> Process the data into the needed shape
-> Scenario 1 : Test Haar
-> Test Hull
-> Scenario 2 : Test RandomForest
'''
#Constants
COMMAND = ""

# does not need to be called
def read_data_from_file():
    f = open('tempFiles/fis.txt', 'r')
    strOne = f.read()
    f.close()
    # print(">> " +strOne)
    data = partition(strOne)
    # print(">> " + strOne)
    return data


# does not need to be called
def partition(strOne):
    if ',' in strOne:
        strOne = strOne.partition(",")[2]
        pad = len(strOne) % 4
        strOne += "=" * pad
        return strOne
    else:
        return strOne

#manages the .txt to .png process
def save_base64_to_img():
    data = read_data_from_file()
    base64_str = partition(data)
    binary_str = base64.b64decode(base64_str)
    # print(b)
    # Here comes the naming solution
    with open("tempFiles/imageToSave.png", "wb") as fh:
        fh.write(binary_str)


# following methods are processing the image
def read_image():
    img = cv2.imread('tempFiles/imageToSave.png', cv2.IMREAD_GRAYSCALE)
    return img


def thresholding (img):
    ret, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    return thresh


def resize_specific_dim(img, val):
    resized_image = cv2.resize(img, (val, val))
    #res = cv2.resize(img, None, fx=1/10, fy=1/10, interpolation=cv2.INTER_CUBIC) #for scaling up
    return resized_image

def resize_percent(img, val):
    new_width = int(img.shape[1] * val/100)
    new_height = int(img.shape[0] * val/100)
    dim = (new_width, new_height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    return resized

def plotData(img, winname):
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
    cv2.imshow(winname, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# callable method
def prepare_img():
    save_base64_to_img()
    img = read_image()
    img = thresholding(img)
    img = resize_percent(img, 50)

    return img


def haar_test(img):
    char_clf = cv2.CascadeClassifier('tempFiles/xml/cascade1.xml')#from my laptop
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    char = char_clf.detectMultiScale(img)
    print(">> Char : " + str(char))
    for (x, y, w, h) in char:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plotData(img, winname='Haar')


def hull_test(img):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = [cv2.convexHull(c) for c in contours]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # final = cv2.drawContours(img, hull, -1, (200, 0, 0), thickness=2)
    err = 2 #error for croping

    new_img = None
    for cnt in contours:
        # cnt = i
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(img, [box], -1, (0, 0, 255), 2)

        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        new_img = img[y-err:y + h + err, x-err:x + w + err]
        img = new_img


    return img

def test_morpho(img):
    backup_img = img # this wont be processed
    #mandatory to do closing first because of the open letters
    kernel = np.ones((4, 8))
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # plotData(dilation, 'dilation')

    img = hull_test(img)

    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # closing[10][10] = (200,100,0)

    new_img = colors_conex_alg(img)


    plotData(new_img , winname='closing')


def colors_conex_alg(img):

    height, width = img.shape[0], img.shape[1]
    im2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # print(int(im2[1][1]))
    queue = list()
    dim = height*width

    while len(queue) < dim :
        for i in 


    return img

if __name__ == "__main__":
    print(">> Start  ")
    timer = time.time()
    img = prepare_img()
    # haar_test(img)
    # res = hull_test(img)
    # plotData(res, winname='hull')
    test_morpho(img)


    print(">> Time elapsed : " + str(time.time()-timer))