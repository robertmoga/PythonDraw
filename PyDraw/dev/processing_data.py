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
    f = open('tempFiles/fis1.txt', 'r')
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
    err = 20 #error for croping

    new_img = None
    for cnt in contours:
        # cnt = i
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(img, [box], -1, (0, 0, 255), 2)

        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        new_img = img[y-err:y + h + err, x-err:x + w + err]
        img = new_img

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def test_morpho(img):
    backup_img = img # this wont be processed
    #mandatory to do closing first because of the open letters
    kernel = np.ones((4, 8))
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # plotData(dilation, 'dilation')

    img = hull_test(img)
    cv2.imwrite('hue.jpg', 255-img)

    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # closing[10][10] = (200,100,0)

    new_img = colors_conex_alg(img)
    plotData(new_img , winname='closing')

def colors_conex_alg(img):

    height, width = img.shape[0], img.shape[1]
    im2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    arr = dict()
    global_tag = 1


    #iterate through pixels
    for i in range(height):
        for j in range(width):
            near_tags = set()
            current_tag = -1
            if im2[i][j] == 0:
                # check for the tags of neighbors
                if (i - 1, j) in arr:
                    # print(">> Am intrat in if nord : " + str(i-1) + "," + str(j))
                    near_tags.add(arr[(i - 1, j)])
                if (i, j + 1) in arr:
                    # print(">> Am intrat in if est : " + str(i) + "," + str(j+1))
                    near_tags.add(arr[(i, j + 1)])
                if (i + 1, j) in arr:
                    # print(">> Am intrat in if sud : " + str(i+1) + "," + str(j))
                    near_tags.add(arr[(i + 1, j)])
                if (i, j - 1) in arr:
                    # print(">> Am intrat in if vest : " + str(i) + "," + str(j - 1))
                    near_tags.add(arr[(i, j - 1)])

                # label the current pixel
                if len(near_tags) == 1:
                    # print('Am intart unde trebe')
                    current_tag = list(near_tags)[0]
                elif len(near_tags) > 1:

                    near_tags = sorted(near_tags)
                    current_tag = list(near_tags)[0]
                    print('Am intart pe belea ' + str(current_tag) + "  " + str(near_tags))

                    # manage the case with two components that become one
                elif len(near_tags) == 0:
                    print('Am adaugat un nou tag')
                    current_tag = global_tag
                    global_tag += 1

                arr[(i, j)] = current_tag

        cv2.imshow('hue', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    print(">> gt : " + str(global_tag))
    import random
    colors = list()
    for tag in range(global_tag):
        r = random.randrange(1, 255)
        g = random.randrange(1, 255)
        b = random.randrange(1, 255)
        # colors.append((b,g,r))
        for i in range(height):
            for j in range(width):
                if arr[i][j] == tag:
                    img[i][j] = (b, g, r)

    plotData(img, 'hue')


    return img

def conex2(img):
    img = hull_test(img)
    im2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = img.shape[0], img.shape[1]
    print("Height : " + str(height) + "  width : " + str(width))

    new_tag = 1
    arr = list()

    for i in range(height):
        arr.append(i)
        arr[i]=list()
        for j in range(width):
            if im2[i][j] == 0:
                arr[i].append(-1)
            else :
                arr[i].append(0)

    for i in range(height):
        for j in range(width):
            if arr[i][j] == -1:
                near_labels = list()
                current_label = -1
                #nord
                if i-1 >= 0 :
                    # print("Intru pe nord")
                    if arr[i-1][j] > 0:
                        near_labels.append(arr[i-1][j])

                #sud
                if i+1 < height:
                    # print("Intru pe sud")
                    if arr[i+1][j] > 0:
                        near_labels.append(arr[i+1][j])

                #est
                if j+1 < width:
                    # print("Intru pe est")
                    if arr[i][j+1] > 0:
                        near_labels.append(arr[i][j+1])

                #vest
                if j-1 >= 0:
                    # print("Intru pe vest  : " + str(i) + "  " + str(j-1) )
                    if arr[i][j-1] > 0:
                        # print(">> vest : trec de conditie ")
                        near_labels.append(arr[i][j-1])

                if len(near_labels) >0:
                    current_label = sorted(near_labels)[0]
                    arr[i][j] = current_label
                else :
                    current_label = new_tag
                    new_tag += 1
                    arr[i][j] = current_label

    print(">> gt : " + str(new_tag))
    import random
    colors = list()
    for tag in range(new_tag):
        r = random.randrange(1,255)
        g = random.randrange(1,255)
        b = random.randrange(1,255)
        # colors.append((b,g,r))
        for i in range(height):
            for j in range(width):
                if arr[i][j] == tag:
                    img[i][j] = (b, g, r)



    plotData(img, 'hue')

    # print(str(arr))

def test():

    for i in range(10):
        if i == 5:
            i = 8
        print(str(i))


'''
Conex3 will return the image with filled in forms 
return the image
'''
def conex3(img):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # final = cv2.drawContours(img, contours, 0, (0, 0, 255), thickness=3)
    cv2.fillPoly(img, contours, color=(255, 255, 255))

    # plotData(img, 'hue')
    kernel1 = np.ones((30, 2))
    kernel2 = np.ones((3, 3))


    dilation = cv2.dilate(img, kernel2, iterations=1)
    # plotData(dilation, 'dilation')
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel1)
    # plotData(closing, 'closing')
    erosion = cv2.erode(closing, kernel2, iterations=2)
    # plotData(erosion, 'erosion')

    # plotData(closing, 'closing')

    return erosion

def getHistogramValues(img):

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

def getBoundariesFromHistogram(white_values):

    boundaries = list()
    index = -1

    def trigger(pos, vec):
        count = 0
        for i in range(pos, len(vec)-1):
            if abs(vec[i] - vec[i + 1]) > 4:
                return count
            count += 1

        return -1

    for i in range(len(white_values)-1):
        if index > -1 and i < index:
            continue
        val1 = white_values[i]
        val2 = white_values[i+1]
        if i>1:
            val3 = white_values[i-1]

        if ( white_values[i] < 10 and white_values[i] > 0 ) and \
                (white_values[i+1] < 10 and white_values[i+1]>0) :
            if abs(white_values[i] - white_values[i + 1]) < 10:

                count = trigger(i, white_values)
                if count != -1:
                    boundaries.append(int(count/2 + i))
                    index = i + count


    return boundaries

def drawBounds(img, bounds):
    height, width = img.shape[0], img.shape[1]

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for y in bounds:
        for x in range(height):
            img[x][y] = (0,255,0)

    return img

def doStuff():

    img = prepare_img()
    img = hull_test(img)
    img_temp = conex3(img)

    whites  = getHistogramValues(img_temp)
    bounds = getBoundariesFromHistogram(whites)

    print(bounds)
    plotData(img_temp, winname='conex')
    img = drawBounds(img, bounds)
    plotData(img, winname='original')

if __name__ == "__main__":
    print(">> Start  ")
    timer = time.time()
    img = prepare_img()
    # haar_test(img)
    # res = hull_test(img)
    # plotData(res, winname='hull')
    # test_morpho(img)
    # test()
    # conex2(img)
    # temp = conex3(img)
    # plotData(temp, 'temp')
    doStuff() #woooorking

    print(">> Time elapsed : " + str(time.time()-timer))