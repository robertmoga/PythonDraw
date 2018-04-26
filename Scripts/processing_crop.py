import cv2
import numpy as np
import threading
import time

def readImage():
    img = cv2.imread('test.png',0)
    return img


def plotImage(img):
    window_name = "Scripts/ developing crop"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 40, 30)  # Move it to (40,30)
    cv2.imshow(window_name, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def revertColors(img):
    return (255-img)


def thresholding(img):
    ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    return thresh

#roi = img[100:150, 100:150]
#img[250:300 , 250:300] = roi

def iterate(img, start, end, height ):

    flag = False
    for i in range(0, height):
        for j in range(start, end):
            if img[i, j] == 0:
                flag = True
                print(str(start) + ' am intrat in for')
                img[i, j] = 100
    if flag:
        print('>>> returnez img ' + str(start) + " <-> " + str(end))
        return 1
        #return img
    else:
        print('>>> returnez N0ne')
        return None

def splitImage(img):
    img = thresholding(img)
    img = revertColors(img)

    height, width = img.shape
    split = int(width/2) #split the image by half of width

    t1 = threading.Thread(target=iterate, args=(img, 0, split-1, height))
    t2 = threading.Thread(target=iterate, args=(img, split, width-1, height))
    imgT1 = t1.start()
    imgT2 = t2.start()

    t1.join()
    t2.join()

    print(">> sleep started")
    time.sleep(2)
    print(">> sleep ended")

    #here join the two threads

    print(imgT1)
    print("-"*20)
    print(imgT2)

    # if imgT1 is not None:
    #     print(">> t1 e bun")
    #     return imgT1
    # elif imgT2 is not None:
    #     print(">> t2 e bun")
    #     return imgT2
    # else:
    #     print(imgT2)
    #     print(">> eroare server ")
    #return img




if __name__ == "__main__":
    print(">>Start")
    img = readImage()
    img = splitImage(img)
    #plotImage(img)

    print(">>End")