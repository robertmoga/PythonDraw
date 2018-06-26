import cv2
import numpy as np
img = None


def readData():
    img = cv2.imread('imageToSave.png', cv2.IMREAD_GRAYSCALE)
    return img


def thresholding (img):
    ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    return thresh

def crop(img):
    x, y = 170, 110
    w, h = 300, 300
    crop_img = img[y:y + h, x:x + w]
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return crop_img

def resize(img):
    resized_image = cv2.resize(img, (64, 64))
    #res = cv2.resize(img, None, fx=1/10, fy=1/10, interpolation=cv2.INTER_CUBIC)
    return resized_image

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def plotData(img):
    winname = "image"
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
    cv2.imshow(winname, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def check_thickness():
    path_1 = "F:\_Licenta\PythonDraw\PyDraw\pd_dataset\images\\b_648.png"
    path_2 = "F:\_Licenta\PythonDraw\PyDraw\dev\pd2_dataset\images\\c_87.png"

    img1 = cv2.imread(path_1, 0)
    img2 = cv2.imread(path_2, 0)
    img1 = resize(img1)
    ker = np.ones((2, 2))
    img1 = cv2.erode(img1, ker,cv2.MORPH_ERODE)
    plotData(img1)
    count = 0
    for i in range(img1.shape[1]):
        if img1[20][i] > 0:
            count += 1
    print(count)
    count = 0
    for i in range(img2.shape[1]):
        if img2[22][i] > 0:
            count += 1
        img2[22][i] = 200
    print(count)

    plotData(img2)


if __name__ == "__main__" :
    print('>>Start')
    # a = readData()
    # a = thresholding(a)
    # a = crop(a)
    # a = resize(a)
    # a = rotateImage(a, 270)
    # plotData(a)
    check_thickness()

    print('>>End')