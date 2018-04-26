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
    resized_image = cv2.resize(img, (28, 28))
    #res = cv2.resize(img, None, fx=1/10, fy=1/10, interpolation=cv2.INTER_CUBIC)
    return resized_image

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def revertColors(img):
    return (255-img)


def plotData(img):
    winname = "image"
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
    cv2.imshow(winname, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#does all the needed processing
#made to be imported in other files
def doMagic():
    a = readData()
    a = thresholding(a)
    a = crop(a)
    a = resize(a)
    a = rotateImage(a, 270)
    a = revertColors(a)
    return a

if __name__ == "__main__" :
    print('>>Start')
    a = readData()
    a = thresholding(a)
    a = crop(a)
    a = resize(a)
    a = rotateImage(a, 270)
    a=revertColors(a)
    plotData(a)

    print('>>End')