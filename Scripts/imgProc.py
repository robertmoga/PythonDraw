import cv2
import numpy as np
import matplotlib.pyplot as plt

def func1():
    """
        load a image in grayscale and plot it
    """
    img = cv2.imread('images/logo.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('images/logoGray.png', img)

def func2():
    img = cv2.imread('images/logo.png', 0) # cv2.IMREAD_GRAYSCALE
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.plot([50,100], [80,100], 'c', linewidth=5)
    plt.show()

'''
opeen a window with data from webcam0, and then save it.
'''

def video1():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('images/video.avi',fourcc,20.0,(640,480))

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray)
        cv2.imshow('frame', frame)
        cv2.imshow('gray', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('everything executed')

if __name__ == "__main__":
    video1()


