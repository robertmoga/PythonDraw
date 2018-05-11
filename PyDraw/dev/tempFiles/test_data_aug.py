import keras
from matplotlib import pyplot as plt
import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import cv2
from scipy import misc, ndimage

def plots(img, figsize=(30,30), rows=1, interp=False, titles=None):
    if type(img[0]) is np.ndarray:
        img = np.array(img).astype(np.uint8)
        # if(img.shape[-1] != 0):
        #     img = img.transpose((0))
    f = plt.figure(figsize = figsize)
    cols = len(img)//rows if len(img) % 2 == 0 else len(img)//rows+1

    for i in range(len(img)):
        sp = f.add_subplot(rows,cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(img[i], interpolation= None if interp else None)
        print(img[i])
        plt.show()

gen = ImageDataGenerator(rotation_range=10)

path = "D:\_licenta\PythonDraw\PyDraw\pd_dataset\images\\a_5.png"

im = cv2.imread(path, 1)

image = np.expand_dims(im, 0)
# image = np.expand_dims(ndimage.imread(path),0)
# plt.imshow(image[0])
# plt.show()

aug_iter = gen.flow(image)
aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]

print(aug_images[0].shape)
# plt.imshow(aug_iter[0])
# plt.show()

# plots(aug_images, rows = 2)



