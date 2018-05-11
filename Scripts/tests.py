import threading
import cv2
import ProcessRawData as prd
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn import ensemble
import time
import random




def init():
    data = prd.doMagic()
    data = prep_array(data, 28)
    return data

def prepare_dataset():
    train = pd.read_csv("D:\Python\datasets\emnist\emnist-letters-train.csv").as_matrix()
    test = pd.read_csv("D:\Python\datasets\emnist\emnist-letters-test.csv").as_matrix()
    train_x, train_y = list(), list()
    for i in range(0, len(train[:10000])):
        train_x.append(train[i][1:])
        train_y.append(train[i][0])

    test_x, test_y = list(), list()
    for i in range(0, len(test[:10000])):
        test_x.append(train[i][1:])
        test_y.append(train[i][0])

    #data = list(train_x, train_y , test_x, test_y)
    return train_x, train_y #, test_x, test_y


def prepare_pd_dataset():
    data = pd.read_csv("D:\_licenta\PythonDraw\PyDraw\pd_dataset\pd_lettersab.csv").as_matrix()
    DATA_SPLIT = 7
    train_x, train_y = list(), list()
    test_x, test_y = list(), list()
    data_x, data_y = list(), list()

    for i in range(0, len(data[:])):
        data_x.append(data[i][1:])
        data_y.append(data[i][0])

    train_index = random.sample(range(len(data_x)), len(data_x) // DATA_SPLIT)
    test_index = [i for i in range(len(data_x)) if i not in train_index]

    train_x = [data_x[i] for i in train_index]
    train_y = [data_y[i] for i in train_index]

    test_x = [data_x[i] for i in test_index]
    test_y = [data_y[i] for i in test_index]

    tr_x = np.array(train_x)
    tr_y = np.array(train_y)
    te_x = np.array(test_x)
    te_y= np.array(test_y)

    print(str(tr_x.shape) + "  " + str(tr_y.shape) + "  " +
          str(te_x.shape) + "  " + str(te_y.shape))

    #data = list(train_x, train_y , test_x, test_y)
    return train_x, train_y, test_x, test_y


def predict_pd_dataset():
    start_time = time.time()
    train_x, train_y, test_x, test_y = prepare_pd_dataset()

    print("Time preparing data  : " + str(time.time() - start_time))

    start_time = time.time()
    classifier = ensemble.RandomForestClassifier()
    classifier.fit(train_x, train_y)
    print("Time training : " + str(time.time() - start_time))

    start_time = time.time()
    score = classifier.score(test_x, test_y)

    print("Score : %.2f  " % (score * 100))

    print("Time test : " + str(time.time() - start_time))


def predict():
    img = init()
    img = np.array(img)
    img = np.reshape(img, -1, 1)

    a = np.array([img])
    # a = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,4,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,9,32,37,39,82,124,77,8,0,0,0,0,0,0,0,0,0,0,0,0,0,2,9,32,37,37,39,82,139,204,215,217,233,249,218,90,7,0,0,0,0,0,2,4,4,4,5,21,34,82,139,204,217,217,217,233,250,254,254,254,254,255,253,200,32,0,0,0,0,7,76,125,127,127,129,172,204,233,250,254,254,254,254,254,254,250,250,250,252,254,252,172,21,0,0,0,0,19,151,215,217,217,217,233,245,252,254,254,254,252,250,250,245,222,217,217,236,254,250,139,9,0,0,0,0,20,168,249,254,254,254,254,252,245,222,217,215,172,129,127,114,51,39,41,146,249,218,77,2,0,0,0,0,2,67,170,215,217,217,204,172,115,51,37,37,21,5,4,4,0,0,9,135,203,90,8,0,0,0,0,0,0,2,21,37,37,37,32,21,4,0,0,0,0,0,0,0,0,0,27,145,58,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,26,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    # a = np.reshape(a, -1, 1)

    start_time = time.time()
    train_x, train_y = prepare_dataset()
    print("Time 1 : " + str(time.time()-start_time))

    start_time = time.time()
    classifier = ensemble.RandomForestClassifier()
    classifier.fit(train_x, train_y)
    print("Time 2 : " + str(time.time()-start_time))

    start_time = time.time()
    result = classifier.predict(a)
    print(">> Result : " + str(result))
    print("Time 3 : " + str(time.time()-start_time))


def prep_array(arr, num):
    l = list()
    for i in range(0,num):
        for j in range(0, num):
                l.append(arr[i][j])

    return l


if __name__ == "__main__":
    # predict() #in case of testing o a single image
    # prepare_pd_dataset()
    predict_pd_dataset()


