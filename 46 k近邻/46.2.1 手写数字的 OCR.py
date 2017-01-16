# coding=utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt

PATH = '../images/'


def save_npz(file_name, train, train_labels):
    np.savez(file_name, train=train, train_labels=train_labels)


def load_npz_test(file_name):
    with np.load(file_name) as data:
        print(data.files)
        train = data['train']
        train_labels = data['train_labels']
        return train, train_labels

if __name__ == '__main__':
    # img = cv2.imread(PATH)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.imread(PATH+'digits.png', 0)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)

    # Now we prepare train_data and test_data.
    train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
    test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k, 250)[:, np.newaxis]
    test_labels = train_labels.copy()

    # Initiate kNN, train the data, then test it with test data for k=1

    # knn = cv2.KNearest()
    knn = cv2.ml.KNearest_create()      # for Open-Python 3.0

    # knn.train(train, train_labels)
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    # ret, result, neighbours, dist = knn.find_nearest(test, k=5)
    ret, result, neighbours, dist = knn.findNearest(test, k=5)   # for Open-Python 3.0

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print(accuracy)

    save_npz('knn_number_data.npz', train=train, train_labels=train_labels)
    # Now load the data
    load_npz_test('knn_number_data.npz')