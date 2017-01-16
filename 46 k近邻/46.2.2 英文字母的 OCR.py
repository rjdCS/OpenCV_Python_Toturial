


import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

PATH = '../images/'

# Load the data, converters convert the letter to a number
data= np.loadtxt(PATH+'letter-recognition.data', dtype='float32', delimiter=',',
                converters= {0: lambda ch: ord(ch)-ord('A')})

# split the data to two, 10000 each for train and test
train, test = np.vsplit(data, 2)

# split trainData and testData to features and responses
responses, trainData = np.hsplit(train, [1])
labels, testData = np.hsplit(test, [1])

# Initiate the kNN, classify, measure accuracy.
# knn = cv2.KNearest()
knn = cv2.ml.KNearest_create()      # for Open-Python 3.0
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)

# ret, result, neighbours, dist = knn.find_nearest(testData, k=5)
ret, result, neighbours, dist = knn.findNearest(testData, k=5)   # for Open-Python 3.0

correct = np.count_nonzero(result==labels)
accuracy = correct*100.0/10000
print(accuracy)

# print(sys.path)
# from .tools import save_npz, load_npz_test
# save_npz('knn_alphabet_data.npz', train, responses)