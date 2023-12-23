import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
import os
import random

import mnist_loader
from MLP import MLP

def border_selection(picture_name):

    img = np.asarray(Image.open(picture_name).convert('L'))
    # plt.imshow(img, cmap='gray')
    # plt.show()

    # Gx + j*Gy

    scharr = np.array([[-3 - 3j, 0 - 10j, +3 - 3j],
                       [-10 + 0j, 0 + 0j, +10 + 0j],
                       [-3 + 3j, 0 + 10j, +3 + 3j]])

    # laplacian= np.array([[0, -1, 0],
    #                     [-1, 4, -1],
    #                     [0, -1, 0]])

    grad = signal.convolve2d(img, scharr, boundary='symm', mode='same')

    # plt.imshow(np.abs(grad), cmap='gray')
    # plt.show()

    return np.abs(grad)

if __name__ == '__main__':
    MNN = MLP([227 * 227, 30, 20, 10, 2], type_activation="sigmoid")

    plt.imshow(border_selection(f"data/concrete/Positive/00001.jpg"), cmap=plt.cm.binary)
    plt.show()

    plt.imshow(np.asarray(Image.open(f"data/concrete/Positive/00002.jpg").convert('L')), cmap=plt.cm.binary)
    plt.show()


    training_data1 = list([tuple((np.reshape(np.asarray(Image.open(f"data/concrete/Negative/{picture}").convert('L')), (-1, 1)), np.array([[1], [0]])))
                     for picture in os.listdir("data/concrete/Negative")[:1000]])

    training_data2 = list([tuple((np.reshape(np.asarray(Image.open(f"data/concrete/Positive/{picture}").convert('L')), (-1, 1)), np.array([[0], [1]])))
                     for picture in os.listdir("data/concrete/Positive")[:1000]])

    training_data = training_data1 + training_data2

    random.shuffle(training_data)
    n_test = 500

    test_data = training_data[-n_test:]
    test_data = [(test_data[i][0], np.argmax(test_data[i][1])) for i in range(n_test)]

    MNN.stochastic_GD(training_data[:-n_test], 3, 10, 0.5, test_data=test_data)
    # MNN.forward_propagation(training_data[0][0])
