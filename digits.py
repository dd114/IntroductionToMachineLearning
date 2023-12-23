import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
import os
import random

import mnist_loader
from MLP import MLP

if __name__ == '__main__':

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # print(test_data[1][1])
    # print(len(test_data[0]))

    MNN = MLP([784, 30, 10], type_activation="sigmoid")
    MNN.stochastic_GD(training_data, 2, 10, 3, test_data=test_data)

    index_digit_in_array = 9
    probability = MNN.forward_propagation(validation_data[index_digit_in_array][0])
    predict_digit = np.argmax(probability)
    # print(probability)
    precise_digit = validation_data[index_digit_in_array][1];
    print(f"Predict digit = {predict_digit}")
    print(f"Precise digit = {precise_digit} with a probability of = {int(probability[precise_digit][0] * 100)} %")

    plt.imshow(validation_data[index_digit_in_array][0].reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"This number is {predict_digit} with a probability of = {int(probability[predict_digit][0] * 100)} %")
    plt.show()

