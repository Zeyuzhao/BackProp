from neural.main import NeuralNet
from neural.mnist_prep import MnistPrepper

import numpy as np

dim = (784, 100, 10)
network = NeuralNet(dim)

training_data, validation_data, testing_data = MnistPrepper().processData()
network.train(training_data, 30, 3, 1)

for i in range(10):
    y = network.fowardRun(training_data[i][0])
    print("Computed {0}; Labelled {1}".format(np.argmax(y), np.argmax(training_data[i][1])))
