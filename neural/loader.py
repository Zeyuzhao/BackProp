import numpy as np
from neural.main import NeuralNet
from neural.mnist_prep import MnistPrepper

stored = NeuralNet(fileName="../data/mnistNN10.pkl")
training_data, validation_data, testing_data = MnistPrepper().processData()

correct = 0
numItems = 1000

for i in range(numItems):
    l = testing_data[i][1]
    y = stored.fowardCompute(testing_data[i][0])
    print("Label: {0}, Computed: {1}".format(l, y))
    if y == l:
        correct += 1
print("Accuracy: {0:.2f}".format(correct/numItems))

