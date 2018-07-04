import numpy as np

from neural.main import *


stored = NeuralNet(fileName="../data/mnistNN_L4_E30.pkl")
training_data, validation_data, testing_data = MnistPrepper().processData()

numCorrect = 0

length = 8
width = 8

start = 200
numItems = length * width
yArray = []
for i in range(numItems):
    imgID = start + i
    l = testing_data[imgID][1]
    y = stored.fowardCompute(testing_data[imgID][0])
    yArray.append(y)

    c = True
    if y == l:
        numCorrect += 1
    else:
        c = False
    print("Label: {0}, Computed: {1}, Correct: {2}".format(l, y, c))

print("Accuracy: {0:.4f}".format(numCorrect / numItems))
print(len(yArray))
print(stored.sizes)

if numItems <= 100:
    showImgs(testing_data, start, length, width, predictedY=yArray)

