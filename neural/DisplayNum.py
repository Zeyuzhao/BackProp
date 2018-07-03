import numpy as np
import matplotlib.pyplot as plt
import itertools

from neural.main import NeuralNet
from neural.mnist_prep import MnistPrepper



def showImgs(imgSet, start, length, width, predictedY=None):
    #Dimensions of the number of subplots
    #length = 8
    #width = 8

    #Where to start within the set of images
    imgID = start

    f, axarr = plt.subplots(length, width, figsize=(6,8))

    f.suptitle("MNIST testing set from pic {0} to {1}".format(imgID, imgID + length * width - 1))
    # Extract the sets of data, X and y, 98from training data

    for i, j in itertools.product(range(length), range(width)):

        picTuple = imgSet[imgID][0]
        picLabel = imgSet[imgID][1]
        # Scale normalized tuple into byte value and then reshape into 2d
        img = (255 * picTuple).astype(int).reshape(28, 28)

        # Show images
        axarr[i, j].imshow(img)
        axarr[i, j].set_title("L: {1}| P: {2}".format(imgID, picLabel, predictedY[imgID - start] if (not predictedY is None) else "N"))
        axarr[i, j].axis('off')

        # Keep count of the current images
        imgID += 1
    # Adjust the margins so the labels don't overlap
    plt.subplots_adjust(bottom=0)

    # Finally Done!
    plt.show()

if __name__ == '__main__':
    dim = (784, 100, 10)
    network = NeuralNet(dim)

    epoch = 10
    training_data, validation_data, testing_data = MnistPrepper().processData()
    network.train(training_data, 30, 2, epoch)
    network.storeNN("../data/mnistNN{0}.pkl".format(epoch))
    predictedY = []

    l = 5
    w = 5

    correct = 0
    numItems = l * w
    for i in range(l * w):
        y = network.fowardCompute(testing_data[i][0])
        l = testing_data[i][1]
        predictedY.append(y)
        print("Labelled {0}; Computed {1}".format(l, y))
        if y == l:
            correct += 1
    print("Accuracy: {0:.2f}".format(correct/numItems))
    showImgs(testing_data, 0, l, w, predictedY)

