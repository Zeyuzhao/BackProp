import numpy as np
import matplotlib.pyplot as plt
import itertools


"""
showImgs()

Shows a grid of images, with dimensions length and width.
Optional feature of displaying the expected output of a NN

imgSet: A list of pairs of data, X and y.
X is a 784 grey scale ndarray (28 X 28 image when rescaled), y is the label
Use dataset from neuralnetworksanddeeplearning.com as an example

start: The index of imgSet to begin at
length: Length of the grid of samples
width: Width of the grid of samples
predictedY: Show the output of a neural network along with 
"""
def showImgs(imgSet, start, length, width, predictedY=None):

    #Where to start within the set of images
    imgID = start

    f, axarr = plt.subplots(length, width, figsize=(length + 3,width + 3))

    f.suptitle("MNIST testing set from pic {0} to {1}".format(imgID, imgID + length * width - 1))
    # Extract the sets of data, X and y, 98from training data

    for i, j in itertools.product(range(length), range(width)):

        picTuple = imgSet[imgID][0]
        picLabel = imgSet[imgID][1]
        # Scale normalized tuple into byte value and then reshape into 2d
        img = (255 * picTuple).astype(int).reshape(28, 28)

        # Show images
        axarr[i, j].imshow(img)
        axarr[i, j].set_title("L: {1}| P: {2}".format(imgID, picLabel, predictedY[imgID - start] if (not predictedY is None) else "N"), fontsize = 7)
        axarr[i, j].axis('off')

        # Keep count of the current images
        imgID += 1
    # Adjust the margins so the labels don't overlap
    plt.subplots_adjust(bottom=-0.2)

    # Finally Done!
    plt.show()

# if __name__ == '__main__':
#     dim = (784, 100, 100, 10)
#     network = NeuralNet(dim)
#
#     epoch = 30
#     training_data, validation_data, testing_data = MnistPrepper().processData()
#     network.train(training_data, 10, 3, epoch, testingData=testing_data)
#     network.storeNN("../data/mnistNN_L{0}_E{1}.pkl".format(network.numLayers, epoch))
#
#     l = 5
#     w = 5
#     correct = 0
#     numItems = l * w
#     predictedY = []
#
#     for i in range(l * w):
#         y = network.fowardCompute(testing_data[i][0])
#         l = testing_data[i][1]
#         predictedY.append(y)
#         print("Labelled {0}; Computed {1}".format(l, y))
#         if y == l:
#             correct += 1
#     print("Accuracy: {0:.2f}".format(correct/numItems))
#     showImgs(testing_data, 0, l, w, predictedY)

