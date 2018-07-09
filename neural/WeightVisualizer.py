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
def weightsVisualizer(imgSet, start, length, width, predictedY=None):

    #Where to start within the set of images
    imgID = start

    f, axarr = plt.subplots(length, width, figsize=(length + 1,width))

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
    plt.subplots_adjust(left = 0, bottom = 0.05, right = 1, top = .90, wspace = 0, hspace = 0.5)

    # Finally Done!
    plt.show()
