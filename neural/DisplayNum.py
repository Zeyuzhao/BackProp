import numpy as np
import matplotlib.pyplot as plt
import itertools
from neural.mnist_loader import load_data



length = 8
width = 8
imgID = 200

f, axarr = plt.subplots(length, width, figsize=(6,8))

f.suptitle("MNIST data set from pic {0} to {1}".format(imgID, imgID + length * width - 1))
# Extract the sets of data, X and y, from training data

training_data, validation_data, test_data = load_data()

picTuples = training_data[0]
picLabels = training_data[1]

for i, j in itertools.product(range(length), range(width)):
    # Scale Normalized number into byte value
    x = (255 * picTuples[imgID]).astype(int)
    # Wrap into a 2d array
    img = x.reshape(28, 28)

    # Show images
    axarr[i, j].imshow(img)
    axarr[i, j].set_title("{1}".format(imgID, picLabels[imgID]))
    axarr[i, j].axis('off')

    # Keep count of the current images
    imgID += 1

# Adjust the margins so the labels don't overlap
plt.subplots_adjust(bottom=0)

# Finally Done!
plt.show()
