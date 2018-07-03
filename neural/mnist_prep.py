
from neural.mnist_loader import load_data
import numpy as np

class MnistPrepper():
    def processData(self):
        tr_d, va_d, te_d = load_data()

        #Resize each input vector into a matrix

        #Format the labels as vectors through vecResult to facilitate training
        tr_in = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        in_vec = [self.vecResult(i) for i in tr_d[1]]
        training_data = tuple(zip(tr_in, in_vec))

        va_in = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = tuple(zip(va_in[0], va_d[1]))

        te_in = [np.reshape(x, (784, 1)) for x in te_d[0]]
        testing_data = tuple(zip(te_in[0], te_d[1]))

        return training_data, validation_data, testing_data

    def vecResult(self, i):
        n = np.zeros((10,1))
        n[i] = 1
        return n
