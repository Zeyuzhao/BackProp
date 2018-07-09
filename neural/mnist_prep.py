import numpy as np
import gzip
import pickle

class MnistPrepper():
    MATRIX_T = "matrix_T"
    MATRIX_ALL = "matrix_A"
    SERIAL = "serial"
    def processData(self, type="serial"):

        f = gzip.open('../data/mnist.pkl.gz', 'rb')
        tr_d, va_d, te_d  = pickle.load(f, encoding="latin1")
        f.close()

        #Resize each input vector into a matrix
        #Format the labels as vectors through vecResult to facilitate training
        tr_in = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        vec_d = [self.vecResult(i) for i in tr_d[1]]

        va_in = [np.reshape(x, (784, 1)) for x in va_d[0]]

        te_in = [np.reshape(x, (784, 1)) for x in te_d[0]]

        if type == "matrix_T":
            #Only the training data remains in the block format; X are in an array, y are in an array
            #The validation and testing data remain in individual pairs of tuple, (X, y)
            training_data = tuple((tr_in, vec_d))
            validation_data = tuple(zip(va_in, va_d[1]))
            testing_data = tuple(zip(te_in, te_d[1]))
        elif type == "matrix_A":
            #All data remains in the block format; X are in an array, y are in an array
            training_data = (tr_in, vec_d)
            validation_data = (va_in, va_d[1])
            testing_data = (te_in, te_d[1])
        elif type == "serial":
            #Package each example into each individual pair of tuples, (X, y)
            training_data = tuple(zip(tr_in, vec_d))
            validation_data = tuple(zip(va_in, va_d[1]))
            testing_data = tuple(zip(te_in, te_d[1]))
        else:
            raise(NotImplementedError("This mode is not yet implemented"))

        return training_data, validation_data, testing_data


    def vecResult(self, i):
        n = np.zeros((10,1))
        n[i] = 1
        return n

if __name__ == '__main__':
    train, valid, testing = MnistPrepper().processData()