import numpy as np
import pickle
import copy

from neural.mnist_prep import MnistPrepper
from neural.DisplayNum import showImgs

class NeuralNet():
    def __init__(self, sizes=None, fileName=None):
        if (fileName is None):
            self.sizes = sizes
            self.numLayers = len(sizes)
            #Init the W and b using normal distribution
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
            self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        else:
            infile = open(fileName, 'rb')
            loadNN = pickle.load(infile)
            infile.close()

            #Deep copy each of the properties from the pickle file
            self.sizes = copy.deepcopy(loadNN.sizes)
            self.numLayers = copy.deepcopy(loadNN.numLayers)
            self.weights = copy.deepcopy(loadNN.weights)
            self.bias = copy.deepcopy(loadNN.bias)

    def createBatches(self, items, size):
        #Unzip the items into 2 large lists, X and y.
        Xy = zip(*items)
        #Process them into a list of pairs of matrices, each item has a X matrix, and a y matrix
        processed = [[np.concatenate(s[i: i + size], axis=1) for i in range(0, len(s), size)] for s in Xy]
        #print(len(processed[1]))
        return tuple(zip(processed[0], processed[1]))

    def train(self, trainingData, batchSize, learningRate, iterations, testingData = None, checkSize = 1000):
        if testingData is not None:
            print("Target Number of Epoch: {0}\nTest Set Size: {1}".format(iterations, checkSize))
        dataBatches = self.createBatches(trainingData, batchSize)
        for epoch in range(iterations):
            print("Epoch {0}".format(epoch))
            for currentBatch in dataBatches:
                totalWeights, totalBias = self.backProp(currentBatch)
                #Update the weights and bias using gradient descent

                weightDim = [w.shape for w in totalWeights]
                biasDim = [b.shape for b in totalBias]

                self.weights = [w - learningRate / batchSize * tW for w, tW in zip(self.weights, totalWeights)]
                self.bias = [b - learningRate / batchSize * tB for b, tB in zip(self.bias, totalBias)]

                weightDim = [w.shape for w in self.weights]
                biasDim = [b.shape for b in self.bias]
                #print("Batch Done")
            if testingData is not None:
                acc = self.checkAccuracy(testingData)
                print("Accuracy: {0:.4f}".format(acc))

    def evalPerform(self, testingData, start = 0, numItems = 1000):
        zippedData = tuple(zip(*testingData))

        #Extract the data
        X = zippedData[0][start:start + numItems]
        y = zippedData[1][start:start + numItems]

        #Concat the X vectors in horizontally
        testDataMatrix = np.concatenate(X, axis = 1)
        predictedY = self.fowardCompute(testDataMatrix)
        correct = 0
        wrongList = []
        for i in range(numItems):
            if (predictedY[i] == y[i]):
                correct += 1
            else:
                wrongList.append(i + numItems)
        return (correct/numItems), wrongList

    def checkAccuracy(self, testingData, start = 0, numItems = 1000):
        return self.evalPerform(testingData, start, numItems)[0]

    def backProp(self, item):
        X = item[0]
        y = item[1]

        #Init gradients
        gradWeights = [np.zeros(w.shape) for w in self.weights]
        gradBias = [np.zeros(b.shape) for b in self.bias]

        #Compute activations from foward prop
        activations, sigDeriv = self.fowardProp(X)

        #Setup the first layer, derivative of L2 norm cost function is simply the difference
        deltaCurrent = (activations[-1] - y) * sigDeriv[-1]
        gradWeights[-1] = deltaCurrent

        #Begin with the last layer (num - 1) and iterate until the second layer (1).
        for layer in range(self.numLayers - 1, 0, -1):
            #Calculate the gradient for each layer
            #The arrays are shifted down by 1 compared to the equation notation, so subtract 1
            gradBias[layer - 1] = np.sum(deltaCurrent, axis = 1).reshape((deltaCurrent.shape[0]), 1)
            gradWeights[layer - 1] = np.dot(deltaCurrent, np.transpose(activations[layer - 1]))
            #Propagate deltas to the previous layer
            deltaCurrent = np.dot(np.transpose(self.weights[layer - 1]), deltaCurrent) * sigDeriv[layer - 1]
        return gradWeights, gradBias

    def fowardProp(self, input):
        X = input
        activ = [X,]
        for layer in range(0, self.numLayers - 1):
            #layer variable is one less than the "real layer"
            activ.append(self.sigmoid(np.dot(self.weights[layer], activ[layer]) + self.bias[layer]))
            #print("Activation: ")
            #print(activ[layer + 1])
            #print("--------------------")
        deriv = [sig * (1 - sig) for sig in activ]
        return activ, deriv

    def fowardPrb(self, input):
        #fowardProp returns activations and costfunction gradients for each layer
        #the last activation represents the predicted y
        return self.fowardProp(input)[0][-1]

    def fowardCompute(self, input):
        #Returns the max of each Y output
        return np.argmax(self.fowardPrb(input), axis=0)

    def sigmoid(self, v):
        return 1 / (1 + np.exp(-v))

    def sigmoidPrime(self, v):
        sig = self.sigmoid(v)
        return sig * (1 - sig)
    def storeNN(self, fileName):
        out = open(fileName, "w+b")
        pickle.dump(self, out)
        out.close()

if __name__ == '__main__':
    dim = (784, 100, 100, 10)
    network = NeuralNet(dim)

    epoch = 30
    training_data, validation_data, testing_data = MnistPrepper().processData()

    network.train(training_data, 20, 3, epoch, testingData=testing_data)

    network.storeNN("../data/mnistNN_L{0}_E{1}.pkl".format(network.numLayers, epoch))
    l = 5
    w = 5

    tX, ty = zip(*testing_data)

    acc, wrong = network.evalPerform(testing_data, 0, 100)

    print("Total Accuracy: {0}".format(acc))

    predictedY = network.fowardCompute(np.concatenate(tX, axis=1)).tolist()
    showImgs(testing_data, 0, l, w, predictedY)
