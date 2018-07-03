import numpy as np
import pickle
import copy
from neural.mnist_prep import MnistPrepper


class NeuralNet():
    def __init__(self, sizes=None, fileName=None):
        if (fileName is None):
            self.sizes = sizes
            self.numLayers = len(sizes)
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
            self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        else:
            infile = open(fileName, 'rb')
            loadNN = pickle.load(infile)
            infile.close()
            self.sizes = copy.deepcopy(loadNN.sizes)
            self.numLayers = copy.deepcopy(loadNN.numLayers)
            self.weights = copy.deepcopy(loadNN.weights)
            self.bias = copy.deepcopy(loadNN.bias)

    def createBatches(self, trainingData, size):
        return [trainingData[i: i + size] for i in range(0, len(trainingData), size)]

    def updateParams(self, gradients):
        pass

    def train(self, trainingData, batchSize, learningRate, iterations):
        dataBatches = self.createBatches(trainingData, batchSize)
        for epoch in range(iterations):
            print("Epoch {0}".format(epoch))
            for currentBatch in dataBatches:
                totalWeights = [np.zeros(w.shape) for w in self.weights]
                totalBias = [np.zeros(b.shape) for b in self.bias]
                for item in currentBatch:
                    #print("Continue")
                    gw, gb = self.backProp(item)
                    #Add the lists pointwise
                    totalWeights = [tW + cW for tW, cW in zip(totalWeights, gw)]
                    #print(totalWeights[1])
                    totalBias = [tB + cB for tB, cB in zip(totalBias, gb)]
                #Update the weights and bias using gradient descent
                self.weights = [w - learningRate / batchSize * tW for w, tW in zip(self.weights, totalWeights)]
                self.bias = [b - learningRate / batchSize * tB for b, tB in zip(self.bias, totalBias)]


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
            gradBias[layer - 1] = deltaCurrent
            gradWeights[layer - 1] = np.dot(deltaCurrent, np.transpose(activations[layer - 1]))
            #Propagate deltas to the previous layer
            deltaCurrent = np.dot(np.transpose(self.weights[layer - 1]), deltaCurrent) * sigDeriv[layer - 1]
        return gradWeights, gradBias
    def fowardProp(self, input):
        X = input
        activ = list(range(self.numLayers))
        activ[0] = X
        for layer in range(0, self.numLayers - 1):
            #layer variable is one less than the "real layer"
            activ[layer + 1] = self.sigmoid(np.dot(self.weights[layer], activ[layer]) + self.bias[layer])
            #print("Activation: ")
            #print(activ[layer + 1])
            #print("--------------------")
        deriv = [sig * (1 - sig) for sig in activ]
        return activ, deriv

    def fowardPrb(self, input):
        return self.fowardProp(input)[0][-1]

    def fowardCompute(self, input):
        return np.argmax(self.fowardPrb(input))

    def sigmoid(self, v):
        return (1 / (1 + np.exp(-v)))

    def sigmoidPrime(self, v):
        sig = self.sigmoid(v)
        return sig * (1 - sig)
    def storeNN(self, fileName):
        out = open(fileName, "w+b")
        pickle.dump(self, out)
        out.close()

if __name__ == '__main__':
    test = NeuralNet((3, 3, 3))
    test.storeNN("../data/test.pkl")
