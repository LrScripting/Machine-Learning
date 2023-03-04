import numpy as np 
import matplotlib.pyplot as plt
import random

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoiddx(x):
    """derivative of sigmoid function"""
    return sigmoid(x)*(1-sigmoid(x))


class NN(object):

    def __init__(self, layerSizes):
        """layerSizes is an array,
        length of sizes is number of layers with each item describing the number of neurons in that layer
        eg [2,1,3] would be three layers, first layer containing 2 neurons, second 1 neuron ect
        the weights and biases are initializes using a normal distribution of mean 0 and variance 1"""
        self.numLayers = len(layerSizes)
        self.sizes = layerSizes
        self.biases = [np.random.randn(y, 1) for y in layerSizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layerSizes[:-1], layerSizes[1:])]

    def forwardProp(self, x):
        """produce network output with a as input"""
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x)+b)
        return x


    def stochGradDecent(self, trainingData, epochs, batchSize, eta, testData=None):
        """Train the network in batches using gradient decent
        this is where you calculate the loss, and then use the chain rule to back propogate through the network and 
        determine which values to change in order to minimalize loss
        This is because : the partial dx/dy of a neuron to the output/loss shows how affecting that weight/bias impacs the overall loss
        Training data is a a touple of input data and labels
        if test data != none test data will be evaluated instead of training network"""
        if testData: testLen = len(testData)
        trainingLen = len(trainingData)
        for j in range(epochs):
            random.shuffle(trainingData)
            batches = [
                    trainingData[k:k+batchSize]
                    for k in range(0, trainingLen, batchSize)]
            for batch in batches:
                self.updateBatch(batch, eta)
            if testData:
                print(f"Epoch {j}: {self.evaluate(testData)} / {testLen}")
            else:
                print(f"Epoch {j} complete")


    def updateBatch(self, batch, lr):
        """update weights and biases
        by applying the backpropagation to each batch
        the batch being a touple of x, y and lr is learning rate"""
        nBiases = [np.zeros(b.shape) for b in self.biases]
        nWeights = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nBiases, delta_nWeights = self.backpropagation(x, y)
            nBiases = [nb+d_nb for nb, d_nb in zip(nBiases, delta_nBiases)]
            nWeights = [nw+d_nw for nw, d_nw in zip(nWeights, delta_nWeights)]
        self.weights = [w-(lr/len(batch))*nw for w, nw in zip(self.weights, nWeights)]
        self.biases = [b-(lr/len(batch))*nb for b, nb in zip(self.biases, nBiases)]

    def backpropagation(self, x, y):
        nBiases = [np.zeros(b.shape) for b in self.biases]
        nWeights = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x] # list all layer activations
        zs = [] #list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.costDerivative(activations[-1], y) *  sigmoiddx(zs[-1])
        nBiases[-1] = delta
        nWeights[-1] = np.dot(delta, activation[-2].transpose())
        
        for l in range(2, self.numLayers):
            z = zs[-l]
            sp = sigmoiddx(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nBiases[-l] = delta
            nWeights[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nBiases, nWeights)


    def evaluate(self, testData):
        """returns the number of correct outputs after running input through the network
        the result is the index of the nueuron in the final layer with the highest activation"""
        testResults = [(np.argmax(self.forwardProp(x)), y) for (x, y) in testData]
        return sum(int(x==y) for (x, y) in testResults)

        
    def costDerivative(self, activations, y):
        return (activations-y)
