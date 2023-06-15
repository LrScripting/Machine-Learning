import math
import numpy as np
from collections import Counter
import re
import string


class encoder:
    def __init__(self, dModel, h):
        self.dModel = dModel
        self.h = h
        self.dK = self.dModel // self.h        
        self.gamma = np.ones(self.dModel)
        self.beta = np.zeros(dModel)
        self.epislon = 1e-6

        self.queryProjections = [np.random.randn(self.dModel, self.dK) for _ in range(self.h)]
        self.keyProjections = [np.random.randn(self.dModel, self.dK) for _ in range(self.h)]
        self.valueProjections = [np.random.randn(self.dModel, self.dK) for _ in range(self.h)]
        self.linearLayerMatrix = np.random.randn(self.h*self.dK , dModel)

        self.weightsOne = np.random.randn(self.dModel, self.dFF)
        self.weightsTwo = np.random.randn(self.dFF, self.dModel)
        self.biasOne = np.zeros(self.dFF)
        self.biasTwo = np.zeros(self.dFF)

    def attentionHead(self, query, keys, value):
        return np.dot(self.softmax(np.dot(query, keys.T)/np.sqrt(self.dK)), value)

    def multiheadAttention(self, query, keys, values):
        batch_size, seq_len, _ = query.shape
        attentionResultArr = []
        for i in range(self.h):
            # Reshape the input arrays to 2D shape
            query_2d = query.reshape(-1, self.dModel)
            key_2d = keys.reshape(-1, self.dModel)
            value_2d = values.reshape(-1, self.dModel)

            queryTransformation = np.dot(query_2d, self.queryProjections[i])
            keyTransformation = np.dot(key_2d, self.keyProjections[i])
            valueTransformation = np.dot(value_2d, self.valueProjections[i])

            headResult = self.attentionHead(queryTransformation, keyTransformation, valueTransformation)
            attentionResultArr.append(headResult.reshape(-1, self.dK))

        joinedVector = np.concatenate(attentionResultArr, axis=-1)

        output = np.dot(joinedVector, self.linearLayerMatrix)

        # Reshape the output to original input shape
        output = output.reshape(batch_size, seq_len, -1)
    
        return output
    # y = activation(dot_product(W, x) + b)
    def feedforwardNetwork(self, input):
        hiddenLayer = self.gelu(0, np.dot(input, self.weightsOne) + self.biasOne)
        output = np.dot(hiddenLayer, self.weightsTwo) + self.biasTwo
        return output    
    def layerNormalisation(self, layerInput):
        inputMean = np.mean(layerInput, axis=1, keepdims=True)
        sDeviation = np.sqrt(np.var(layerInput, axis=-1, keepdims=True) + self.epislon)
        xHat = (layerInput - inputMean) / sDeviation
        normalisedOutput = self.gamma * xHat - self.beta
        return normalisedOutput

    def dropout(self, inputLayer, training=True):
        if not training:
            return inputLayer
        mask = np.random.rand(inputLayer.shape[0], inputLayer.shape[1]) > self.dropoutRate
        return mask * inputLayer / (1.0 -self.dropoutRate)

    def residualConnection(self, inputLayer, layerFunction, training=True):
        
        output= self.dropout(layerFunction(inputLayer), training=training)
        residualOutput = inputLayer + output
        return residualOutput     
    
    def forward(self, input, training=True):
        attentionLayerOutput = self.residualConnection(input, lambda x: self.multiheadAttention(x, x, x), training=training)
        normalised = self.layerNormalisation(attentionLayerOutput)
        feedforwardOutput = self.residualConnection(self.feedforwardNetwork(normalised), attentionLayerOutput, training=training)
        return self.layerNormalisation(feedforwardOutput)

        
