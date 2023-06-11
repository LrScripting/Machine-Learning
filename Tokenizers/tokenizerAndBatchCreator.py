#applies a full word tokenization algorithm then dynamically creates batches of varying length depending on the length of the longest sentence in the batch.

import re
import string
from collections import Counter
import numpy as np

class BatchCreator:
    def __init__(self, sequences):
        self.rawSequences = self.sort_sequences(sequences)
        self.padValue = 0
        self.batchLength = 12

    def sort_sequences(self, sequences):
        return sorted(sequences, key=len, reverse=True)

    def pad_batch(self, sequences, padValue):
        seqMaxlength = max(len(seq) for seq in sequences)
        return [list(seq) + [padValue] * (seqMaxlength - len(seq)) for seq in sequences]
    
    def create_batches(self):
        batches = [] 
        for i in range(0, len(self.rawSequences), self.batchLength):
            sequences = self.rawSequences[i: i+self.batchLength]
            paddSequences = self.pad_batch(sequences, self.padValue)
            batches.append(paddSequences)
        return batches

 class Token:
    def __init__(self):
        self.tokenDictionary = None
        self.unknownToken = "<UNK>"
    
    def buildTokenDict(self, text):
        text = text.lower()
        text = re.sub(r'['+string.punctuation+']', '', text)
        inputArr = text.split()
        wordCount = Counter() 

        for word in inputArr:
            wordCount[word] +=1

        sortedOutput = sorted(wordCount, key=wordCount.get, reverse=True)
        self.tokenDictionary = {subseq: i for i , subseq in enumerate(sortedOutput)}
        self.tokenDictionary[self.unknownToken] = len(self.tokenDictionary)
        

    def tokenizer(self, inputVector):
        if self.tokenDictionary == None:
            raise Exception("please build tokenDict first")

        inputVector = inputVector.lower()
        
        inputVector = re.sub(r'['+string.punctuation+']', '', inputVector)
        inputV = inputVector.split()


        tokenizedOutput = []
        for word in inputV:
            if word in self.tokenDictionary:
                tokenizedOutput.append(self.tokenDictionary[word])
            else:
                tokenizedOutput.append(self.tokenDictionary[self.unknownToken])

        return tokenizedOutput

    def detokenize(self, tokenVector):
        translatedVecor = []
        if self.tokenDictionary is None:
            raise Exception("No token Dictionary")
        detokenDict = {i: word for word, i in self.tokenDictionary.items()}
        translatedVecor = [detokenDict[word] for word in tokenVector]
        return translatedVecor
    

def testBatches(sentences):
    tokenizer = Token()
    tokenizer.buildTokenDict(sentences)
    sentences = sentences.split(".")
    tokenized_sentences = [tokenizer.tokenizer(sentence) for sentence in sentences]
    batch_creator = BatchCreator(tokenized_sentences)
    arr = batch_creator.create_batches()
    return arr

