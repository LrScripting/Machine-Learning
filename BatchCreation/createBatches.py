# takes tokenized sentences and creates batches dynamically at the length of the longest sentence in the batch

import string
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
