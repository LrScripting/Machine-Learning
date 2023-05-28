# efficient byte pair encoding tokenizer class, which can create a token dict from the main text, create token sequences of input data from that dict, and then decode the tokens.

from collections import Counter
import re
class Token:
    def __init__(self):
        self.tokenDictionary = None
        self.unknownToken = "<UNK>"
    
    def buildTokenDict(self, text):
        inputArr = re.sub(r'[^\w\s]', '', text.lower()).split(" ")
        subseqCount = Counter() 

        for word in inputArr:
            subsequences = [word[i:i+2] for i in range(len(word) -1)]
            for subsequence in subsequences:
                subseqCount[subsequence] +=1

        sortedOutput = sorted(subseqCount, key=subseqCount.get, reverse=True)
        self.tokenDictionary = {subseq: i for i , subseq in enumerate(sortedOutput)}
        self.tokenDictionary[self.unknownToken] = len(self.tokenDictionary)

    def tokenizer(self, text):
        if self.tokenDictionary is None:
            raise Exception("Token dictionary is not built yet. Call buildTokenDict first.")
        
        inputArr = re.sub(r'[^\w\s]', '', text.lower()).split(" ")
        tokenizedInput = []

        for word in inputArr:
            subsequences = [word[i:i+2] for i in range(len(word) -1)]
            for subseq in subsequences:
                if subseq in self.tokenDictionary:
                    tokenizedInput.append(self.tokenDictionary[subseq])
                else:
                    tokenizedInput.append(self.tokenDictionary[self.unknownToken])

        return tokenizedInput

    def detokenizer(self, tokenizedSequence):
        if self.tokenDictionary is None:
            raise Exception("Token dictionary is not built yet, call buildTokenDict to create token dictionary")
        
        detokenDict = {i: subseq for subseq, i in self.tokenDictionary.items()}
        detokenizesSequence = [detokenDict[i] for i in tokenizedSequence]
        
        return detokenizesSequence
    
tokeniza = Token()
tokenDict = tokeniza.buildTokenDict("this is a megakek looool")
tokensequences = tokeniza.tokenizer("this is a onmikek zx")
decodedTokens = tokeniza.detokenizer(tokensequences)
print(decodedTokens)
