from collections import Counter
import re
class Token:
    def __init__(self):
        self.tokenDictionary = None
        self.unknownToken = "<UNK>"
    
    def buildTokenDict(self, text):
        inputArr = re.sub(r'[^\w\s]', '', text.lower()).split(" ")
        wordCount = Counter() 

        for word in inputArr:
            wordCount[word] +=1

        sortedOutput = sorted(wordCount, key=wordCount.get, reverse=True)
        print(sortedOutput)
        print(wordCount)
        self.tokenDictionary = {subseq: i for i , subseq in enumerate(sortedOutput)}
        self.tokenDictionary[self.unknownToken] = len(self.tokenDictionary)
        print(self.tokenDictionary)
        
    def tokenizer(self, inputVector):
        if self.tokenDictionary == None:
            raise Exception("please build tokenDict first")

        inputV = re.sub(r'[^\w\s]', '', inputVector.lower()).split(" ")

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


tokeniza = Token()
tokeniza.buildTokenDict("HELLO HI GOODBYE")
lol = tokeniza.tokenizer("HELLO HI BYE")
