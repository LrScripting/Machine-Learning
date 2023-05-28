# Basic, essentially brute force bytepair encoding algorithm to produce tokens for a transformer/llm NN
from collections import Counter
import re

def tokenizer(text):
    subsequences = []
    inputArr = re.sub(r'[^\w\s]', '', text.lower()).split(" ")
    finalArr = []
    subseqCount = Counter() 

    for i in range(len(inputArr)):
        subsequences.append(list(zip(list(inputArr[i]), list(inputArr[i][1:]))))
    for i in range(len(subsequences)):
        for x, y in subsequences[i]:
            finalArr.append(str(x + y))


    
    sortedOutput = sorted(finalArr, key=lambda x: subseqCount[x], reverse=False)


    tokenIndex = 0
    tokenDictionary = {}
    for subseq in sortedOutput:
        if subseq not in tokenDictionary:

            tokenDictionary[subseq] = tokenIndex
            tokenIndex+=1


    return tokenDictionairy

tokenizer("waffle squack Scribble homounculous peppermint pepper pepper, ",)
