from itertools import count
import sys
import numpy as np
from collections import defaultdict

def read_data(in_file, linesToSkip = 1):
    filename = in_file
    data = np.genfromtxt(filename, delimiter="\t", dtype = str)
    data = data[linesToSkip:,]
    dataAsArray = np.array(data)
    return dataAsArray

def countLabels(labels):
    # count labels in last column
    countTable = defaultdict(int)
    for label in labels:
        countTable[label] += 1
    allLabels = list(countTable.items())
    label1, count1 = allLabels[0]
    label2, count2 = allLabels[1]
    total = float(count1 + count2)
    return (count1, count2, total)

def getEntropy(labels):
    count1, count2, total = countLabels(labels)
    prob1 = count1 / total 
    prob2 = count2 / total
    entropy = 0
    # if labels are not pure then count
    if not (prob1 == 0 or prob2 == 0):
        entropy = - (prob1 * np.log2(prob1) + prob2 * np.log2(prob2))
    return entropy
    
def error(labels):
    count1, count2, total = countLabels(labels)
    # if count1 is majority vote
    if count1 > count2:
        #by majority vote label2 will all count as errors
        countErrors = count2 
    else: # if count2 is majority vote
        countErrors = count1

    return countErrors/total

if __name__ == "__main__":
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    dataArray = read_data(inputFile)
    labels = dataArray[:, -1]
    entropy = getEntropy(labels)
    errorRate = error(labels)

    with open(outputFile, 'w') as f:
        f.writelines("entropy: " + str(entropy) + "\n")
        f.writelines("error: " + str(errorRate) + "\n")