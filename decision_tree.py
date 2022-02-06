import sys
import numpy as np
from collections import defaultdict

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self):
        # access nodes and branches
        self.left = None
        self.right = None
        self.branches = None
        # leaf, internal node attributes
        self.type = None
        self.attr = None
        self.vote = None
        self.dataSet = None
        self.parent = None
        self.depth = 0
    
    def __repr__(self):
        toRet = "type: " + self.type + "\n"
        if self.type == 'leaf':
            toRet += "vote: " + self.vote + "\n"
            toRet += "depth: " + str(self.depth)
        else:
            print("\n")
            print("branches: ", self.branches)
            toRet += "attr: " + self.attr + "\n"
        toRet += "\n"
        return toRet
    
    def getNextDecision(self, attribute):
        #if branches exist (not leaf) go down branch containing attribute value
        if self.branches: 
            return self.branches[attribute]
            
######### Start Testing Functions ##########
def printTree(node, level=0):
    if node != None:
        printTree(node.left, level + 1)
        if node.left is None and node.right is None:
            print(' ' * 8 * level + '->', node.vote)
        else: print(' ' * 8 * level + '->', node.attr)
        printTree(node.right, level + 1)

def pretty_print(node):
    if node != None:
        # print root node
        if not node.parent:
            (att1,count1), (att2,count2), _ = countValues(node.dataSet[:,-1])
            print("[" + str(count1) + att1 + "/" + str(count2) + att2 + "]")
        # print branches and next nodes
        if node.branches:
            for bin_val in sorted(list(node.branches.keys())):
                # retrieve next node
                next = node.getNextDecision(bin_val)
                (att1,count1), (att2,count2), _ = countValues(next.dataSet[:,-1])
                print("| " * (next.depth - 1) + next.parent + " = " + bin_val + "[" + str(count1) + " " + att1 + "/" + str(count2) + " " + att2 + "]")
                pretty_print(next)
######### End Testing Functions ##########

######### Start Helper Functions ##########
'''Read File'''
def read_data(in_file, linesToSkip = 1):
    filename = in_file
    data = np.genfromtxt(filename, delimiter="\t", dtype = str)
    headers = data[0,]
    data = data[linesToSkip:,]
    dataAsArray = np.array(data)
    return (headers,dataAsArray)

'''Data manipulation functions'''
# count the occurances of binary values
def countValues(values):
    countTable = defaultdict(int)
    for value in values:
        countTable[value] += 1
    allKeys= sorted(list(countTable.keys()))
    att1, count1 = allKeys[0], countTable[allKeys[0]]
    # case when there may only be one label in labels
    if len(allKeys) > 1:
        att2, count2 = allKeys[1], countTable[allKeys[1]]
    else: 
        att2 = ''
        count2 = 0
    total = float(count1 + count2)
    return (att1,count1), (att2,count2), total

def getFeatureIndex(feature, headers):
    for i, header in enumerate(headers):
        if header == feature: return i
    return -1

'''Entropy + Mutual Information Functions'''
def entropyFormula(prob1, prob2):
    return -(prob1 * np.log2(prob1) + prob2 * np.log2(prob2))

def getEntropy(data):
    (_, count1), (_, count2), total = countValues(data)
    prob1 = count1 / total 
    prob2 = count2 / total
    entropy = 0
    # if labels are not pure then count
    if not (prob1 == 0 or prob2 == 0):
        entropy = entropyFormula(prob1, prob2)
    return entropy

def getConditionalEntropy(dataXm, dataLabels):
    y0 = dataLabels[0] #set first var of label as y = 0 
    x0 = dataXm[0] #set first var of dataXm as x = 0 
    # initialize count vars for y0 given x0 occurred, y0 given x1 .... y1 given x1
    y0_x0, y0_x1, y1_x0, y1_x1 = 1, 0, 0, 0 
    for x,y in zip(dataXm[1:], dataLabels[1:]):
        if y == y0:
            if x == x0:
                y0_x0 += 1
            else: # x == x1
                y0_x1 += 1
        else: # y == y1
            if x == x0:
                y1_x0 += 1
            else: # x == x1
                y1_x1 += 1
    x0_count = float(y0_x0 + y1_x0)
    x1_count = float(y0_x1 + y1_x1)

    prob_x0 = x0_count/ (x0_count + x1_count)
    prob_x1 = x1_count/ (x0_count + x1_count)
    if (prob_x0 == 0 or prob_x1 == 0): return 0 # pure data set occurance
    prob_y0_x0 = y0_x0/x0_count
    prob_y1_x0 = y1_x0/x0_count
    prob_y0_x1 = y0_x1/x1_count
    prob_y1_x1 = y1_x1/x1_count

    entropy_y_x0 = 0
    # if sub-dataset y0 given x0 or y1 given x0 is not pure
    #   calculate entropy
    if not (prob_y0_x0 == 0 or prob_y1_x0 == 0): 
        entropy_y_x0 = entropyFormula(prob_y0_x0, prob_y1_x0)
    entropy_y_x1 = 0
    # if sub-dataset y0 given x1 or y1 given x1 is not pure
    #   calculate entropy
    if not(prob_y0_x1 == 0 or prob_y1_x1 == 0):
        entropy_y_x1 = entropyFormula(prob_y0_x1, prob_y1_x1)
    # sum of entropy
    # print("x0: ", entropy_y_x0)
    # print("x1: ", entropy_y_x1)
    entropy_y_x = prob_x0 * entropy_y_x0 + prob_x1 * entropy_y_x1
    return entropy_y_x
    
def getMutualInfo(dataXm, dataLabels):
    Hy = getEntropy(dataLabels)
    if Hy == 0: return 0 # check if labels are pure
    Hy_x = getConditionalEntropy(dataXm, dataLabels)
    if Hy_x == 0: return 0 # check if attributes of features are pure
    return Hy - Hy_x
    
def mutual_info_split(D, headers):
    maxMI = 0
    maxMI_feature = ''
    labels = D[:, -1]
    for i, feature in enumerate(headers[:-1]):
        colAtFeature = D[:, i]
        MI = getMutualInfo(colAtFeature, labels)
        if MI > maxMI: 
            maxMI = MI
            maxMI_feature = feature
    return maxMI, maxMI_feature

'''Majority Vote Function'''
def majorityVote(D):
    majorityLabel = ""
    if dataEmpty(D): 
        return majorityLabel
    (label1, count1), (label2, count2), total = countValues(D[:, -1]) # Count labels
    if count1 > count2: majorityLabel =  label1
    elif count1 < count2: majorityLabel = label2
    else:  #if count equal, majority label is alphabetically later label
        if label1 < label2: 
            majorityLabel =  label2 
        else: 
            majorityLabel =  label1
    return majorityLabel

''' Stopping Condition Functions'''
def isMaxDepth(node, maxDepth):
    return node.depth > maxDepth

def dataEmpty(data):
    return data.size == 0

def invalidMutualInfo(value):
    return value <= 0

######### End Helper Functions ##########

######### Start Main Functions ##########
def train(D, headers, maxDepth):
    def train_recursive(D, curDepth, parent):
        '''Base Case'''
        p = Node()
        p.depth = curDepth
        p.parent = parent
        p.dataSet = D
        # figure out best splitting feature and its mutual information 
        MI_val, XmToSplit = mutual_info_split(D, headers)
        # print(XmToSplit)
        # print(D)
        # print(D.shape)
        # check if non-empty dataset, max depth reached, and mutual info > 0
        if (dataEmpty(D) or isMaxDepth(p, maxDepth) or 
            invalidMutualInfo(MI_val)):
            p.type = 'leaf'
            p.vote = majorityVote(D)
            return p

        p.type = 'internal'
        p.attr = XmToSplit # Best feature to Split
        p.depth = curDepth

        # divide data set into two, each representing one probable binary value
        splittedDataTable = defaultdict(list)
        for i, binKey in enumerate(D[:, getFeatureIndex(XmToSplit, headers)]):
            splittedDataTable[binKey].append(D[i,:]) 
        # sort keys to make sure one binary key alwys stays in one side of tree
        binaryKeys = sorted(list(splittedDataTable.keys())) 
        val_xm_0, D_Xm_Val0 = binaryKeys[0], splittedDataTable[binaryKeys[0]]

        if len(binaryKeys) == 2: #if set only contains two binary values
            val_xm_1, D_Xm_Val1 = binaryKeys[1], splittedDataTable[binaryKeys[1]]
        else: # if set only contains one binary value
            val_xm_1, D_Xm_Val1 = '', np.empty((1,1))

        # recursively build tree
        p.left = train_recursive(np.array(D_Xm_Val0), curDepth + 1, XmToSplit)
        p.right = train_recursive(np.array(D_Xm_Val1), curDepth + 1, XmToSplit)

        # let left, right branch take on binary values possible at feature
        p.branches = dict()
        p.branches[val_xm_0] = p.left
        p.branches[val_xm_1] = p.right

        # reduce tree depth if both leaf nodes have the same vote
        # if (p.left.type == 'leaf' and p.right.type == 'leaf' and
        #     p.left.vote == p.right.vote):
        #     print(p)
        #     # set current node to leaf node with the vote
        #     p.type = 'leaf'
        #     p.attr = None
        #     p.vote = p.left.vote
        #     p.left = None
        #     p.right = None
        #     p.branches = None
        #     return p

        return p

    root = train_recursive(D, 1, None)
    return root

def test(model, headers, D, out_file):
    with open(out_file, 'w') as f:
        for row in D:
            root = model
            assert(root) #ensure root exists
            while root.type == 'internal':
                attrToSplit = root.attr
                valAtFeature = row[getFeatureIndex(attrToSplit, headers)]
                root = root.branches[valAtFeature]
            assert(root.type == 'leaf')
            f.writelines(root.vote + "\n")
            

def error(train, trainPred, test, testPred, metricsFile):
    assert(len(train) == len(trainPred))
    assert(len(test) == len(testPred))
    incorrectCountTrain = 0
    incorrectCountTest = 0
    # Compute train error
    for i in range(len(train)):
        if train[i][-1] != trainPred[i]: 
            incorrectCountTrain += 1
    trainError = float(incorrectCountTrain)/len(trainPred)

    # Compute test error
    for i in range(len(test)):
        if test[i][-1] != testPred[i]: 
            incorrectCountTest += 1
    testError = float(incorrectCountTest)/len(testPred)

    # Write data to metrics
    with open(metricsFile, 'w') as f:
        f.writelines("error(train): " + str(trainError) + "\n")
        f.writelines("error(test): " + str(testError) + "\n")

######### End Main Functions ##########
if __name__ == '__main__':
    # read commandline input
    trainInput = sys.argv[1]
    testInput = sys.argv[2]
    maxDepth = int(sys.argv[3])
    trainOutput = sys.argv[4]
    testOutput = sys.argv[5]
    metrics = sys.argv[6]
    # train model
    headers,trainD = read_data(trainInput)
    model = train(trainD, headers, maxDepth)
    # test train-data
    test(model, headers, trainD, trainOutput)
    _, predictedTrain = read_data(trainOutput, 0)
    # test test-data
    headers,testD = read_data(testInput)
    test(model, headers, testD, testOutput)
    _, predictedTest = read_data(testOutput, 0)
    # metrics and print tree
    error(trainD, predictedTrain, testD, predictedTest, metrics)
    # printTree(model)
    pretty_print(model)


