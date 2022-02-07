import numpy as np
def entropyFormula(prob1, prob2):
    return -(prob1 * np.log2(prob1) + prob2 * np.log2(prob2))

a = entropyFormula(1/3.0, 2/3.0) *3/5
# b = entropyFormula(3/5.0, 2/5.0) *5/8
# c = a + b
# print(1 - c)
e = entropyFormula(1/2.0, 1/2.0)*2/5
main = entropyFormula(4/5.0, 1/5.0)
print("A: ", main - a)
print("B: ", main - e)

def mutualInfo(prob_y0_x0, prob_y1_x0, prob_y0_x1, prob_y1_x1):
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

