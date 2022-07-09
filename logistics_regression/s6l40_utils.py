import numpy as np


def sigmoid(A):
    return 1/(1+np.exp(-A))



def sigmoid_cost(T, Y):
    return -(T*np.log(Y)+(1-T)*np.log(1-Y)).sum()



def error_rate(targets, predictions):
    return np.mean(targets!=predictions)



def getBinaryData():
    Y = []
    X = []
    first = True
    i = 0
    for line in open('../large_files/fer2013.csv'):
        i+=1
        if first: # column names
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y==0 or y==1:
                if i%1000 == 0:
                    print("i:", i)
                Y.append(y)
                try:
                    X.append([int(p) for p in row[1].split()])
                except Exception:
                    print("row[1]:", row[1])
    return np.array(X) /255.0, np.array(Y)


