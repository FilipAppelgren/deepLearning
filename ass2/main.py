import pickle
import numpy as np
import random as rand
import math
import matplotlib.pyplot as plt
from tqdm import tqdm



def init(batch):
	x, labels = unpickle("../datasets/cifar-10-batches-py/"+batch)
	return x, labels

def unpickle(file):

    with open(file, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
        x = batch['data'].T
        labels = batch['labels']
    return x, labels

def createLabels(labels):
    y = np.zeros((10, 10000))
    for column, label in enumerate(labels):
        y[label][column] = 1
    return y

def normalize(x):
    d = len(x)
    # Calculate mean and std for every pixel
    pixel_means = np.mean(x, axis=1).reshape(d, 1)
    pixel_std = np.std(x, axis=1).reshape(d, 1)
    
    # Normalize by subtracting the mean and dividing by std. Linear transforms.
    x = x - pixel_means
    x = x / pixel_std
    return x

def initBiasWeight():
    w1 = np.random.normal(0, (1/np.sqrt(d))**2, (m, d))
    w2 = np.random.normal(0, (1/np.sqrt(d))**2, (k, m))
    b1 = np.zeros((m, 1))
    b2 = np.zeros((k, 1))

    return b1, b2, w1, w2

def crossLoss(x, w1, b1, w2, b2, y):

    p,  h_arr = evaluateClassifier(x, w1, b1, w2, b2)
    cross = np.zeros((1, len(p[0])))
    
    for i in range(len(p[0])):
        temp = p.T[i].reshape(k, 1)
        temp = temp.T
        temp_one = y.T[i].reshape(k, 1)
        #print(temp_one)
        cross[0, i] = -np.log(temp@temp_one)
    #cross = (y*p).sum(axis=0)
    #cross[cross == 0] = np.finfo(float).eps
    #cross = -np.log(cross)

    return cross, p, h_arr

def computeCost(x, y, w1, b1, w2, b2, reg):
	
    loss, p, h_arr = crossLoss(x, w1, b1, w2, b2, y)
    w1_2 = np.square(w1)
    w2_2 = np.square(w2)

    return (1/len(x[0]) * np.sum(loss)) + (reg * ((np.sum(w1_2)) + np.sum(w2_2))), p, h_arr


def computeGradsNum(x, y, w1, b1, w2, b2, reg, h):

    grad_W1 = np.zeros(w1.shape)
    grad_b1 = np.zeros((b1.shape[0], 1))
    grad_W2 = np.zeros(w2.shape)
    grad_b2 = np.zeros((b2.shape[0], 1))

    c = computeCost(x, y, w1, b1, w2, b2, reg)[0]

    b1_try = np.copy(b1)
    for j in range(0, len(b1)):
        b1_try[j] += h
        c2 = computeCost(x, y, w1, b1_try, w2, b2, reg)[0]
        grad_b1[j] = (c2 - c)/h
        b1_try[j] -= h

    b2_try = np.copy(b2)
    for j in range(0, len(b2)):
        b2_try[j] += h
        c2 = computeCost(x, y, w1, b1, w2, b2_try, reg)[0]
        grad_b2[j] = (c2 - c)/h
        b2_try[j] -= h

    w1_try = np.copy(w1)
    #for i in tqdm(range(0, len(w1))):
    #    for j in tqdm(range(0, len(w1[0]))):
    for i in range(0, len(w2)):
        for j in range(0, len(w2[0])):        
            w1_try[i][j] += h
            c2 = computeCost(x, y, w1_try, b1, w2, b2, reg)[0]
            grad_W1[i][j] = (c2 - c)/h
            w1_try[i][j] -= h

    w2_try = np.copy(w2)
    #for i in tqdm(range(0, len(w2))):
        #for j in tqdm(range(0, len(w2[0]))):
    for i in range(0, len(w2)):
        for j in range(0, len(w2[0])):
            w2_try[i][j] += h
            c2 = computeCost(x, y, w1, b1, w2_try, b2, reg)[0]
            grad_W2[i][j] = (c2 - c)/h
            w2_try[i][j] -= h

    return grad_W1, grad_W2, grad_b1, grad_b2

def computeAccuracy(p, y):
    n = len(y[0])
    results = np.amax(p, axis=0)
    counter = 0

    for row, result in enumerate(results):
        if y.T[row][np.where(p.T[row] == result)[0]] == 1:
            counter += 1
    return (counter / n) * 100


def computeGradients(x, y, p, h_arr, w1, w2, b1, b2, reg):

    n = len(x[0])
    dL_w1 = np.zeros(w1.shape)
    dL_b1 = np.zeros((b1.shape[0], 1))
    dL_w2 = np.zeros(w2.shape)
    dL_b2 = np.zeros((b2.shape[0], 1))

    g = -(y - p)
 
    dL_w2 = (g@h_arr.T) * (1/n)
    dL_w2 += 2 * reg * w2
    dL_b2 = np.mean(g, axis=-1, keepdims=True)

    g = w2.T@g

    h_temp = invRelu(h_arr)
    g = np.multiply(g, h_temp)

    dL_w1 = (1/n) * g@x.T
    dL_w1 += 2 * reg * w1
    dL_b1 = np.mean(g, axis=-1, keepdims=True)

    return dL_w1, dL_w2, dL_b1, dL_b2


def evaluateClassifier(x, w1, b1, w2, b2):

    s1 = w1@x + b1
    h = ReLU(s1)
    s = w2@h + b2
    p = softMax(s)

    return p, h

def invRelu(s):
    s[s>0] = 1
    return s

def ReLU(s):
    s[s<0] = 0
    return s

def softMax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def split(x, y, ratio):
    x = x[0:, 0:100]
    y = y[0:, 0:100]

    return x, y

def accuracyCost(trainX, trainY, valX, valY, w1, w2, b1, b2, reg):

    trainCost, trainP = computeCost(trainX, trainY, w1, b1, w2, b2, reg)[0:2]
    valCost, valP = computeCost(valX, valY, w1, b1, w2, b2, reg)[0:2]
    trainAccuracy = computeAccuracy(trainP, trainY)
    valAccuracy = computeAccuracy(valP, valY)

    return trainCost, valCost, testCost, trainAccuracy, valAccuracy, testAccuracy


if __name__ == "__main__":
    trainX, trainLabels = init("data_batch_1")
    valX = trainX[0:, 5000:10000]
    valY = trainLabels[5000:10000]

    trainX = trainX[0:, 0:5000]
    trainLabels = trainLabels[0:5000]

    for i in range(2, 6):
        tempX, tempLabels = init("data_batch_"+str(i))
        trainX = np.append(trainX, tempX, axis=1)
        

    #valX, valLabels = init("data_batch_2")
    #testX, testLabels = init("test_batch")
    
    m = 50
    n = len(trainX[0])
    d = len(trainX)
    batch = 100

    eta_min = 1e-5
    eta_max = 1e-1
    eta = eta_min
    n_s = 2 * np.floor(n/batch)
    cycles = 2
    n_epochs = (n_s/batch)*2*cycles
    t = 1
    l = 0

    k = 10
    reg = 0.01
    h = 1e-6

    trainY = createLabels(trainLabels)


    minmax_epochs = (n_s/(n / batch))*2
    trainX = normalize(trainX)

    valX = normalize(valX)
    valY = createLabels(valY)

    #testX = normalize(testX)
    #testY = createLabels(testY)

    b1, b2, w1, w2 = initBiasWeight()

    trainCost, valCost, testCost, trainAccuracy, valAccuracy, testAccuracy = accuracyCost(
        trainX, trainY, valX, valY, w1, w2, b1, b2, reg)
    
    print("Train cost", trainCost, "Accuracy", trainAccuracy)
    print("")
    print("Val cost", valCost, "Accuracy", valAccuracy)


    for i in range(int(n_epochs)):
        
        if(i % minmax_epochs) == 0 and i != 0:
            l += 1
        

        for j in range(0, n, batch):
            
            if t >= (2 * l * n_s) and t <= (2 * l + 1)*n_s:
                eta = eta_min + (t - 2*l*n_s)/n_s * (eta_max - eta_min)
                
            elif t >= (2*l + 1)*n_s and t <= 2*(l + 1)*n_s:
                eta = eta_max - (t-(2*l + 1)*n_s)/n_s * (eta_max - eta_min)
        
            x_temp = trainX[0:, j:j+batch]
            y_temp = trainY[0:, j:j+batch]
            cost, p, h_arr = computeCost(x_temp, y_temp, w1, b1, w2, b2, reg)
            dL_w1, dL_w2, dL_b1, dL_b2 = computeGradients(x_temp, y_temp, p, h_arr, w1, w2, b1, b2, reg)
            #dL_w1, dL_w2, dL_b1, dL_b2 = computeGradsNum(x_temp, y_temp, w1, b1, w2, b2, reg, h)

            w1 = w1 - (eta*dL_w1)
            w2 = w2 - (eta*dL_w2)
            b1 = b1 - (eta*dL_b1)
            b2 = b2 - (eta*dL_b2)
            t+=1
            print(eta)
    
    trainCost, valCost, testCost, trainAccuracy, valAccuracy, testAccuracy = accuracyCost(
            trainX, trainY, valX, valY, w1, w2, b1, b2, reg)

    print("Train cost", trainCost, "Accuracy", trainAccuracy)
    print("")
    print("Val cost", valCost, "Accuracy", valAccuracy)