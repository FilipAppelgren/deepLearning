import pickle
import numpy as np
import random
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
    y = np.zeros((10, len(labels)))
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

def initBiasWeight(n_Layers):
    w = []
    b = []

    w_1 = np.random.normal(0, (1/np.sqrt(d))**2, (m, d))
    b_1 = np.zeros((m, 1))

    w.append(w_1)
    b.append(b_1)

    for _ in range(n_Layers - 2):
        w_k = np.random.normal(0, (1/np.sqrt(d))**2, (m, m))
        b_k = np.zeros((m, 1))
        w.append(w_k)
        b.append(b_k)

    w_l = np.random.normal(0, (1/np.sqrt(d))**2, (k, m))
    b_l = np.zeros((k, 1))
    w.append(w_l)
    b.append(b_l)

    return w, b


def crossLoss(x, w, b, y):

    p, h_arr = evaluateClassifier(x, w, b)
    l_cross = (p * y).sum(axis=0)
    l_cross[l_cross == 0] = np.finfo(float).eps
    '''cross = np.zeros((1, len(p[0])))

    for i in range(len(p[0])):
        temp = p.T[i].reshape(k, 1)
        temp = temp.T
        temp_one = y.T[i].reshape(k, 1)
        cross[0, i] = -np.log(temp@temp_one)'''

    return l_cross, p, h_arr

def computeCost(x, y, w, b, reg):
	
    loss, p, h_arr = crossLoss(x, w, b, y)
    w_sum = 0
    for w_i in w:
        w_sum += np.sum(np.square(w_i))
    #print("Loss", loss)
    #print("W_sum", w_sum)
    #return (1/len(x[0]) * np.sum(loss)) + (reg * w_sum), p, h_arr
    return np.sum(-np.log(loss) / len(x[0])) + (reg * w_sum), p, h_arr
    

def computeGradsNum(x, y, w, b, reg, h):

    
    biasGrad = []
    weightGrad = []
    c = computeCost(x, y, w, b, reg)[0]

    b_try = np.copy(b)

    for i, bias in enumerate(b_try):
        grad_bias = np.zeros_like(bias)
        for j in range(0, len(b_try)):
            b_try[i][j] += h
            c2 = computeCost(x, y, w, b_try, reg)[0]
            grad_bias[j] = (c2 - c)/h
            b_try[i][j] -= h
        biasGrad.append(grad_bias)

    w_try = np.copy(w)
    for k, weight in enumerate(w_try):
        grad_weight = np.zeros_like(weight)
        for i in range(0, len(weight)):
            for j in range(0, len(weight[0])):
                w_try[k][i][j] += h
                c2 = computeCost(x, y, w_try, b, reg)[0]
                grad_weight[i][j] = (c2 - c) / h
                w_try[k][i][j] -= h
        weightGrad.append(grad_weight)

    return biasGrad, weightGrad


def computeAccuracy(p, y):
    y = np.argmax(y, axis=0)
    temp = np.sum(np.argmax(p, axis=0) == y)/len(y)

    return temp


def yoMamaComputeAccuracy(p, y):
    n = len(y[0])
    results = np.amax(p, axis=0)

    counter = 0

    for row, result in enumerate(results):
        if y.T[row][np.where(p.T[row] == result)[0]] == 1:
            counter += 1
    return (counter / n) * 100


def computeGradients(x, y, p, h_arr, w, b, reg):

    n = len(x[0])
    dL_dW = [0] * len(w)
    dL_dB = [0] * len(b)
    g = -(y - p)

    for l in range(len(w) - 1, 0, -1):
        dL_dW_l = (g@h_arr[l - 1].T) / n
        dL_dW_l += 2 * reg * w[l]
        dL_db_l = g@np.ones((n, 1)) / n

        g = w[l].T@g

        h_temp = invRelu(h_arr[l - 1])
        g = g * h_temp

        dL_dW[l] = dL_dW_l
        dL_dB[l] = dL_db_l

    dL_dW_1 = (g@x.T) / n
    dL_dW_1 += 2 * reg * w[0]
    dL_db_1 = g@np.ones((n, 1)) / n
    dL_dW[0] = dL_dW_1
    dL_dB[0] = dL_db_1

    return dL_dW, dL_dB


def evaluateClassifier(x, w, b):

    h_list = []
    w_1 = w[0]
    b_1 = b[0]

    s_1 = (w_1@x) + b_1
    h_1 = ReLU(s_1)
    h_list.append(h_1)
    w_cut = w[1:len(w) - 1]

    for l, w_l in enumerate(w_cut):
        s_l = w_l@h_list[l]
        s_l += b[l + 1]
        h_l = ReLU(s_l)
        h_list.append(h_l)
        
    k = len(w) - 1
    s_k = w[k]@h_list[k - 1]
    s_k += b[k]
    p_k = softMax(s_k)
    return p_k, h_list

def invRelu(s):
    return np.where(s > 0, 1, 0)

def ReLU(s):
    return np.where(s > 0, s, 0)

def softMax(x):
    ex = np.exp(x)
    return ex / ex.sum(axis=0)

def split(x, y, ratio):
    x = x[0:, 0:100]
    y = y[0:, 0:100]

    return x, y

def accuracyCost(trainX, trainY, valX, valY, w, b, reg):

    trainCost, trainP = computeCost(trainX, trainY, w, b, reg)[0:2]
    valCost, valP = computeCost(valX, valY, w, b, reg)[0:2]
    trainAccuracy = computeAccuracy(trainP, trainY)
    valAccuracy = computeAccuracy(valP, valY)

    return trainCost, valCost, trainAccuracy, valAccuracy

if __name__ == "__main__":

    n_Layers = 4

    trainX, trainY = init("data_batch_1")
    valX = trainX[0:, 5000:10000]
    valY = trainY[5000:10000]

    trainX = trainX[0:, 0:5000]
    trainY = trainY[0:5000]
    m = 50
    
    #trainX = trainX[0:100, 0:100]
    #trainLabels = trainLabels[0:100]
    #valX = valX[0:100, 0:100]
    #valY = valY[0:100]
    d = len(trainX)
    k = 10
    #h = 1e-6
    

    #for i in range(2, 6):
    #    tempX, tempLabels = init("data_batch_"+str(i))
    #    trainX = np.append(trainX, tempX, axis=1)
    #    trainLabels = trainLabels + tempLabels
    
    batch = 100
    n = len(trainX[0])
    eta_min = 1e-5
    eta_max = 1e-1
    eta = eta_min
    n_s = 5 * 45000 / batch
    cycles = 8
    h = 1e-6
    
    minmax_epochs = (n_s/(n / batch))*2

    n_epochs = minmax_epochs * cycles 

    trainY = createLabels(trainY)    
    trainX = normalize(trainX)
    valY = createLabels(valY)
    valX = normalize(valX)

    searchRuns = 1
    for searches in range(searchRuns):
        t = 1
        l = 0
        w, b = initBiasWeight(n_Layers)
        #reg = 0.00045
        reg = 0.0

        trainCost, valCost, trainAccuracy, valAccuracy = accuracyCost(
            trainX, trainY, valX, valY, w, b, reg)

        print("Lambda", "{0:.5g}".format(reg))
        #print("RegP", regP)
        print("Init train cost", "{0:.5g}".format(trainCost), "Accuracy", "{0:.5g}".format(trainAccuracy))
        print("Init val cost", "{0:.5g}".format(valCost), "Accuracy", "{0:.5g}".format(valAccuracy))

        for i in range(int(n_epochs)):
            if(i % minmax_epochs) == 0 and i != 0:
                l += 1

            for j in range(0, n, batch):
                #print(j)
                if t >= (2 * l * n_s) and t <= (2 * l + 1)*n_s:
                    eta = eta_min + (t - 2*l*n_s)/n_s * (eta_max - eta_min)
                    
                elif t >= (2 * l + 1)*n_s and t <= 2 * (l + 1) * n_s:
                    eta = eta_max - (t-(2*l + 1)*n_s)/n_s * (eta_max - eta_min)
            
                x_temp = trainX[0:, j:j+batch]
                y_temp = trainY[0:, j:j+batch]

                cost, p, h_arr = computeCost(x_temp, y_temp, w, b, reg)
                dL_dw, dL_db = computeGradients(x_temp, y_temp, p, h_arr, w, b, reg)
                #biasGrads, weightGrads = computeGradsNum(x_temp, y_temp, w, b, reg, h)

                #for i, num in enumerate(biasGrads):
                #    print("Bias", i, np.sum(np.square(num)) - (np.sum(np.square(dL_db[i]))))

                #for i, num in enumerate(weightGrads):
                #    print("Weight", i, np.sum(np.square(num)) - (np.sum(np.square(dL_dw[i]))))
                
                #print(np.sum(np.square(dL_dw[0])))
                for i, (w_i, b_i) in enumerate(zip(w, b)):
                    #print(i)
                    #print(dL_db[n_Layers - 1])
                    eta = 0.001
                    w[i] = w[i] - (eta*dL_dw[i])
                    b[i] = b[i] - (eta*dL_db[i])
                    
                t+=1
                #print(eta)
        
        trainCost, valCost, trainAccuracy, valAccuracy = accuracyCost(trainX, trainY, valX, valY, w, b, reg)
        print("")
        print("Train cost", "{0:.5g}".format(trainCost), "Accuracy", "{0:.5g}".format(trainAccuracy))
        print("Val cost", "{0:.5g}".format(valCost), "Accuracy", "{0:.5g}".format(valAccuracy))
        print("")
        print("Cycles", cycles)
        print("=================================")