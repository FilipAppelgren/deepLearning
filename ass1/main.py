import numpy as np
import random as rand
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

k = 0
d = 0
n = 0

def init(batch):
	x, labels = unpickle("../datasets/cifar-10-batches-py/"+batch)
	return x, labels

def unpickle(file):
    
    with open(file, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
        x = batch['data'].T
        labels = batch['labels']
    return x, labels

def evaluateClassifier(x, w, b):
    k = 10
    d = len(x)
    n = len(x[0])
    P = np.zeros((n, k))
    for i in range(len(x[0])):
        temp = x.T[i].reshape(d, 1)
        s = w@temp + b
        p = softMax(s)
        p = p.reshape(k)
        P[i] = p
    return P

def crossLoss(x, y, w, b):
    k = 10
    n = len(x[0])

    p = evaluateClassifier(x, w, b)
    cross = np.zeros((1, n))

    for i in range(len(p)):
        temp = p[i].reshape(k, 1).T
        temp_one = y.T[i].reshape(k, 1)
        cross[0, i] = -np.log(temp@temp_one)

    return cross, p

def computeCost(x, y, w, b, reg):
	
	loss, p = crossLoss(x, y, w, b)
	w_2 = np.square(w)
	return (1/len(x[0]) * sum(loss[0]) + reg * sum((sum(w_2)))), p


def softMax(x):
	""" Standard definition of the softmax function """    
	return np.exp(x) / np.sum(np.exp(x), axis=0)


def computeAccuracy(p, y, w, b):
	n = len(y[0])
	results = np.amax(p, axis=1)
	counter = 0

	for row, result in enumerate(results):
		if y.T[row][np.where(p[row] == result)[0]] == 1:
			counter += 1
	return (counter / n) * 100


def computeGradients(x, y, p, w, b, reg):

    n = len(x[0])

    g = -(y-p)
    grad_w = (1/n) * (g@x.T)
    grad_b = np.mean(g, keepdims=True)

    grad_w += 2*reg*w

    return grad_w, grad_b

def ComputeGradsNum(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))


    c, p = computeCost(X, Y, W, b, lamda)

    for i in range(len(b)):
        #print(len(b))
        b_try = np.array(b)

        b_try[i] += h
        c2, p = computeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
        print(W.shape[0])
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i,j] += h
            c2, p = computeCost(X, Y, W_try, b, lamda)
            grad_W[i,j] = (c2-c) / h

    return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1, p = computeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2, p = computeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1, p = computeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2, p = computeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]

'''def montage(W):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            im  = W[i+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    plt.savefig("images"+str(eta)+"_lambda=" + str(reg) + "_epochs" + str(n_epochs)+".pdf")
    plt.show()'''

def montage(weights):
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    images = []
    for i in range(weights.shape[0]):
        im = weights[i].reshape((32, 32, 3), order='F')
        images.append((im - im.min()) / (im.max() - im.min()))
        images[i] = np.rot90(images[i], -1)

    columns = 5
    rows = 2
    fig=plt.figure(figsize=(columns, rows), dpi= 100)
    plt.rc('axes', titlesize=10)
    plt.rc('figure', titlesize=12)

    for i in range(1, columns*rows + 1):
        try:
            fig.add_subplot(rows, columns, i)
            plt.imshow(images[i - 1])
            plt.axis('off')
            plt.title(labels[i - 1], y=-0.3)
        except:
            continue
    plt.suptitle("Units", y=1.1)
    plt.show()

def createLabels(labels):
    one_hot = np.zeros((10, 10000))
    for column, label in enumerate(labels):
        one_hot[label][column] = 1
    return one_hot

def normalize(x):
    # Calculate mean and std for every pixel
    pixel_means = np.mean(x, axis=1).reshape(d, 1)
    pixel_std = np.std(x, axis=1).reshape(d, 1)
    
    # Normalize by subtracting the mean and dividing by std. Linear transforms.
    x = x - pixel_means
    x = x / pixel_std
    return x

def trainSplit(x, one_hot, ratio):
    x_train = x[0:, 0:ratio*100]
    x_val = x[0:, ratio*100:10000]
    
    one_hot_train = one_hot[0:, 0:ratio*100]
    one_hot_val = one_hot[0:, ratio*100:10000]

    return x_train, x_val, one_hot_train, one_hot_val

if __name__ == "__main__":

    x, labels = init("data_batch_1")
    one_hot = createLabels(labels)
    k = 10
    d = len(x)
    n = len(x[0])
    # x contains 10 000 pictures. These are represented by 3072 values in each slot in the array.
    # The first 1024 values per slot represents all the red values in the picture, going row by row.
    # The next 1024 are green values, and then blue.

    x = normalize(x)

    # Rows represent labels, and columns represent pixels. Their values are weights.
    w = np.random.normal(0, (1/np.sqrt(d))**2, (k, d))

    #w = np.random.normal(0, 1, (k, d))
    # Rows represent labels, one value per label.
    b = np.random.normal(0, 0.01**2, (k, 1))
    
    # p contains all probabilites for each picture. Transposed has 10 rows, 10 000 columns. 
    # Each column is one picture, row represents probability of pic.
    #Slicing
    n_batch = 100
    eta = 0.1
    n_epochs = 40
    reg = 1
    h = 1e-6

    percentages_val = np.zeros(n_epochs)
    costs_val = np.zeros(n_epochs)

    percentages_train = np.zeros(n_epochs)
    costs_train = np.zeros(n_epochs)

    '''for counter in range(1, 6):

        print("data_batch_"+str(counter))
        x, labels = init("data_batch_"+str(counter))
        one_hot = createLabels(labels)
        x = normalize(x)
        x_train, x_val, one_hot_train, one_hot_val = trainSplit(x, one_hot, ratio=90)

        val_cost, val_p = computeCost(x_val, one_hot_val, w, b, reg)
        train_cost, train_p = computeCost(x_train, one_hot_train, w, b, reg)

        print("Val", val_cost)
        print("Train", train_cost)'''
    
    x_train, x_val, one_hot_train, one_hot_val = trainSplit(x, one_hot, ratio=90)

    for i in tqdm(range(n_epochs)):

        val_cost, val_p = computeCost(x_val, one_hot_val, w, b, reg)
        #val_percentage = computeAccuracy(val_p, one_hot_test, w, b)
        #percentages_test[i] = test_percentage
        costs_val[i] = val_cost

        train_cost, train_p = computeCost(x_train, one_hot_train, w, b, reg)
        #train_percentage = computeAccuracy(train_p, one_hot_train, w, b)
        #percentages_train[i] = train_percentage
        costs_train[i] = train_cost


        for j in range(0, len(x), n_batch):
            x_temp = x_train[0:, j:j+n_batch]
            one_hot_temp = one_hot_train[0:, j:j+n_batch]
                        
            cost, p = computeCost(x_temp, one_hot_temp, w, b, reg)
            p = p.T
            
            grad_w, grad_b = computeGradients(x_temp, one_hot_temp, p, w, b, reg)
            w = w - eta * grad_w
            b = b - eta * grad_b
            num_grad_w, num_grad_b = ComputeGradsNum(x_temp, one_hot_temp, p, w, b, reg, h)
            mse_b = np.sum(np.square(grad_b - num_grad_b))
            mse_w = np.sum(np.square(grad_w - num_grad_w))

        eta = eta * 0.9
    #print(num_grad_w,  num_grad_b)

    montage(w)
    x_test, labels_test = init("test_batch")
    one_hot_test = createLabels(labels_test)
    x_test = normalize(x_test)
    test_cost, test_p = computeCost(x_test, one_hot_test, w, b, reg)
    test_percentage = computeAccuracy(test_p, one_hot_test, w, b)
    #f = plt.figure()
    #x_p = np.arange(0, 100, 2.5)
    #x_c = np.arange(0, 60, 1.5)

    #plt.plot(x_c, test_cost, 'r', label="Test")
    #plt.plot(x_c, costs_train, 'b', label="Training")

    #plt.xlabel("Epochs")
    #plt.ylabel("Cost")
    print(test_cost)
    print(test_percentage)
    #print(percentages_test)
    #plt.figtext(0.3, 0.8, "eta = " + str(eta) + " epochs = " + str(n_epochs) + "\n" + "batch = " + str(n_batch) + " reg = " + str(reg))
    #plt.legend()
    #plt.savefig("eta="+str(eta)+"_lambda"+str(reg)+"_epochs="+str(n_epochs)+".pdf")
    #plt.show()