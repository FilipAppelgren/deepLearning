import numpy as np
import pickle

k = 0
d = 0
n = 0

def init():
	x, labels = unpickle("datasets/cifar-10-batches-py/data_batch_1")
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
	return (1 - (counter / n)) * 100


def computeGradients(x, y, p, w, b, reg):

	n = len(x[0])

	g = -(y-p)
	grad_w = (1/n) * (g@x.T)
	grad_b = np.mean(g, axis=-1, keepdims=True)
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
		b_try = np.array(b)
		b_try[i] += h
		c2, p = computeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
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

def montage(W):
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
	plt.show()
