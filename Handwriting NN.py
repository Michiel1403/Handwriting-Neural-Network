import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plot


class Network(object):
	def __init__(self,sizes):
		self.numNeurons = len(sizes)
		self.sizes = sizes
		self.bias = [np.random.randn(y,1) for y in sizes[1:]]
		self.weight = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

	def feedForward(self,a):
		for w,b in zip(self.weight, self.bias):
			a = sigmoid(np.dot(w,a) + b)
		return a

	def stochasticGradientDescent(self,trainingData, epochs, miniBatchSize, learningRate, testData=None):
		if testData: nTest = len(testData)
		n = len(trainingData)
		for j in range(epochs):
			random.shuffle(trainingData)
			miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0,n,miniBatchSize)]
		for miniBatch in miniBatches:
			self.updateMiniBatch(miniBatch,learningRate)
		if testData:
			print ("Epoch {0}: {1} / {2}".format(j,self.evaluate(testData), nTest))
		else:
			print ("Epoch {0} complete".format(j))

	def updateMiniBatch(self, miniBatch, learningRate):
		nablaB = [np.zeros(b.shape) for b in self.bias]
		nablaW = [np.zeros(w.shape) for w in self.weight]
		for x,y in miniBatch:
			deltaNablaB, deltaNablaW = self.backProp(x,y)
			nablaB = [(nb+dnb) for nb, dnb in zip(nablaB, deltaNablaB)]
			nablaW = [(nw+dnw) for nw, dnw in zip(nablaW, deltaNablaW)]
		self.weight = [w-(learningRate/len(miniBatch)) * nw for w,nw in zip(self.weight, nablaW)]
		self.bias = [b-(learningRate/len(miniBatch)) * nb for b,nb in zip(self.bias,nablaB)]

	def backProp(self, x, y):
		print(x.shape)
		nablaB = [np.zeros(b.shape) for b in self.bias]
		nablaW = [np.zeros(w.shape) for w in self.weight]
		activation = x
		activations = [x]
		zs = []
		for b, w in zip(self.bias,self.weight):
			z = np.dot(w,activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		delta = self.costDerivative(activations[-1], y) * sigmoidPrime(zs[-1])
		nablaB[-1] = delta
		nablaW[-1] = np.dot(delta, activations[-2].transpose())
		for l in range(2, self.numNeurons):
			z = zs[-l]
			sp = sigmoidPrime(z)
			print(sp)
			delta = np.dot(self.weight[-l+1].transpose(), delta) * sp #error
			nablaB[-l] = delta
			nablaW[-l] = np.dot(delta, activation[-l-1].transpose())
		return (nablaB, nablaW)

	def evaluate(self,testData):
		testResults = [(np.argmax(self.feedForward(x)), y) for x,y in testData]
		return sum(int(x==y) for x,y in testResults)

	def costDerivative(self, outputActivations, y):
		return (outputActivations-y)

def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

def sigmoidPrime(z):
	return sigmoid(z) * (1 - sigmoid(z))

(xTrain,yTrain),(xTest, yTest) = tf.keras.datasets.mnist.load_data()
plot.imshow(xTrain[0],cmap="Greys")
flattened = xTrain[0].flatten()
trainingData=[]
testData=[]
for x,y in zip(xTrain,yTrain):
	arr = np.reshape(x,784), y
	trainingData.append(arr)

for x,y in zip(xTest,yTest):
	arr = np.reshape(x,784),y
	testData.append(arr)
net = Network([784,30,10])
net.stochasticGradientDescent(trainingData,30,10,3,testData=testData)
#plot.show()
