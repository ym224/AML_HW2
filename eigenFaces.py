import numpy as np
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as cm

train_file = './faces/train.txt';
test_file = './faces/test.txt';

def loadData(filename):
	labels, data = [], []
	for line in open(filename):
	    im = misc.imread(line.strip().split()[0])
	    data.append(im.reshape(2500,))
	    labels.append(line.strip().split()[1])
	data, labels = np.array(data, dtype=float), np.array(labels, dtype=int)
	print (data.shape, labels.shape)
	return data, labels

def displayTrainSample(data):
	plt.imshow(data[10, :].reshape(50,50), cmap = cm.Greys_r)
	plt.title('Train Sample')
	plt.savefig('training_image.png')
	plt.close()

def displayTestSample(data):
	plt.imshow(data[10, :].reshape(50,50), cmap = cm.Greys_r)
	plt.title('Test Sample')
	plt.savefig('test_image.png')
	plt.close()	

def calculateAverageFace():
	mu = np.mean(train_data, axis=0)
	print (mu)
	plt.imshow(mu.reshape(50,50), cmap = cm.Greys_r)
	plt.title('Average face $\mu$')
	plt.savefig('average_image.png')
	return mu

def displayAdjustedData(mu):
	_train_data = train_data - mu
	_test_data = test_data - mu
	plt.figure(figsize=(6, 6))
	plt.subplot(2,2,1)
	plt.imshow(train_data[10, :].reshape(50,50), cmap = cm.Greys_r)
	plt.title("Train sample")

	plt.subplot(2,2,2)
	plt.imshow(_tr[10,:].reshape(50,50), cmap = cm.Greys_r)
	plt.title("Train sample - $\mu$")

	plt.subplot(2,2,3)
	plt.imshow(test_data[10,:].reshape(50,50), cmap = cm.Greys_r)
	plt.title("Test sample")

	plt.subplot(2,2,4)
	plt.imshow(_te[10,:].reshape(50,50), cmap = cm.Greys_r)
	plt.title("Test sample - $\mu$")

	plt.tight_layout()
	plt.show()


train_data, train_labels = loadData(train_file)
test_data, test_labels = loadData(test_file)
displayTrainSample(train_data)
displayTestSample(test_data)
calculateAverageFace()