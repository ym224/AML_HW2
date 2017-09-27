import numpy as np
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression

train_file = './faces/train.txt';
test_file = './faces/test.txt';

def loadData(filename):
	labels, data = [], []
	for line in open(filename):
	    im = misc.imread(line.strip().split()[0])
	    data.append(im.reshape(2500,))
	    labels.append(line.strip().split()[1])
	data, labels = np.array(data, dtype=float), np.array(labels, dtype=int)
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
	plt.imshow(mu.reshape(50,50), cmap = cm.Greys_r)
	plt.title('Average face $\mu$')
	plt.savefig('average_image.png')
	plt.close()
	return mu

def displayAdjustedData(train_data, test_data, _train_data, _test_data):
	plt.figure(figsize=(6, 6))
	plt.subplot(2,2,1)
	plt.imshow(train_data[10, :].reshape(50,50), cmap = cm.Greys_r)
	plt.title('Train Sample')

	plt.subplot(2,2,2)
	plt.imshow(_train_data[10,:].reshape(50,50), cmap = cm.Greys_r)
	plt.title('Train Sample - $\mu$')

	plt.subplot(2,2,3)
	plt.imshow(test_data[10,:].reshape(50,50), cmap = cm.Greys_r)
	plt.title('Test Sample')

	plt.subplot(2,2,4)
	plt.imshow(_test_data[10,:].reshape(50,50), cmap = cm.Greys_r)
	plt.title('Test Sample - $\mu$')

	plt.tight_layout()
	plt.savefig('original_and_adjusted_images.png')
	plt.close()

def performSVD(_train_data):
	return np.linalg.svd(_train_data)

def displayEigenfaces(eigenvectors):
	plt.figure(num=None, figsize=(80, 200), dpi=85, facecolor='w', edgecolor='k')
	for i in range(10):
		plt.subplot(5, 2, i+1)
		plt.imshow(eigenvectors[i,:].reshape(50,50), cmap = cm.Greys_r)
		if (i == 1):
			suf = 'st'
		elif (i == 2):
			suf = 'nd'
		elif (i == 3):
			suf = 'rd'
		else:
			suf = 'th'
		plt.subplots_adjust(top=.8)
		plt.title(str(i) + suf + ' eigenface', fontsize=20, y=2)
	plt.savefig('first_ten_eigenfaces.png')
	plt.close()

def computeRankApprox(U, S, VT, r):
	rankR = U[:, :r].dot(np.diag(S[:r]).dot(VT[:r, :]))
	return rankR

def computeRankApproxErr(U, S, VT, train_data):
	low_rank_errs = []
	for i in range(1, 201):
		normalized = np.linalg.norm(train_data - computeRankApprox(U, S, VT, i))
		low_rank_errs.append(normalized)
	return low_rank_errs

def plotRankApproxErrForRank(low_rank_errs):
    plt.figure(figsize=(6,4))
    plt.plot(range(1, 201), low_rank_errs, 'r', linewidth=2, color = "#0a306d")
    plt.xticks(range(0, 201, 50))
    plt.title("Rank-r Approximation Error", fontsize=14)
    plt.xlabel('r', fontsize=12)
    plt.ylabel('Approximation Error', fontsize=12)
    plt.tight_layout()
    plt.savefig('low_rank_approximation_err.png')
    plt.close()

def computeEigenfaceFeature(train_data, test_data, VT, r):
	train_eigenface_feature = train_data.dot(VT[:r, :].transpose())
	test_eigenface_feature = test_data.dot(VT[:r, :].transpose())
	return train_eigenface_feature, test_eigenface_feature

def trainLogisticRegression(train_ef_feature, test_ef_feature, train_labels, test_labels):
	model = LogisticRegression()
	model.fit(train_ef_feature, train_labels)
	return model.score(test_ef_feature, test_labels)

def plotClassificationAccuracy(train_data, test_data, VT, train_labels, test_labels):
	accuracies = []
	for r in range(1, 201):
		train_ef_feature, test_ef_feature = computeEigenfaceFeature(train_data, test_data, VT, r)
		score = trainLogisticRegression(train_ef_feature, test_ef_feature, train_labels, test_labels)
		accuracies.append(score)
	plt.figure()
	plt.plot(range(1, 201), np.array(accuracies))
	plt.xticks(range(0,201,50))
	plt.xlabel('Face Space')
	plt.ylabel('Classification Accuracy')
	plt.title('Classification Accuracy for Multiple Dimensions of Face Space')
	plt.savefig('face_recognition_classification_accuracy.png')
	plt.close()

# load train and test data into 1 dimensional arrays
train_data, train_labels = loadData(train_file)
test_data, test_labels = loadData(test_file)

# plot a sample from train data
displayTrainSample(train_data)
# plot a sample from test data
displayTestSample(test_data)

# calculate and display the average face mu in train data
mu = calculateAverageFace()

# subtract mu from train and test data
_train_data = train_data - mu
_test_data = test_data - mu

# plot original and adjusted samples from train and test data
displayAdjustedData(train_data, test_data, _train_data, _test_data)

# perform singular value decomposition on adjusted training data to get the eigenvalues and eigenvectors
U, S, VT = performSVD(_train_data)

# display first 10 eigenfaces (eigenvectors in V)
displayEigenfaces(VT)

# compute the low rank approximation errors for r in range 1:200
low_rank_errs = computeRankApproxErr(U, S, VT, _train_data)
# plot the low rank approximation errors as a function of r
plotRankApproxErrForRank(low_rank_errs)

# extract the eigenfeatures of train and data set for r=10
train_ef_feature, test_ef_feature = computeEigenfaceFeature(_train_data, _test_data, VT, 10) 
accuracy_score = trainLogisticRegression(train_ef_feature, test_ef_feature, train_labels, test_labels)
print ('accuracy score: ' + str(accuracy_score))

# plot the classification accuracy of face recognition for varying dimensions of face space
plotClassificationAccuracy(_train_data, _test_data, VT, train_labels, test_labels)