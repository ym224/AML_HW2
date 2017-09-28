import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import chain

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation

train_file = 'train.json'
test_file = 'test.json'

def load_data(filename):
	train_data = pd.read_json(filename)
	return train_data

def getSamplesAndCategories(data):
	sampleSize = len(data.index)
	labels = data.cuisine
	categorySize = len(labels.unique())
	return sampleSize, categorySize

def getAllIngredients(data):
	return set(chain.from_iterable(data.ingredients))

def transformIngredientsAndLabels(data, ingredients):
	labelEncoder = LabelEncoder()
	# use label encoder to fit and transform cuisine labels into numerical values
	labels = labelEncoder.fit_transform(data.cuisine)
	enc = CountVectorizer(vocabulary=ingredients, tokenizer=lambda x : x.split('.'))
	ingredients = map(lambda r: ".".join(r), data.ingredients)
	ingredients = enc.fit_transform(ingredients)
	return ingredients, labels
 
def transformIngredients(data, ingredients):
	enc = CountVectorizer(vocabulary=ingredients, tokenizer=lambda x : x.split('.'))
	ingredients = map(lambda r: ".".join(r), data.ingredients)
	return enc.fit_transform(ingredients)

def performKFoldBayes(data, labels, prior):
	train_indices = []
	test_indices = []
	for train_index, test_index in cross_validation.KFold(data.shape[0], n_folds=3):
	    train_indices.append(train_index)
	    test_indices.append(test_index)
	if (prior == 'Bernoulli'):
		classifier = BernoulliNB()
	else:
		classifier = GaussianNB()
	classifier.fit(data[train_index].toarray(), labels[train_index])
	print (classifier.score(data[test_index].toarray(), labels[test_index]))

def performKFoldLogisticRegression(data, labels):
	train_indices = []
	test_indices = []
	for train_index, test_index in cross_validation.KFold(data.shape[0], n_folds=3):
	    train_indices.append(train_index)
	    test_indices.append(test_index)
	    logisticRegression = LogisticRegression()
	    logisticRegression.fit(data[train_index].toarray(), labels[train_index])
	print (logisticRegression.score(data[test_index].toarray(), labels[test_index]))

def performLogisticRegression(train_ingreidents, train_labels, test_ingredients):
	mdl = LogisticRegression()
	mdl.fit(train_ingredients, train_labels)
	return mdl.predict(test_ingredients)

def saveResults(train_data, test_data, predictions):
	labelEncoder = LabelEncoder()
	labelEncoder.fit(train_data.cuisine)
	predictions = labelEncoder.inverse_transform(predictions)
	results = np.column_stack((test_data.id, predictions))
	file = open('test_cooking_predictions.csv', 'w')
	file.write('id,cuisine')
	for test_id, cuisine in results:
		file.write('\n')
		file.write(str(test_id) + ',' + cuisine)
	file.close()

train_data = load_data(train_file)
test_data = load_data(test_file)
sampleSize, categorySize = getSamplesAndCategories(train_data)
# 39774 samples and 20 categories
ingredients = getAllIngredients(train_data)
# 6714 unique ingredients
print (len(ingredients))

train_ingredients, train_labels = transformIngredientsAndLabels(train_data, ingredients)
print (train_ingredients.shape, train_labels.shape)
# perform K fold with Naiive Bayes Classifier with Bernoulli prior
performKFoldBayes(train_ingredients, train_labels, 'Bernoulli')
# perform K fold with Naiive Bayes Classifier with Gaussian prior
performKFoldBayes(train_ingredients, train_labels, '')
performKFoldLogisticRegression(train_ingredients, train_labels)

test_ingredients = transformIngredients(test_data, ingredients)
predictions = performLogisticRegression(train_ingredients, train_labels, test_ingredients)
saveResults(train_data, test_data, predictions)
