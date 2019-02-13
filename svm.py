from read_file import read_vectors
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from ngram_model import get_ngrams

path_to_vectors_test = "/home/darg2/PycharmProjects/svm_gender_prediction/vectors_rnnwa_test.txt"
path_to_vectors_training = "/home/darg2/PycharmProjects/svm_gender_prediction/vectors_rnnwa_training.txt"
kernels = ['linear', 'rbf', 'poly']

training_vectors = np.asarray(read_vectors(path_to_vectors_training))
test_vectors = np.asarray(read_vectors(path_to_vectors_test))

X = training_vectors[:,1]
Y = np.asarray(training_vectors[:,2])

new_y = []
for element in Y:
	new_y.append(element[0])

Y = np.asarray(new_y)

X = list(X)
Y = list(Y)

clf = svm.SVC(kernel=kernels[2])
clf.fit(X, Y)

test_X = test_vectors[:,1]
test_Y = np.asarray(test_vectors[:,2])

new_y = []
for element in test_Y:
	new_y.append(element[0])

test_Y = np.asarray(new_y)

test_X = list(test_X)
test_Y = list(test_Y)

predictions = clf.predict(test_X)

print(accuracy_score(test_Y, predictions))



