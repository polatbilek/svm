from read_file import read_vectors
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from ngram_model import get_ngrams

path_to_vectors_test = "/home/darg2/PycharmProjects/svm_gender_prediction/vectors_rnnwa_test.txt"
path_to_vectors_training = "/home/darg2/PycharmProjects/svm_gender_prediction/vectors_rnnwa_training.txt"
kernels = ['linear', 'rbf', 'poly']

training_ngrams, training_users, test_ngrams, test_users = get_ngrams()

clf = svm.SVC(kernel=kernels[2])
clf.fit(training_ngrams, training_users)

predictions = clf.predict(test_ngrams)

print(accuracy_score(test_users, predictions))



