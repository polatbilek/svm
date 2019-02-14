from read_file import read_vectors
from read_file import combine
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from ngram_model import get_ngrams
from sklearn.decomposition import TruncatedSVD

log_file = "/home/darg2/Desktop/svm_log.txt"

path_to_vectors_test = "/home/darg2/PycharmProjects/svm_gender_prediction/vectors_rnnwa_test.txt"
path_to_vectors_training = "/home/darg2/PycharmProjects/svm_gender_prediction/vectors_rnnwa_training.txt"

training_rnn_vectors = np.asarray(read_vectors(path_to_vectors_training))
test_rnn_vectors = np.asarray(read_vectors(path_to_vectors_test))

kernels = ['linear', 'rbf', 'poly']
ns = [3, 4, 5]
modes = ["char", "word"]

for mode in modes:
	for n in ns:
		try:
			del training_ngrams
			del training_users
			del test_ngrams
			del test_users
			del clf
			del predictions
		except:
			pass

		training_ngrams, training_users, test_ngrams, test_users = get_ngrams(n, mode=mode)

		svd_training = TruncatedSVD(n_components=300, n_iter=7)
		svd_test = TruncatedSVD(n_components=300, n_iter=7)

		training_ngrams = svd_training.fit_transform(training_ngrams)
		test_ngrams = svd_test.fit_transform(test_ngrams)


		training_vectors, training_targets, test_vectors, test_targets = combine(training_ngrams, training_users, test_ngrams, test_users, training_rnn_vectors, test_rnn_vectors)

		print(np.shape(training_vectors))
		print(np.shape(training_targets))
		print(np.shape(test_vectors))
		print(np.shape(test_targets))

		for kernel in kernels:
			clf = svm.SVC(kernel=kernel, gamma='auto')
			clf.fit(training_vectors, training_targets)

			predictions = clf.predict(test_vectors)

			file_handler = open(log_file, "a")

			line = "mode: " + str(mode) + ", n: " + str(n) + ", kernel: " + str(kernel)
			line = line + "\n" + "accuracy: " + str(accuracy_score(test_targets, predictions)) + "\n"
			print(line)

			file_handler.write(line)
			file_handler.close()



