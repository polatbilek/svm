import os
from tqdm import tqdm
import numpy as np
import sys

def read_vectors(path):

	user_vectors = []

	file_handler = open(path, "r")

	for line in file_handler:
		data = []

		data.append(str(line.strip().split(":::")[0]))
		data.append(list(map(float, line.strip().split(":::")[1].split(","))))
		data.append([int(line.strip().split(":::")[2].split(",")[0]), int(line.strip().split(":::")[2].split(",")[1])])

		user_vectors.append(data)

	file_handler.close()

	return user_vectors


def combine(training_ngrams, training_users, test_ngrams, test_users, training_rnn_vectors, test_rnn_vectors):

	training_vectors = []
	training_targets = []
	test_vectors = []
	test_targets = []

	print(np.shape(training_rnn_vectors))
	print(np.shape(test_rnn_vectors))

	for i in tqdm(range(len(training_users))):
		for rnn_vectors in training_rnn_vectors:
			if rnn_vectors[0] == training_users[i][0]:
				concatenated_vector = list(training_ngrams[i])+list(rnn_vectors[1])
				training_vectors.append(concatenated_vector)
				training_targets.append(training_users[i][1])


	for i in tqdm(range(len(test_users))):
		for rnn_vectors in test_rnn_vectors:
			if rnn_vectors[0] == test_users[i][0]:
				concatenated_vector = list(test_ngrams[i]) + list(rnn_vectors[1])
				test_vectors.append(concatenated_vector)
				test_targets.append(test_users[i][1])

	return training_vectors, training_targets, test_vectors, test_targets