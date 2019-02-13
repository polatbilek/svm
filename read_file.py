import os

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