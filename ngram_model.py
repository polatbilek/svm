from preprocess import readData
from nltk import ngrams
from preprocess import FLAGS
import nltk
from tqdm import tqdm
import operator
import numpy as np

def preprocess(tweet_list, users, target_values):
	new_list = []

	for tweet in tqdm(tweet_list):
		for word in tweet:
			if "https://" in word or "http://" in word:
				index = tweet.index(word)
				tweet.pop(index)
				tweet.insert(index, "<URL>")

			elif word.startswith("@"):
				index = tweet.index(word)
				tweet.pop(index)
				tweet.insert(index, "<MENTION>")

		tweet.append("<ENDOFTWEET>")
		new_list.append(tweet)

	count = 0
	concatenated_tweets = []
	user_tweet_text = []

	for tweet in new_list:
		if (count + 1) % 100 == 0:
			concatenated_tweets = concatenated_tweets + tweet
			user_tweet_text.append(concatenated_tweets)
			concatenated_tweets = []
		else:
			concatenated_tweets = concatenated_tweets + tweet

		count += 1

	user_list = []
	user_count = 0

	for user in users:
		if user_count% 100 == 0:
			user_list.append([user, target_values[user][0]])

		user_count += 1

	return user_tweet_text, user_list



def ngram_extract(tweets, n):
	ngram_dict = {}

	for i in tqdm(range(len(tweets))):
		output = list(ngrams(tweets[i], n))
		fdist = nltk.FreqDist(output)

		for k, v in fdist.items():
			ngram_dict[k] = 0

	user_ngram_vectors = []

	for i in tqdm(range(len(tweets))):
		dict = ngram_dict.copy()
		output = list(ngrams(tweets[i], n))
		fdist = nltk.FreqDist(output)

		for k, v in fdist.items():
			dict[k] = v

		user_ngram_vectors.append(dict.values())

	return user_ngram_vectors



def get_ngrams():
	n = 2

	training_tweets, training_users, training_target_values = readData(FLAGS.training_data_path)
	test_tweets, test_users, test_target_values = readData(FLAGS.test_data_path)

	training_tweets, training_users = preprocess(training_tweets, training_users, training_target_values)
	test_tweets, test_users = preprocess(test_tweets, test_users, test_target_values)


	training_ngrams = ngram_extract(training_tweets, n)
	test_ngrams = ngram_extract(test_tweets, n)

	return training_ngrams, training_users, test_ngrams, test_users

