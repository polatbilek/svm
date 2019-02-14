from preprocess import readData
from nltk import ngrams
from preprocess import FLAGS
import nltk
from tqdm import tqdm
import operator
import numpy as np
import sys

ngram_dict = {}

def preprocess(tweet_list, users, target_values, mode="word"):
	new_list = []

	for tweet in tweet_list:
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

	if mode == "char":
		user_tweets = []
		for user_text in user_tweet_text:
			char_tweets = []
			for i in range(len(user_text)): # user_text[i] represents a word in tweet
				for char in user_text[i]:
					char_tweets.append(char)

				if user_text[i] != "<ENDOFTWEET>":
					if user_text[i+1] != "<ENDOFTWEET>":
						char_tweets.append(" ")

			user_tweets.append(char_tweets)

		user_tweet_text = user_tweets

	user_list = []
	user_count = 0

	for user in users:
		if user_count% 100 == 0:
			user_list.append([user, target_values[user][0]])

		user_count += 1

	return user_tweet_text, user_list


############################################################################################
def ngram_extract(tweets, n, mode):
	global ngram_dict
	df_dict = {} #document freq dict

	if mode == "training":
		for i in tqdm(range(len(tweets))):
			output = list(ngrams(tweets[i], n))
			fdist = nltk.FreqDist(output)

			for k, v in fdist.items():
				ngram_dict[k] = 0

				try:
					df_dict[k] = df_dict[k] + 1
				except:
					df_dict[k] = 1


		new_dict = ngram_dict.copy()

		for k, v in df_dict.items():
			if v < 3:
				new_dict.pop(k)

		ngram_dict = new_dict

	user_ngram_vectors = []

	for i in tqdm(range(len(tweets))):
		dict = ngram_dict.copy()
		output = list(ngrams(tweets[i], n))
		fdist = nltk.FreqDist(output)

		for k, v in fdist.items():
			try:
				dict[k] = dict[k] + int(v)
			except:
				pass

		user_ngram_vectors.append(list(dict.values()))

	return user_ngram_vectors


############################################################################################
def get_ngrams(n, mode="word"):

	training_tweets, training_users, training_target_values = readData(FLAGS.training_data_path)
	test_tweets, test_users, test_target_values = readData(FLAGS.test_data_path)

	training_tweets, training_users = preprocess(training_tweets, training_users, training_target_values, mode)
	test_tweets, test_users = preprocess(test_tweets, test_users, test_target_values, mode)


	training_ngrams = ngram_extract(training_tweets, n, mode="training")
	test_ngrams = ngram_extract(test_tweets, n, mode="test")


	return list(training_ngrams), list(training_users), list(test_ngrams), list(test_users)

