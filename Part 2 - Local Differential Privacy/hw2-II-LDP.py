import os
import sys
import pandas as pd
import numpy as np
import urllib.request
import math
from collections import Counter
from matplotlib import pyplot as plt

def data_loader():

	data_folder = sys.path[0] + '/data'
	if not os.path.exists(data_folder):
		os.makedirs(data_folder)

		url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
		url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
		f_train = os.path.join(data_folder, "adult_train.csv")
		f_test = os.path.join(data_folder, "adult_test.csv")
		urllib.request.urlretrieve(url_train, f_train)
		urllib.request.urlretrieve(url_test, f_test)


	adult_train = pd.read_csv('./data/adult_train.csv')
	headers =  ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
				'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
	adult_train.columns = headers
	adult_test = pd.read_csv('./data/adult_test.csv')
	adult_test.columns = headers
	adult = pd.concat([adult_train,adult_test])

	QI = adult[['age']]

	return QI


def cal_prob_unary(epsilon):
	p = math.exp(epsilon/2)/(math.exp(epsilon/2)+1)
	return p


#generalized RR
def RR_categorize(df):

	ranges = [i for i in range(15, 95, 5)]
#	print(ranges, len(ranges))

	labels = [str(i) for i in range(len(ranges)-1)]
	df['bins'] = pd.cut(df['age'], bins=ranges, labels=labels)
	return df, labels


def cal_prob_RR(epsilon, d):
	p = math.exp(epsilon)/(math.exp(epsilon)+d-1)
	q = 1/(math.exp(epsilon)+d-1)
	return p, q


def rand_res(response, labels, p, q):

	sample = np.random.random()
	if sample <= p:
		return response
	else:
		others = labels[:]
		others.remove(response)	
		return np.random.choice(others)

def aggregate(responses, p, q, labels):
	sums = pd.Series(responses).value_counts().reindex(labels, fill_value=0)
	n = len(responses)
	estimator = [(v - n*q)/(p-q) for v in sums]
	return estimator	


#Unary coding	
def encode(response, labels):
	one_hot = [1 if d == response else 0 for d in labels]
	return one_hot

def perturb(encoded_response, p):	
	return [perturb_bit(b, p) for b in encoded_response]

def perturb_bit(bit, p):
	sample = np.random.random()
	if bit == 1:
		return 1 if sample <= p else 0
	elif bit == 0:
		return 1 if sample >= p else 0

def unary_aggregate(responses, p):
	sums = np.sum(responses, axis=0)
#	print(sums, len(sums))
	n = len(responses)
	return [(v - n*(1-p))/(2*p-1) for v in sums] 


def main():
	output = []
	output1 = sys.path[0] + '/results_lap_0.5.csv'
	
	df = data_loader()

	df, labels = RR_categorize(df)
	print(df.head(), df.shape)
	true_stats = df.bins.value_counts()[labels]

	#experiments on epsilon
	epsilons = [i+1 for i in range(10)]
	dists_00 = []
	dists_10 = []

	for epsilon in epsilons:
		p, q = cal_prob_RR(epsilon, len(labels))
		responses = [rand_res(r, labels, p, q) for r in df['bins']]
		stats = aggregate(responses, p, q, labels)
		dist = np.linalg.norm(stats-true_stats, 1)
		dists_00.append(dist)
#	print(dists_00)

	
	for epsilon in epsilons:
		p = cal_prob_unary(epsilon)
		responses = [perturb(encode(r, labels), p) for r in df['bins']]
		stats = unary_aggregate(responses, p)
		dist = np.linalg.norm(stats-true_stats, 1)
		dists_10.append(dist)
#	print(dists_10)
	plt.plot(epsilons, dists_00, label="RR")
	plt.plot(epsilons, dists_10, label="unary")
	plt.legend()
	plt.xlabel("Epsilon")
	plt.ylabel("L1-distance")
	plt.show()

	fractions = [0.1*(i+1) for i in range(10)]
	epsilon = 2
	dists_01 = []
	
	#experiments on records#
	for frac in fractions:
		df_frac = df.sample(frac=frac)
		df_frac, labels = RR_categorize(df_frac)
		true_stats = df_frac.bins.value_counts()[labels]
	#	print(true_stats)
		p, q = cal_prob_RR(epsilon, len(labels))
		responses = [rand_res(r, labels, p, q) for r in df_frac['bins']]
		stats = aggregate(responses, p, q, labels)
	#	print(stats)
		dist = np.linalg.norm(stats-true_stats, 1)
		dists_01.append(dist)

	dists_11 = []
	for frac in fractions:
		df_frac = df.sample(frac=frac)
		df_frac, labels = RR_categorize(df_frac)
		true_stats = df_frac.bins.value_counts()[labels]
		p = cal_prob_unary(epsilon)
		responses = [perturb(encode(r, labels), p) for r in df_frac['bins']]
#		print(responses)
		stats = unary_aggregate(responses, p)
		dist = np.linalg.norm(stats-true_stats, 1)
		dists_11.append(dist)
#	print(dists_11)
#	print(df.shape[0])
	plt.plot([frac*df.shape[0] for frac in fractions], dists_01, label="RR")
	plt.plot([frac*df.shape[0] for frac in fractions], dists_11, label="unary")
	plt.legend()
	plt.xlabel("#record")
	plt.ylabel("L1-distance")
	plt.show()


if __name__ == "__main__":
	main()