#!/usr/bin/env python

from os.path import isdir
from os import mkdir
from json import load

from numpy import array, linspace, sum, arange, log, pi, mean, var, sqrt
from numpy.random import seed, randint
from matplotlib import pyplot as plt
from scipy.special import beta, binom, factorial, loggamma
from scipy.optimize import minimize_scalar

data_dir = "data"
if not isdir(data_dir):
	print("Downloading data...")
	from urllib.request import urlretrieve
	mkdir(data_dir)
	for i in [1, 2, 3]:
		urlretrieve(f"https://github.com/zhwangs/UCSB-comp-phys/raw/refs/heads/main/data/sections/dataset_{i}.json", f"{data_dir}/dataset_{i}.json")

# a
def binomial_dist(m, n, p):
	return p**m * (1-p)**(n-m) * binom(n, m)

def prior(p):
	return 1

def stat_mean(values, probs):
	return sum(values * probs) / sum(probs)

def stat_var(values, probs):
	return stat_mean((values - stat_mean(values, probs))**2, probs)

def fisher_info(n, p, dp=1e-6):
	m = arange(n+1)
	prob = binomial_dist(m, n, p)
	score = (log(binomial_dist(m, n, p+dp)) - log(prob)) / dp
	return 1/stat_mean(score**2, prob)

for i in [1, 2, 3]:
	data = load(open(f"{data_dir}/dataset_{i}.json"))
	n = len(data)
	m = sum(data)
	p = linspace(0, 1, 100)
	prob = binomial_dist(m, n, p) * prior(p) / beta(m+1, n-m+1) / binom(n, m)
	plt.plot(p, prob, label=f"Dataset {i}")
	mean_p = stat_mean(p, prob)
	print(f"Dataset {i}: mean = {mean_p:.2f}, var = {stat_var(p, prob):.5f}, Fisher info var = {fisher_info(n, mean_p):.5f}")

plt.xlabel("$p$")
plt.ylabel("Posterior probability density")
plt.legend()
plt.show()

# b
fig, axs = plt.subplots(2, 1)
n = arange(11)
axs[0].scatter(n, log(factorial(n)), label="Factorial")
n = linspace(0, 10, 100)
stirling = n*log(n) - n + log(2*pi*n)/2
gamma = loggamma(n+1)
axs[0].plot(n, stirling, label="Stirling")
axs[0].plot(n, gamma, label="Gamma")
axs[0].set_xlabel("$n$")
axs[0].set_ylabel(r"$\log n!$")
axs[0].legend()

axs[1].plot(n, stirling - gamma)
axs[1].set_xlabel("$n$")
axs[1].set_ylabel("Difference between Stirling and Gamma")

plt.show()

mle_p = {}
for i in [1, 2, 3]:
	data = load(open(f"{data_dir}/dataset_{i}.json"))
	n = len(data)
	m = sum(data)
	p = minimize_scalar(lambda x: -m*log(x) - (n-m)*log(1-x), bounds=(0,1), method="bounded").x
	m_span = arange(n+1)
	mle_p[i] = p
	print(f"MLE estimate of p of dataset {i}: {p:.2f}")

# e
seed(1108)

bootstrapping_count = 100
sample_sizes = [5, 15, 40, 60, 90, 150, 210, 300, 400]

m_values = {}
for i in [1, 2, 3]:
	m_values[i] = {}
	fig, axs = plt.subplots(3, 3)
	data = load(open(f"{data_dir}/dataset_{i}.json"))
	n = len(data)
	for j, sample_size in enumerate(sample_sizes):
		m_values[i][j] = []
		for _ in range(bootstrapping_count):
			m_values[i][j].append(sum([data[k] for k in randint(n, size=sample_size, dtype=int)]))
		axs[j//3,j%3].hist(m_values[i][j], bins=arange(sample_size+1), density=True, align='left')
		axs[j//3,j%3].set_title(f"{sample_size}")
	plt.show()

for i in [1, 2, 3]:
	actual_means = []
	actual_vars = []
	mle_means = []
	mle_vars = []
	for j, sample_size in enumerate(sample_sizes):
		m_samples = m_values[i][j]
		actual_means.append(mean(m_samples))
		actual_vars.append(var(m_samples))
		p = mle_p[i]
		mle_means.append(p * sample_size)
		mle_vars.append(p * (1-p) * sample_size)
	plt.errorbar(sample_sizes, actual_means, yerr=sqrt(actual_vars), label="Bootstrapping", fmt='o', capsize=5)
	plt.errorbar(sample_sizes, mle_means, yerr=sqrt(mle_vars), label="MLE", fmt='o', capsize=5)
	plt.xlabel("Sample size")
	plt.ylabel("Mean and variance")
	plt.legend()
	plt.show()
