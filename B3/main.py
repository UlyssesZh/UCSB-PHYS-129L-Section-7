#!/usr/bin/env python

from numpy import exp, log, cos, sqrt
from numpy.random import rand, seed as srand
import matplotlib.pyplot as plt

# a
def pdf(t, a, b):
	return exp(-b*t)*cos(a*t)**2

# tf is the upper bound of any possible samples.
# One should expect the the largest of all N samples to be around inverse_CDF(1-1/N),
# so tf should be larger than that.
# In this case, the CDF is
# CDF(t) = 1 - exp(-b t) (4 a^2 + b^2 + b^2 cos(2 a t) - 2 a b sin(2 a t)) / (4 a^2 + 2 b^2).
# Its inverse cannot be found analytically, but it has bound
# CDF(t) <= 1 - exp(-b t) (4 a^2 + b^2 - b sqrt(4 a^2 + b^2)) / (4 a^2 + 2 b^2).
# Now this bound can be inverted analytically to get
# inverse_CDF(F) <= log((4 a^2 + b^2 - b sqrt(4 a^2 + b^2)) / (4 a^2 + 2 b^2) / (1 - F)) / b.
# The RHS can then be a good choice of tf, replaceing F with 1-1/N.
def decay_sample(N, a, b, seed=1108):
	srand(seed)
	tf = log((4*a**2+b**2-b*sqrt(4*a**2+b**2))/(4*a**2+2*b**2)*N)/b
	amp = pdf(0, a, b)
	result = []
	reject_count = 0
	while len(result) < N:
		sample = tf * rand()
		if rand() < pdf(sample, a, b) / amp:
			result.append(sample)
		else:
			reject_count += 1
	return result, reject_count

a = 4
b = 4
for N in [100, 1000, 10000]:
	samples, reject_count = decay_sample(N, a, b)
	print(f"For N = {N}, {reject_count} samples are rejected, accpt/reject = {N/reject_count:.2f}")
	plt.hist(samples, bins=round(sqrt(N)), density=True)
	plt.show()

# b
def decay_sample_exp(N, a, b, seed=1108):
	srand(seed)
	result = []
	reject_count = 0
	while len(result) < N:
		sample = -log(1-rand())/2
		if rand() < pdf(sample, a, b) / exp(-2*sample): # only works if a >= 2
			result.append(sample)
		else:
			reject_count += 1
	return result, reject_count

for N in [100, 1000, 10000]:
	samples, reject_count = decay_sample_exp(N, a, b)
	print(f"For N = {N}, {reject_count} samples are rejected, accpt/reject = {N/reject_count:.2f}")
	plt.hist(samples, bins=round(sqrt(N)), density=True)
	plt.show()
