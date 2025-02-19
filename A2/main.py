#!/usr/bin/env python

from os.path import isdir
from os import mkdir
from json import load

from numpy import histogram, exp, pi, sqrt, linspace, array, sum, log, zeros
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.special import erf

data_dir = "data"
if not isdir(data_dir):
	print("Downloading data...")
	from urllib.request import urlretrieve
	mkdir(data_dir)
	for decay_type in ["Vacuum", "Cavity"]:
		urlretrieve(f"https://github.com/zhwangs/UCSB-comp-phys/raw/refs/heads/main/data/sections/{decay_type}_decay_dataset.json", f"{data_dir}/{decay_type}_decay_dataset.json")

# a
data = array(load(open(f"{data_dir}/Cavity_decay_dataset.json")))
data -= data.min()
bins_count = 200
hist, bin_edges = histogram(data, bins=bins_count, density=True)
bin_centers = (bin_edges[1:]+bin_edges[:-1])/2

def fit_distribution(x, lmd, sgm, mu, normal_height):
	exponential = exp(-x/lmd)/lmd
	normal = exp(-(x-mu)**2/2/sgm**2)/sqrt(2*pi)/sgm
	return (1-normal_height)*exponential + normal_height*normal

popt, pcov = curve_fit(fit_distribution, bin_centers, hist, p0=[1.25, 1, 6, 0.2])

fit_lmd, fit_sgm, fit_mu, fit_normal_height = popt
print("Fit parameters:")
print(f"lambda = {fit_lmd:.2f}, sigma = {fit_sgm:.2f}, mu = {fit_mu:.2f},")
print(f"height of exponential distribution and normal distribution: {1-fit_normal_height:.2f}, {fit_normal_height:.2f}")
print(f"Covariance matrix: {pcov}")

'''
plt.hist(data, bins=bins_count, density=True, label="Data")
x_span = linspace(data.min(), data.max(), 50)
plt.plot(x_span, fit_distribution(x_span, *popt), label="Fit")
plt.show()
'''

def mle_estimator(args, distribution_function):
	return -sum(log(distribution_function(data, *args)))

mle_result = minimize(mle_estimator, [1.25, 1, 6, 0.2], args=(fit_distribution,), bounds=[(0, None), (0, None), (None, None), (0, 1)])
mle_lmd, mle_sgm, mle_mu, mle_normal_height = mle_result.x
print("MLE parameters:")
print(f"lambda = {mle_lmd:.2f}, sigma = {mle_sgm:.2f}, mu = {mle_mu:.2f},")
print(f"height of exponential distribution and normal distribution: {1-mle_normal_height:.2f}, {mle_normal_height:.2f}")

def fisher_info(distribution_function, args, delta=1e-6, N=10000, x_range=(0,20)):
	x_values = linspace(*x_range, N)
	distribution = distribution_function(x_values, *args)
	score = zeros((len(args), N))
	for i in range(len(args)):
		args_plus = args.copy()
		args_plus[i] += delta
		args_minus = args.copy()
		args_minus[i] -= delta
		score[i] = (log(distribution_function(x_values, *args_plus)) - log(distribution_function(x_values, *args_minus))) / (2*delta)
		score[i] *= distribution
	result = zeros((len(args), len(args)))
	for i in range(len(args)):
		for j in range(len(args)):
			result[i, j] = sum(score[i]*score[j])
	return result

print(f"Inverse Fisher information matrix: {inv(fisher_info(fit_distribution, popt))}")

# b
print("H0: there is no second exponential distribution")

# lmd1 is the parameter for an additional exponential distribution.
# height1 * (1-normal_height) is its height.
# (1-height1) * (1-normal_height) is the height of the original exponential distribution.
def fit_distribution_alternative(x, lmd, sgm, mu, normal_height, lmd1, height1):
	exponential = exp(-x/lmd)/lmd
	normal = exp(-(x-mu)**2/2/sgm**2)/sqrt(2*pi)/sgm
	exponential1 = exp(-x/lmd1)/lmd1
	return (1-normal_height)*(1-height1)*exponential + normal_height*normal + (1-normal_height)*height1*exponential1

popt, pcov = curve_fit(fit_distribution_alternative, bin_centers, hist, p0=[1.25, 1, 6, 0.2, 1, 0])
z_score = popt[4] / sqrt(pcov[4, 4])
p_value = erf(z_score/sqrt(2))
print("p-value:", p_value)
if p_value < 0.05:
	print("Reject H0")
elif p_value > 0.95:
	print("Accept H0")
else:
	print("Cannot make a decision")

