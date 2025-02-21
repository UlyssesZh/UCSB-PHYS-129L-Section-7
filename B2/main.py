#!/usr/bin/env python

from numpy import sqrt, pi, arccos, arccosh, abs, meshgrid, geomspace, linspace, vectorize, array, exp, log, sin, logical_and
from numpy.random import seed as srand, rand
from scipy.integrate import quad, fixed_quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# b
def true_area(b, c):
	if b == c:
		return 4*pi*b**2
	if b < c:
		return 2*pi*b**2*(1+c**2/(b*sqrt(c**2-b**2))*arccos(b/c))
	if b > c:
		return 2*pi*b**2*(1+c**2/(b*sqrt(b**2-c**2))*arccosh(b/c))
true_area = vectorize(true_area)

def integrand(r, b, c):
	return 4*pi*r*sqrt(1+c**2*r**2/(b**2*(b**2-r**2)))

def area_by_midpoint(b, c):
	r_values = linspace(0, b, 1000)
	midpoints = (r_values[1:]+r_values[:-1])/2
	delta_r = r_values[1:]-r_values[:-1]
	return sum(integrand(midpoints, b, c)*delta_r)
area_by_midpoint = vectorize(area_by_midpoint)

def area_by_gaussian(b, c):
	return fixed_quad(integrand, 0, b, args=(b, c))[0]
area_by_gaussian = vectorize(area_by_gaussian)

b_values, c_values = meshgrid(geomspace(1, 1000, 100), geomspace(1, 1000, 100))
true_areas = true_area(b_values, c_values)
midpoint_areas = area_by_midpoint(b_values, c_values)
gaussian_areas = area_by_gaussian(b_values, c_values)

image = plt.pcolor(b_values, c_values, (midpoint_areas-true_areas)/true_areas)
plt.colorbar(image)
plt.title("relative error of midpoint quadrature")
plt.show()

image = plt.pcolor(b_values, c_values, (gaussian_areas-true_areas)/true_areas)
plt.colorbar(image)
plt.title("relative error of Gaussian quadrature")
plt.show()

# c
def monte_carlo_area(b, c, N, seed=1108):
	srand(seed)
	r_values = b*rand(N)
	return b/N * sum(integrand(r_values, b, c))

b = 0.5
c = 1
N_values = geomspace(10, 100000, 5, dtype=int)
monte_carlo_areas = array([monte_carlo_area(b, c, N) for N in N_values])
true_areas = true_area(b, c)
plt.plot(N_values, (monte_carlo_areas-true_areas)/true_areas, label="uniform")

# d
def monte_carlo_importance_area(b, c, N, pdf, inverse_cdf, seed=1108):
	srand(seed)
	r_values = b*inverse_cdf(rand(N))
	return 1/N * sum(integrand(r_values, b, c) / (pdf(r_values/b)/b))

monte_carlo_areas_exp = array([monte_carlo_importance_area(
	b, c, N,
	lambda r: 3/(1-exp(-3)) * exp(-3*r),
	lambda p: log(1-(1-exp(-3))*p)/-3,
) for N in N_values])
plt.plot(N_values, (monte_carlo_areas_exp-true_areas)/true_areas, label=r"$\exp(-3x)$")

monte_carlo_areas_sin = array([monte_carlo_importance_area(
	b, c, N,
	lambda r: sin(5*r)**2/(1/2-sin(10)/20),
	vectorize(lambda p: fsolve(lambda r: (r/2-sin(10*r)/20)/(1/2-sin(10)/20)-p, 0.5)[0]),
) for N in N_values])
plt.plot(N_values, (monte_carlo_areas_sin-true_areas)/true_areas, label=r"$\sin^2(5x)$")

plt.xscale("log")
plt.legend()
plt.title("relative error of Monte Carlo integration")
plt.show()

# e
def gaussian_sample(N, seed=1108):
	srand(seed)
	u1 = rand(N)
	u2 = rand(N)
	return sqrt(-2*log(u1))*sin(2*pi*u2)

for N in N_values:
	plt.hist(gaussian_sample(N), density=True, bins=max(round(sqrt(N)), 5))
	plt.show()

# f
def monte_carlo_gaussian_area(b, c, N, mu, sigma, seed=1108):
	srand(seed)
	r_values = gaussian_sample(N)*sigma+mu
	r_values = r_values[logical_and(r_values >= 0, r_values <= b)]
	return 1/N * sum(integrand(r_values, b, c) / (exp(-(r_values-mu)**2/2/sigma**2)/sqrt(2*pi)/sigma))

monte_carlo_areas_gaussian = array([monte_carlo_gaussian_area(b, c, N, 0, 1) for N in N_values])
plt.plot(N_values, (monte_carlo_areas-true_areas)/true_areas, label="uniform")
plt.plot(N_values, (monte_carlo_areas_exp-true_areas)/true_areas, label=r"$\exp(-3x)$")
plt.plot(N_values, (monte_carlo_areas_sin-true_areas)/true_areas, label=r"$\sin^2(5x)$")
plt.plot(N_values, (monte_carlo_areas_gaussian-true_areas)/true_areas, label="Gaussian")
plt.xscale("log")
plt.legend()
plt.title("relative error of Monte Carlo integration")
plt.show()

N = 10000
mu_values, sigma_values = meshgrid(linspace(0, 1, 10), linspace(0.1, 2, 10))
monte_carlo_areas_gaussian = vectorize(monte_carlo_gaussian_area)(b, c, N, mu_values, sigma_values)
image = plt.pcolor(mu_values, sigma_values, (monte_carlo_areas_gaussian-true_areas)/true_areas)
plt.colorbar(image)
plt.xlabel(r"$\mu$")
plt.ylabel(r"$\sigma$")
plt.title("relative error of Monte Carlo integration with Gaussian sampling")
plt.show()
# The error does not seem to vary very much for different mu and sigma,
# except for mu = 0 and mu = 1 and small sigma,
# which is because too many samples are wasted in those cases.
