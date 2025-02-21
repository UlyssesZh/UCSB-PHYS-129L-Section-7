#!/usr/bin/env python

from itertools import islice, count

from numpy import array, arange
import matplotlib.pyplot as plt

def sobol_sequence(m, primitive):
	s = len(primitive)
	d = len(m)
	for i in count():
		while i >> d:
			m.append(0)
			for j in range(s):
				m[d] ^= primitive[j]*m[d-j] << j%(s-1)
			d += 1
		result = 0
		for j in range(d):
			if i >> j & 1:
				result ^= m[j] << d-j
		yield result / (1<<d+1)

n = 50
sobol_sequence = islice(sobol_sequence([1, 3, 5], [1, 1, 1, 1]), n)
plt.scatter(arange(n), tuple(sobol_sequence))
plt.show()
