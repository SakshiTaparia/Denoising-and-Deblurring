import numpy
from numpy import linalg

import scipy
from scipy import signal
from scipy.fftpack import ifftshift

import matplotlib.pyplot as plt

import cv2


def gaussian_noise (m, n):

	# Normal distribution with mean 0 and variance 1

	array = numpy.random.randn(m,n)

	array = numpy.multiply(array, 10)

	array = array.astype(int)

	return array


gaussian = gaussian_noise(100, 100)

S = numpy.power(gaussian, 2)

sum_S = numpy.sum(S)

print(sum_S)

N = numpy.fft.fft2(gaussian)

S_n = numpy.power(numpy.absolute(N), 2)

sum_S_n = numpy.sum(S_n)

print(sum_S_n)