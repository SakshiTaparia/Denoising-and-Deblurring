import numpy
from numpy import linalg

import scipy
from scipy import signal

import cv2


# To create random numbers following a distribution with CDF F(x)
# select random numbers between 0 to 1 and select corresponding x 

def add_gaussian_noise (m, n):

	# CDF of normal distribution is not explicitly defined

	array = numpy.random.randn(m,n)

	# random floats sampled from normal distribution of mean 0 and variance 1

	array = numpy.multiply(array, 50)

	# making sigma as 40

	# 5(sigma) = 250 = half of the range 0 to 255 (99.9% pixels covered)

	# Mean of array is zero 

	array = array + 0

	array = numpy.rint(array)

	# array = array.astype(numpy.uint8)
	
	return array


def add_salt_pepper_noise (m, n):

	array = numpy.random.rand(m,n)

	for i in range (0,m):

		for j in range (0,n):

			if (array[i][j] < 0.1):

				array[i][j] = -255

			elif (array[i][j] > 0.9):

				array[i][j] = 255

			else:

				array[i][j] = 0

	# make the elements either 0 or 255

	# array = array.astype(numpy.uint8)
	
	return array


# size had to be an odd positive integer

def mean_filter(array, size):

	value = 1/(size*size)

	filter_matrix = numpy.full((size, size), value) 

	array = scipy.signal.convolve2d(array, filter_matrix, boundary='symm', mode='same')

	return array


def median_filter (array, size):

	array = cv2.medianBlur(array, size)

	return array

######################################################################################################

# Imread reads in BGR format instead of RGB

# -1 reads the image in unmodified format
# 0 reads the image in greyscale format
# 1 reads the image in colour format

img = cv2.imread('barbara.png', 0)
img_arr = numpy.array(img)

cv2.imshow('input',img_arr)
cv2.waitKey(0)

gaussian = add_gaussian_noise(img_arr.shape[0], img_arr.shape[1])

gaussian_noise = gaussian + img_arr
gaussian_noise = numpy.clip(gaussian_noise, 0, 255)
gaussian_noise = gaussian_noise.astype(numpy.uint8)

cv2.imshow('gaussian_noise',gaussian_noise)
cv2.waitKey(0)

size = 1

while (size < 20):

	mean = mean_filter(gaussian_noise, size)
	mean = numpy.clip(mean, 0, 255)
	mean = mean.astype(numpy.uint8)

	median = median_filter(gaussian_noise, size)

	cv2.imshow('mean', mean)
	cv2.waitKey(0)

	cv2.imshow('median', median)
	cv2.waitKey(0)

	psnr_mean = numpy.power( (img_arr - mean), 2)
	psnr_mean = numpy.divide(psnr_mean, img_arr.shape[0]*img_arr.shape[1])
	psnr_mean = numpy.sum(psnr_mean)

	psnr_mean = 10 * (numpy.log10( (255*255)/ psnr_mean))

	psnr_median = numpy.power( (img_arr - median), 2)
	psnr_median = numpy.divide(psnr_median, img_arr.shape[0]*img_arr.shape[1])
	psnr_median = numpy.sum(psnr_median)

	psnr_median = 10 * (numpy.log10( (255*255)/ psnr_median))

	print(size)
	print(psnr_mean)
	print(psnr_median)

	size = size + 2


sp = add_salt_pepper_noise(img_arr.shape[0], img_arr.shape[1])

sp_noise = sp + img_arr
sp_noise = numpy.clip(sp_noise, 0, 255)
sp_noise = sp_noise.astype(numpy.uint8)

cv2.imshow('sp_noise', sp_noise)
cv2.waitKey(0)

size = 1

while (size < 20):

	mean = mean_filter(sp_noise, size)
	mean = numpy.clip(mean, 0, 255)
	mean = mean.astype(numpy.uint8)

	median = median_filter(sp_noise, size)

	cv2.imshow('mean', mean)
	cv2.waitKey(0)

	cv2.imshow('median', median)
	cv2.waitKey(0)

	psnr_mean = numpy.power( (img_arr - mean), 2)
	psnr_mean = numpy.divide(psnr_mean, img_arr.shape[0]*img_arr.shape[1])
	psnr_mean = numpy.sum(psnr_mean)

	psnr_mean = 10 * (numpy.log10( (255*255)/ psnr_mean))

	psnr_median = numpy.power( (img_arr - median), 2)
	psnr_median = numpy.divide(psnr_median, img_arr.shape[0]*img_arr.shape[1])
	psnr_median = numpy.sum(psnr_median)

	psnr_median = 10 * (numpy.log10( (255*255)/ psnr_median))

	print(size)
	print(psnr_mean)
	print(psnr_median)

	size = size + 2
