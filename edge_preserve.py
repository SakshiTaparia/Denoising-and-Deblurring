import numpy
from numpy import linalg

import scipy
from scipy import signal

import cv2


# To create random numbers following a distribution with CDF F(x)
# select random numbers between 0 to 1 and select corresponding x 

def add_gaussian_noise (m, n):

	array = numpy.random.randn(m,n)

	array = numpy.multiply(array, 50)

	array = array + 0

	array = numpy.rint(array)

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
	
	return array

######################################################################################################

def total_variation_denoising (image, standard_deviation):

	A = cv2.getGaussianKernel(7, standard_deviation)
	
	B = numpy.reshape(A, (1,7))

	# 7*7 gaussian kernel

	kernel = numpy.dot(A,B)
	

	padded_image = numpy.zeros((image.shape[0], image.shape[1]))
	
	for i in range (0, image.shape[0]):

		for j in range (0, image.shape[1]):

			padded_image[i][j] = image[i][j]


	zeros_H = numpy.zeros((3, image.shape[0]))

	padded_image = numpy.vstack((zeros_H, padded_image))
	padded_image = numpy.vstack((padded_image, zeros_H))

	zeros_V = numpy.zeros((image.shape[1] + 6, 3))

	padded_image = numpy.hstack((zeros_V, padded_image))
	padded_image = numpy.hstack((padded_image, zeros_V))
	

	Neighbourhood = numpy.zeros((image.shape[0], image.shape[1], 7, 7)) 

	for i in range (3, image.shape[0]+3):

		for j in range (3, image.shape[1]+3):

			matrix = padded_image[ i-3: i+4, j-3: j+4]

			Neighbourhood[i-3][j-3] = scipy.signal.convolve2d(matrix, kernel, boundary='symm', mode='same')

			print(i-3)
			print(j-3)

			print(Neighbourhood[i-3][j-3])


	array = numpy.zeros((image.shape[0], image.shape[1]))

	for i in range (0, image.shape[0]):

		for j in range (0, image.shape[1]):

			weights = numpy.zeros((image.shape[0], image.shape[1])) 

			for k in range (i-10, i+11):

				if (k < 0 or k > image.shape[0] - 1):

					continue

				else:

					for l in range (j-10, j+11):

						if (l < 0 or l > image.shape[1] - 1):

							continue

						else:

							weights[k][l] = numpy.exp((-1)*linalg.norm(Neighbourhood[i][j] - Neighbourhood[k][l])/(10*standard_deviation))


			normalization = numpy.sum(weights)

			weights = numpy.divide(weights, normalization)

			answer = numpy.multiply(image, weights)

			array[i][j] = numpy.sum(answer)

			print(i)
			print(j)

			print(array[i][j])

	
	array = numpy.clip(array, 0 ,255)
	array = array.astype(numpy.uint8)

	return array

######################################################################################################

img = cv2.imread('barbara.png', 0)
img_arr = numpy.array(img)

gaussian = add_gaussian_noise(img_arr.shape[0], img_arr.shape[1])

gaussian_noise = gaussian + img_arr
gaussian_noise = numpy.clip(gaussian_noise, 0, 255)
gaussian_noise = gaussian_noise.astype(numpy.uint8)

cv2.imshow('gaussian_noise',gaussian_noise)
cv2.waitKey(0)

gaussian_output = total_variation_denoising(gaussian_noise, 50)

cv2.imshow('gaussian_output', gaussian_output)
cv2.waitKey(0)

######################################################################################################

sp = add_salt_pepper_noise(img_arr.shape[0], img_arr.shape[1])

sp_noise = sp + img_arr
sp_noise = numpy.clip(sp_noise, 0, 255)
sp_noise = sp_noise.astype(numpy.uint8)

cv2.imshow('sp_noise', sp_noise)
cv2.waitKey(0)

sp_output = total_variation_denoising(sp_noise, 50)

cv2.imshow('sp_output', sp_output)
cv2.waitKey(0)