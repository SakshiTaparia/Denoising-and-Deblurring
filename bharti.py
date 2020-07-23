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

	array = numpy.multiply(array, 7.5)

	array = numpy.rint(array)

	array = numpy.clip(array, 0, 255)

	array = array.astype(numpy.uint8)

	return array


def generatePSF(image):

	# A = cv2.getGaussianKernel(791, 1)
	
	# B = numpy.reshape(A, (1,791))

	# PSF = numpy.dot(A,B)

	# PSF = PSF[:, 20:772]


	center_coordinates = ((int)(image.shape[0]/2), (int)(image.shape[1]/2))

	PSF = numpy.zeros((image.shape[0], image.shape[1]))

	PSF = cv2.circle(PSF, center_coordinates, 3, (255), -1)
	PSF = numpy.divide(PSF, numpy.sum(PSF))

	return PSF


def RGB_to_luminance (array):

	L = numpy.zeros((array.shape[0], array.shape[1]))

	for i in range (0, array.shape[0]):

		for j in range (0, array.shape[1]):

			# array is in BGR format instead of RGB

			L[i][j] = 0.114*array[i][j][0] + 0.587*array[i][j][1] + 0.299*array[i][j][2]

			if (L[i][j] == 0):

				L[i][j] = 0.01

	return (L)


def wiener_filter(H, G, S_n, S_f):

	H_spectrum = numpy.power(numpy.absolute(H), 2)

	array = numpy.divide(S_n, S_f) + H_spectrum

	array = numpy.divide(H_spectrum, array)

	array = numpy.divide(array, H)

	array = numpy.multiply(array, G)

	array = numpy.fft.ifft2(array)

	print(array)

	array = numpy.clip(array, 0, 255)

	array = array.astype(numpy.uint8)

	return array


def Luminance_to_RGB (original_image, original_luminance, final_luminance):

	new_image = numpy.zeros((original_image.shape[0], original_image.shape[1], original_image.shape[2]))

	for i in range (0, original_image.shape[0]):

		for j in range (0, original_image.shape[1]):

			# array is in BGR format instead of RGB
			
			new_image[i][j][0] = (original_image[i][j][0]*final_luminance[i][j])/original_luminance[i][j]
			new_image[i][j][1] = (original_image[i][j][1]*final_luminance[i][j])/original_luminance[i][j]
			new_image[i][j][2] = (original_image[i][j][2]*final_luminance[i][j])/original_luminance[i][j]

	
	new_image = numpy.clip(new_image, 0, 255)

	new_image = new_image.astype(numpy.uint8)

	return (new_image)


# Imread reads in BGR format instead of RGB

# -1 reads the image in unmodified format
# 0 reads the image in greyscale format
# 1 reads the image in colour format

img = cv2.imread('bharti.png', 0)
img_arr = numpy.array(img)

img_arr = img_arr[108:756, :]

print(img_arr.shape)

cv2.imshow('image', img_arr)
cv2.waitKey(0)

colour_image = cv2.imread('bharti.png', 1)
colour_image_array = numpy.array(colour_image)

colour_image_array = colour_image_array[108:756, :]

colour_image_luminance = RGB_to_luminance(colour_image_array)


constant = img_arr[100:150, 0:100]

cv2.imshow('constant', constant)
cv2.waitKey(0)

constant = constant.flatten()

plt.hist(constant, bins = 10, density = True)
plt.show()

# Noise is gaussian with mean = 0 and standard devation = 10/3

shifted_image = ifftshift(img_arr)

G = numpy.fft.fft2(shifted_image)


gaussian = gaussian_noise(img_arr.shape[0], img_arr.shape[1])

N = numpy.fft.fft2(gaussian)

S_n = numpy.power(numpy.absolute(N), 2)


PSF = generatePSF(img_arr)

H = numpy.fft.fft2(PSF)

print(H)


F = numpy.power(numpy.absolute(G), 2)

print(F)

S_f = F

recovered_image = wiener_filter(H, G, S_n, S_f)

print(recovered_image)

cv2.imshow('recovered_image', recovered_image)
cv2.waitKey(0)

median = cv2.medianBlur(recovered_image, 5)

cv2.imshow('median', median)
cv2.waitKey(0)


kernel_sharpening = numpy.array([[0,-1,0], [-1, 5,-1], [0,-1,0]])

sharpened = cv2.filter2D(median, -1, kernel_sharpening)

cv2.imshow('sharpened', sharpened)
cv2.waitKey(0)

sharpened = sharpened.astype(float)

colour_output = Luminance_to_RGB(colour_image_array, colour_image_luminance, sharpened)

cv2.imshow('colour_output', colour_output)
cv2.waitKey(0)

