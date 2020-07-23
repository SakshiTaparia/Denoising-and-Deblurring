import numpy
from numpy import linalg

import scipy
from scipy import signal
from scipy.fftpack import ifftshift

import cv2

def generatePSF(image):

	center_coordinates = ((int)(image.shape[0]/2), (int)(image.shape[1]/2))

	PSF = numpy.zeros((image.shape[0], image.shape[1]))

	PSF = cv2.circle(PSF, center_coordinates, 4, (255), -1)

	PSF = numpy.divide(PSF, numpy.sum(PSF))

	return PSF


def blur_using_PSF(image, PSF):

	PSF_fourier = numpy.fft.fft2(PSF)
	
	shift_image = ifftshift(image)

	image_fourier = numpy.fft.fft2(shift_image)

	
	answer = numpy.fft.ifft2( numpy.multiply(image_fourier, PSF_fourier) )

	answer = numpy.clip(answer, 0 , 255)

	answer = answer.astype(numpy.uint8)

	return answer


def gaussian_noise (m, n):

	# Normal distribution with mean 0 and variance 1

	array = numpy.random.randn(m,n)

	array = numpy.multiply(array, 25)

	array = numpy.rint(array)

	array = numpy.clip(array, 0, 255)

	array = array.astype(numpy.uint8)

	return array


def wiener_filter(H, G, S_n, S_f):

	H_spectrum = numpy.power(numpy.absolute(H), 2)

	array = numpy.divide(S_n, S_f) + H_spectrum

	array = numpy.divide(H_spectrum, array)

	array = numpy.divide(array, H)

	array = numpy.multiply(array, G)

	array = numpy.fft.ifft2(array)

	array = numpy.rint(array)

	array = numpy.clip(array, 0, 255)

	array = array.astype(numpy.uint8)

	return array

######################################################################################################

img = cv2.imread('barbara.png', 0)
img_arr = numpy.array(img)

PSF = generatePSF(img_arr)

blurred_image = blur_using_PSF(img_arr, PSF)

cv2.imshow('blurred_image', blurred_image)
cv2.waitKey(0)


gaussian = gaussian_noise(img_arr.shape[0], img_arr.shape[1])

cv2.imshow('gaussian_noise', gaussian)
cv2.waitKey(0)


blur_with_gaussian = gaussian + blurred_image

blur_with_gaussian = numpy.clip(blur_with_gaussian, 0, 255)
blur_with_gaussian = blur_with_gaussian.astype(numpy.uint8)

cv2.imshow('blur_with_gaussian', blur_with_gaussian)
cv2.waitKey(0)


# H(u,v) is the degradation function = fourier transform of PSF

# G(u,v) is the degraded image = fourier transform of blur_with_gaussian

# S_n(u,v) is the power spectrum of gaussian noise

# S_f(u,v) is the estimated power spectrum of original undegraded image


blur_with_gaussian = ifftshift(blur_with_gaussian)


H = numpy.fft.fft2(PSF)

G = numpy.fft.fft2(blur_with_gaussian)

N = numpy.fft.fft2(gaussian)

S_n = numpy.power(numpy.absolute(N), 2)


shift_image = ifftshift(img_arr)

F = numpy.fft.fft2(shift_image)

F = numpy.power(numpy.absolute(F),2)


S_f = numpy.full( (img_arr.shape[0], img_arr.shape[1]), (numpy.amax(F) + numpy.amin(F))/1000000 )

constant_noise_spectrum = wiener_filter(H, G, S_n, S_f)

cv2.imshow('constant_noise_spectrum', constant_noise_spectrum)
cv2.waitKey(0)


S_f = F

original_ground_truth = wiener_filter(H, G, S_n, S_f)

cv2.imshow('original_ground_truth', original_ground_truth)
cv2.waitKey(0)


S_f = numpy.zeros(( img_arr.shape[0], img_arr.shape[1] ))

for i in range (0, img_arr.shape[0]):

	for j in range (0, img_arr.shape[1]):

		S_f[i][j] = numpy.power(i+1 ,2) + numpy.power(j+1 ,2)

print(S_f)

alpha = 3.25

S_f = numpy.power(S_f, alpha/2)

alpha_power_spectrum = wiener_filter(H, G, S_n, S_f)

cv2.imshow('alpha_power_spectrum4', alpha_power_spectrum)
cv2.waitKey(0)
