import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from scipy import signal
from statistics import median

def sin_cos(amplitude, frequency, time, phase, choice):
	'''
	np.cos and np.sin require input as radians (not degrees)
	'''
	y = 0 # preallocate memory
	if choice == 'cos':
		y = amplitude*np.cos(2*np.pi*frequency*time + phase)
	elif choice == 'sin':
		y = amplitude*np.sin(2*np.pi*frequency*time + phase)
	return y

def sin_cos_plot(amplitude, frequency, time, phase, choice, title):
	fig = plt.figure(figsize=(10, 5))
	y = sin_cos(amplitude, frequency, time, phase, choice)
	plt.plot(time, y, color='cyan', linestyle='--', linewidth=2)
	plt.xticks(fontsize=16); plt.yticks(fontsize=16)
	plt.xlabel(r'radian ($\pi$)', fontsize=18)
	plt.ylabel('amplitude', fontsize=18)
	plt.title(title, fontsize=18, fontweight='bold')
	# plt.savefig('image_out/' + title + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def ricker(frequency, length=0.128, dt=0.004): # bug here exaggerate frequency
	# http://subsurfwiki.org/wiki/Ricker_wavelet   	
	time = np.arange(-length/2, (length-dt)/2, dt)
	wiggle = (1.0 - 2.0*(np.pi**2)*(frequency**2)*(time**2)) * np.exp(-(np.pi**2)*(frequency**2)*(time**2))
	return wiggle

def clip(model, perc):
	(ROWs, COLs) = model.shape
	reshape2D_1D = model.reshape(ROWs*COLs)
	reshape2D_1D = np.sort(reshape2D_1D)
	if perc != 100:
		min_num = reshape2D_1D[ round(ROWs*COLs*(1-perc/100)) ]
		max_num = reshape2D_1D[ round((ROWs*COLs*perc)/100) ]
	elif perc == 100:
		min_num = min(model.flatten())
		max_num = max(model.flatten())
	if min_num > max_num:
		dummy = max_num
		max_num = min_num
		min_num = dummy
	return max_num, min_num 

def reflectivity(vp, frequency):
	wiggle = ricker(frequency)
	(ROWs, COLs) = vp.shape
	reflectivity = np.zeros_like(vp, dtype='float')
	conv = np.zeros_like(vp, dtype='float')
	rho = 2700
	for col in range (0, COLs):
		for row in range (0, ROWs-1):
			reflectivity[row, col] = (vp[row+1, col]*rho - vp[row, col]*rho) / (vp[row+1, col]*rho + vp[row, col]*rho)
		# flip polarity
		conv[:, col] = signal.convolve((reflectivity[:, col]*-1), wiggle, mode='same') / sum(wiggle)
	laplacian = cv2.Laplacian(conv, cv2.CV_64F)
	return laplacian

def scaling_velocity(y, lowest_value, highest_value):
	y = (y - y.min()) / (y.max() - y.min())
	return y * (highest_value - lowest_value) + lowest_value

def convert2velocity(img, nx, nz, filter_sigma):
	img    = rgb2gray(img)
	model  = scaling_velocity(img, 2.0, 5.) # 2 - 5 velocity (km/s)
	model  = resize(model, (nx, nz), anti_aliasing=True)
	smooth = gaussian_filter(model, sigma=filter_sigma)
	return model, smooth

def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X

def plot_spectra(X, sampling_rate): 
	# calculate the frequency
	N = len(X)
	n = np.arange(N)
	T = N/sampling_rate
	freq = n/T 
	half = int(len(freq)/2)
	freq = freq[:half]
	X = X[:half]
	# print(X.shape)

	fig = plt.figure(figsize=(12, 8))  
	plt.stem(freq, abs(X), 'b', markerfmt=" ", basefmt="-b")
	plt.xlim(0, 30)
	plt.xlabel('frequency (Hz)')
	plt.ylabel('amplitude')
	# plt.savefig('image_out/' + 'DFT_2' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def compute_frequency(X, sampling_rate):
	# calculate the frequency
	N = len(X)
	n = np.arange(N)
	T = N/sampling_rate
	freq = n/T 
	half = int(len(freq)/2)
	freq = freq[:half]
	X = X[:half]
	freq = freq[2:]; X = X[2:] # remove the first value
	p_45 = np.percentile(X, 45); p_55 = np.percentile(X, 55)
	count = 0.; avg_freq = 0.
	for index, i in enumerate(X):
		if i >= p_45 and i <= p_55:
			avg_freq = avg_freq + freq[index]
			count += 1
	avg_freq = avg_freq/count
	return avg_freq