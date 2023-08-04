import numpy as np

def jackknife(arr, axis):
	"""
	Jackknife estimate of the error in the mean of arr (along axis)
	
	Arguments:
		arr: numpy array
		axis: int
	"""
	arr = np.swapaxes(arr, axis, 0)
	
	n = arr.shape[0]
	if n < 2:
		raise ValueError(f"Cannot estimate error with only one sample. {arr.shape = }, {axis = }")
	
	means = []
	for i in range(n):
		rest = [i for j in range(n) if j!=i]
		means.append(np.average(arr[rest], axis=0))
	means = np.array(means)
	
	m1 = np.average(means, axis=0)
	m2 = np.average(means**2, axis=0)
	var = m2 - m1**2 #biased estimator of variance
	std = np.sqrt(var*n/(n-1.5)) #less biased estimator of standard deviation, assuming errors are Gaussian and uncorrelated between samples. Correction factor copied from Wikipedia (https://en.wikipedia.org/wiki/Standard_deviation#Unbiased_sample_standard_deviation ; accessed 1:55 PM IST, 04-Aug-2023)
	
	return m1, std
