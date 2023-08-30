import numpy as np
import os
import warnings
import sys

def jackknife(arr, axis):
	"""
	Jackknife estimate of the error in the mean of arr (along axis)
	
	Arguments:
		arr: numpy array
		axis: int
	"""
	tr = [axis]
	tr.extend([i for i in range(arr.ndim) if i!=axis])
	arr = np.transpose(arr, axes=tr)
	
	n = arr.shape[0]
	if n < 2:
		raise ValueError(f"Cannot estimate error with only one sample. {arr.shape = }, {axis = }")
	
	means = []
	for i in range(n):
		rest = [j for j in range(n) if j!=i]
		means.append(np.average(arr[rest], axis=0))
	means = np.array(means)
	
	m1 = np.average(means, axis=0)
	m2 = np.average(means**2, axis=0)
	var = m2 - m1**2 #biased estimator of variance
	std = np.sqrt(var*n/(n-1.5)) #less biased estimator of standard deviation, assuming errors are Gaussian and uncorrelated between samples. Correction factor copied from Wikipedia (https://en.wikipedia.org/wiki/Standard_deviation#Unbiased_sample_standard_deviation ; accessed 1:55 PM IST, 04-Aug-2023)
	
	return m1, std

class fig_saver():
	"""
	savefig: bool
	savedir: string, path to save the figure
	"""
	def __init__(self, savefig, savedir):
		self.savefig = savefig
		self.savedir = savedir
	
	def __call__(self, fig, name, **kwargs):
		if not self.savefig:
			return
		
		if not os.path.exists(self.savedir):
			#Create directory if it does not exist
			os.makedirs(self.savedir)
		elif not os.path.isdir(self.savedir):
			raise FileExistsError(f"Save location {self.savedir} exists but is not a directory.")
		
		try:
			import git
			repo = git.Repo(path = os.path.dirname(__file__), search_parent_directories = True)
			
			if "metadata" in kwargs.keys():
				raise ValueError("Git was found. Do not specify metadata manually.")
			
			scriptname = sys.argv[0]
			id_str = f"{os.path.basename(scriptname)} at git commit {repo.head.object.hexsha}"
			
			if name[-4:] == ".pdf":
				kwargs['metadata'] = {'Creator': id_str}
			elif name[-4:] == ".png":
				kwargs['metadata'] = {'Software': id_str}
			else:
				raise ValueError(f"Could not infer file type for name {name}")
		except Exception as e:
			warnings.warn(f"Git info will not be saved in the image. {repr(e)}")
		
		loc = os.path.join(self.savedir, name)
		loc_dir = os.path.dirname(loc)
		if not os.path.exists(loc_dir):
			os.makedirs(loc_dir)
		fig.savefig(loc, **kwargs)

def rebin(k_vec, spec, bin_boundaries, axis=0):
	"""
	Given an array spec such that the values along axis correspond to values at corresponding entries of k_vec, rebin those values into bins specified by the list bin_boundaries. Rebinned values are normalized assuming a uniform measure.
	
	Arguments:
		k_vec: array
		spec: array
		bin_boundaries: array
		axis: int
	"""
	
	spec = np.swapaxes(spec, 0, axis)
	
	old_bin_widths = (np.roll(k_vec, -1) - np.roll(k_vec, 1))/2
	old_bin_widths[0] = k_vec[1] - k_vec[0]
	old_bin_widths[-1] = k_vec[-1] - k_vec[-2]
	
	new_bin_widths = bin_boundaries[1:] - bin_boundaries[:-1]
	
	n_bins = len(bin_boundaries)-1
	rebinned = np.zeros([n_bins, *np.shape(spec)[1:]], dtype=spec.dtype)
	ib = 0
	for ik, k in enumerate(k_vec):
		if bin_boundaries[ib+1] <= k:
			ib += 1
		if ib >= n_bins:
			break
		if not bin_boundaries[ib] <= k < bin_boundaries[ib+1]:
			raise RuntimeError(f"Something wrong with specified bins.\n{ik = }\n{ib = }\n{k = }\n{bin_boundaries = }\n{k_vec = }")
		
		rebinned[ib] += spec[ik]*old_bin_widths[ik]/new_bin_widths[ib]
	
	rebinned = np.swapaxes(rebinned, 0, axis)
	return rebinned

def downsample_half(k, arr, axis=0):
	"""
	Given values of arr at wavenumbers k, rebin to double the k-spacing.
	
	Arguments:
		k: numpy array
		arr: numpy array
		axis: int
	"""
	k_old = k
	dk = k_old[1] - k_old[0] #assumes k are equispaced.
	k = k_old[::2]
	bin_bounds = np.linspace(-dk,k_old[-1]+dk,len(k)+1)
	arr = rebin(k_old, arr, bin_bounds, axis=axis)
	return k, arr

if __name__ == "__main__":
	from termcolor import cprint
	
	k = np.linspace(0,9,10)
	new_bin_boundaries = np.linspace(-0.5, 9.5, 11)
	a = np.ones_like(k)
	b = rebin(k, a, new_bin_boundaries)
	assert np.all(np.isclose(a,b))
	
	new_bin_boundaries = np.array([0,2,4,6,8,10])
	b = rebin(k, a, new_bin_boundaries)
	assert np.all(b == 1)
	
	#####
	cprint("All tests passed", attrs=['bold'])
