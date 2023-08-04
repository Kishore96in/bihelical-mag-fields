"""
Set the magnetic field to zero in regions where its magnitude is below some threshold.
"""

import numpy as np
import scipy.fft
from astropy.io import fits

def get_data(fname):
	with fits.open(fname) as f:
		hdu = f[0]
		data = hdu.data
	data = np.nan_to_num(data)
	return data

def get_Bvec(fname, threshold=0):
	"""
	Read B_vector from FITS files, set all the weak-field regions to zero, Fourier-transform it, and return it as an array.
	
	Arguments:
		fname: string of the form "hmi.b_synoptic_small.2267". The input file for Br should be called fname+".Br.fits" (and similar for Bt, Bp). Resulting filename may be anything that is handled by astropy.io.fits.open
		threshold: if the magnitude of the magnetic field is below this value, set it to zero.
	"""
	#Below, the magnetic field has not been Fourier-transformed yet
	Br = get_data(f"{fname}.Br.fits")
	Bt = get_data(f"{fname}.Bt.fits")
	Bp = get_data(f"{fname}.Bp.fits")
	
	Bmag = np.sqrt(Br**2 + Bt**2 + Bp**2)
	
	Br = np.where(Bmag>threshold, Br, 0)
	Bt = np.where(Bmag>threshold, Bt, 0)
	Bp = np.where(Bmag>threshold, Bp, 0)
	
	Br = scipy.fft.fft2(Br, norm='forward')
	Bt = scipy.fft.fft2(Bt, norm='forward')
	Bp = scipy.fft.fft2(Bp, norm='forward')
	
	B_vec = np.stack([Br, Bp, -Bt])
	B_vec = np.swapaxes(B_vec, -1, -2)
	
	return B_vec
