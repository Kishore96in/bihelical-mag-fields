"""
Read the FITS files provided by HMI
"""

import numpy as np
import scipy.fft
from astropy.io import fits

def get_data_fft(fname):
	"""
	Given the name of a FITS file, return the Fourier transform of the data in it.
	
	Normalization of the Fourier transform is chosen to match that of IDL.
	"""
	with fits.open(fname) as f:
		hdu = f[0]
		data = hdu.data
	data = np.nan_to_num(data)
	return scipy.fft.fft2(data, norm='forward')

def get_B_vec(fname):
	"""
	Read B_vector from FITS files (synoptic vector magnetograms), Fourier-transform it, and return it as an array. A pseudo-Cartesian coordinate system is used, where we map r,phi,mu=cos(theta) to x,y,z (a right-handed coordinate system).
	
	Arguments:
		fname: string of the form "hmi.b_synoptic_small.2267". The input file for Br should be called fname+".Br.fits" (and similar for Bt, Bp). Resulting filename may be anything that is handled by astropy.io.fits.open
	
	"""
	Br = get_data_fft(f"{fname}.Br.fits")
	Bt = get_data_fft(f"{fname}.Bt.fits")
	Bp = get_data_fft(f"{fname}.Bp.fits")
	
	B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
	B_vec = np.swapaxes(B_vec, -1, -2) #The FITS files would've had spatial coordinates latitude,longitude.
	
	return B_vec

def get_data_fft_dbllat(fname):
	"""
	Given the name of a FITS file, double the domain in the latitudinal direction and return the Fourier transform of the data in it.
	
	Normalization of the Fourier transform is chosen to match that of IDL.
	"""
	with fits.open(fname) as f:
		hdu = f[0]
		data = hdu.data
	
	n_lat, n_lon = np.shape(data)
	if not n_lon == n_lat*2:
		raise RuntimeError(f"Unexpected FITS data size: {np.shape(data)}")
	if not n_lat%2 == 0:
		raise RuntimeError("n_lat is odd, so unclear how to split into two for stacking.")
	nlatb2 = int(n_lat/2)
	data = np.concatenate((data[nlatb2:], data, data[:nlatb2]), axis=0)
	if not np.shape(data) == (2*n_lat, n_lon):
		raise RuntimeError(f"Something went wrong while stacking: {np.shape(data) = }; expected ({2*n_lat}, {n_lon})")
	
	data = np.nan_to_num(data)
	return scipy.fft.fft2(data, norm='forward')

def get_B_vec_dbllat(fname):
	"""
	Read B_vector from FITS files (synoptic vector magnetograms), double the domain in the latitudinal directions, Fourier-transform the B vector, and return it as an array. A pseudo-Cartesian coordinate system is used, where we map r,phi,mu=cos(theta) to x,y,z (a right-handed coordinate system).
	
	Arguments:
		fname: string of the form "hmi.b_synoptic_small.2267". The input file for Br should be called fname+".Br.fits" (and similar for Bt, Bp). Resulting filename may be anything that is handled by astropy.io.fits.open
	
	"""
	Br = get_data_fft_dbllat(f"{fname}.Br.fits")
	Bt = get_data_fft_dbllat(f"{fname}.Bt.fits")
	Bp = get_data_fft_dbllat(f"{fname}.Bp.fits")
	
	B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
	B_vec = np.swapaxes(B_vec, -1, -2) #The FITS files would've had spatial coordinates latitude,longitude.
	
	return B_vec
