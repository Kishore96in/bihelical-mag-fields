"""
Read the FITS files provided by HMI
"""

import numpy as np
import scipy.fft
from astropy.io import fits

def get_data_fft(fname, dbllat=False):
	"""
	Given the name of a FITS file, return the Fourier transform of the data in it.
	
	Normalization of the Fourier transform is chosen to match that of IDL.
	
	Arguments:
		fname: string, name of the file
		dbllat: bool, whether to double the domain in the latitudinal direction
	"""
	with fits.open(fname) as f:
		hdu = f[0]
		data = hdu.data
	data = np.nan_to_num(data)
	
	if dbllat:
		n_lat, n_lon = np.shape(data)
		if not n_lon == n_lat*2:
			raise RuntimeError(f"Unexpected FITS data size: {np.shape(data)}")
		if not n_lat%2 == 0:
			raise RuntimeError("n_lat is odd, so unclear how to split into two for stacking.")
		nlatb2 = int(n_lat/2)
		data = np.concatenate((data[nlatb2:], data, data[:nlatb2]), axis=0)
		if not np.shape(data) == (2*n_lat, n_lon):
			raise RuntimeError(f"Something went wrong while stacking: {np.shape(data) = }; expected ({2*n_lat}, {n_lon})")
	
	return scipy.fft.fft2(data, norm='forward')

def get_B_vec(fname, dbllat=False):
	"""
	Read B_vector from FITS files (synoptic vector magnetograms), Fourier-transform it, and return it as an array. A pseudo-Cartesian coordinate system is used, where we map r,phi,mu=cos(theta) to x,y,z (a right-handed coordinate system).
	
	Arguments:
		fname: string of the form "hmi.b_synoptic_small.2267". The input file for Br should be called fname+".Br.fits" (and similar for Bt, Bp). Resulting filename may be anything that is handled by astropy.io.fits.open
		dbllat: bool. whether to double the domain in the latitudinal direction
	"""
	Br = get_data_fft(f"{fname}.Br.fits", dbllat=dbllat)
	Bt = get_data_fft(f"{fname}.Bt.fits", dbllat=dbllat)
	Bp = get_data_fft(f"{fname}.Bp.fits", dbllat=dbllat)
	
	B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
	B_vec = np.swapaxes(B_vec, -1, -2) #The FITS files would've had spatial coordinates latitude,longitude.
	
	return B_vec

class FITSreader():
	"""
	Intended to be used like
	```
	r = FITSreader(mask_threshold=100, dbllat=True)
	B_vec = r(fname)
	```
	"""
	def __init__(**kwargs):
		for key, value in kwargs.items():
			setattr(self, key, value)
	
	def read(self, fname):
		"""
		Read FITS files, and return an array [Br, Bp, -Bt] (axis order r,phi,theta). This method does not perform a Fourier transform.
		"""
		raise NotImplementedError
	
	def stack_latitude(self, data):
		return data
	
	def mask(self, data):
		"""
		This allows subclasses to perform any operations on B-vector before the domain is doubled or the FFT is taken.
		"""
		return data
	
	def fft(self, data):
		return scipy.fft.fft2(data, norm='forward')
	
	def __call__(self, fname):
		B_vec = self.read(fname)
		B_vec = self.mask(B_vec)
		B_vec = self.stack_latitude(B_vec)
		return self.fft(B_vec)

class StackLatitudeMixin():
	def stack_latitude(self, data):
		n_vec, n_lon, n_lat = np.shape(data)
		if not n_lon == n_lat*2:
			raise RuntimeError(f"Unexpected FITS data size: {np.shape(data)}")
		if not n_lat%2 == 0:
			raise RuntimeError("n_lat is odd, so unclear how to split into two for stacking.")
		nlatb2 = int(n_lat/2)
		data = np.concatenate((data[:,:,nlatb2:], data, data[:,:,:nlatb2]), axis=-1)
		if not np.shape(data) == (n_vec, n_lon, 2*n_lat):
			raise RuntimeError(f"Something went wrong while stacking: {np.shape(data) = }; expected ({n_vec}, {n_lon}, {2*n_lat})")

class ExciseLatitudeMixin():
	def mask(self, data):
		#Assumes the last axis is latitude, and that it extends from -90 to 90.
		if not hasattr(self, "max_lat"):
			raise AttributeError("Set max_lat to use this class.")
		
		n_lat = np.shape(data)[-1]
		lat = np.linspace(-90,90,n_lat)
		return np.where(np.abs(lat) > self.max_lat, 0, data)

class HMIreader(FITSreader):
	def read(self, fname):
		Br = get_data_fft(f"{fname}.Br.fits")
		Bt = get_data_fft(f"{fname}.Bt.fits")
		Bp = get_data_fft(f"{fname}.Bp.fits")
		
		B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
		B_vec = np.swapaxes(B_vec, -1, -2) #The FITS files would've had spatial coordinates latitude,longitude.
		
		return B_vec

class HMIreader_dbl(StackLatitudeMixin, HMIreader):
	pass

class SOLISreader(ExciseLatitudeMixin, FITSreader):
	def read(self, fname):
		with fits.open(fname) as f:
			hdu = f[0]
			data = hdu.data
		data = np.nan_to_num(data)
	
		B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
		B_vec = np.swapaxes(B_vec, -1, -2) #The FITS files would've had spatial coordinates latitude,longitude.
		
		return B_vec

class SOLISreader_dbl(ExciseLatitudeMixin, StackLatitudeMixin, SOLISreader):
	pass
