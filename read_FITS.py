"""
Classes to read HMI and SOLIS synoptic maps
"""

import os
import numpy as np
import scipy.fft
from astropy.io import fits

class FITSreader():
	"""
	Intended to be used like
	```
	r = FITSreader(mask_threshold=100, dbllat=True)
	B_vec = r(fname)
	```
	"""
	def __init__(self, **kwargs):
		for key, value in kwargs.items():
			setattr(self, key, value)
	
	def read(self, fname):
		"""
		Read B_vector from FITS files (synoptic vector magnetograms) and return it as an array. A pseudo-Cartesian coordinate system is used, where we map r,phi,mu=cos(theta) to x,y,z (a right-handed coordinate system).
		"""
		raise NotImplementedError
	
	def stack_latitude(self, data):
		return data
	
	def mask(self, data):
		"""
		This allows subclasses to perform any operations on B-vector before the domain is doubled or the FFT is taken.
		"""
		return data
	
	def apodize(self, data):
		"""
		This allows subclasses to perform any operations on B-vector before the domain is doubled or the FFT is taken.
		"""
		return data
	
	def fft(self, data):
		"""
		Normalization of the Fourier transform is chosen to match that of IDL.
		"""
		return scipy.fft.fft2(data, norm='forward')
	
	def __call__(self, fname):
		B_vec = self.read(fname)
		B_vec = self.mask(B_vec)
		B_vec = self.apodize(B_vec)
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
		return data

class MaskWeakMixin():
	def mask(self, B_vec):
		if not hasattr(self, "threshold"):
			raise AttributeError("Set threshold to use this class.")
		
		Bmag = np.sqrt(np.sum(B_vec**2, axis=0, keepdims=True))
		return np.where(Bmag>self.threshold, B_vec, 0)

class ExciseLatitudeMixin():
	def apodize(self, data):
		#Assumes the last axis is latitude, and that it extends from -90 to 90.
		if not hasattr(self, "max_lat"):
			raise AttributeError("Set max_lat to use this class.")
		
		n_lat = np.shape(data)[-1]
		lat = np.linspace(-90,90,n_lat)
		return np.where(np.abs(lat) > self.max_lat, 0, data)

class HMIreader(FITSreader):
	def _get_data(self, fname):
		with fits.open(fname) as f:
			hdu = f[0]
			data = hdu.data
		data = np.nan_to_num(data)
		return data
	
	def get_Brtp(self, fname):
		Br = self._get_data(f"{fname}.Br.fits")
		Bt = self._get_data(f"{fname}.Bt.fits")
		Bp = self._get_data(f"{fname}.Bp.fits")
		return np.array([Br, Bt, Bp])
	
	def read(self, fname):
		Br, Bt, Bp = self.get_Brtp(fname)
		B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
		B_vec = np.swapaxes(B_vec, -1, -2) #The FITS files would've had spatial coordinates latitude,longitude.
		return B_vec

class HMIreader_dbl(StackLatitudeMixin, HMIreader):
	pass

class SOLISreader(ExciseLatitudeMixin, FITSreader):
	def get_Brtp(self, fname):
		with fits.open(fname) as f:
			hdu = f[0]
			data = hdu.data
		data = np.nan_to_num(data)
		Br, Bt, Bp, _ = data
		return np.array([Br, Bt, Bp])
	
	def read(self, fname):
		Br, Bt, Bp = self.get_Brtp(fname)
		B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
		B_vec = np.swapaxes(B_vec, -1, -2) #The FITS files would've had spatial coordinates latitude,longitude.
		return B_vec

class SOLISreader_dbl(StackLatitudeMixin, SOLISreader):
	pass

def get_fname_SOLIS(cr):
	"""
	Get the filename of the FITS file for a given Carrington rotation.
	
	Arguments:
		cr: int, Carrington rotation number
	"""
	img_loc = "images_SOLIS"
	
	match = lambda f: f[-30:] == f"c{cr:04d}_000_int-mas_dim-180.fits" and f[:5] == "kcv9g"
	files = [f for f in os.listdir(img_loc) if match(f)]
	if len(files) > 1:
		raise RuntimeError(f"Too many matches for {cr = }; {files = }")
	return os.path.join(img_loc, files[0])
