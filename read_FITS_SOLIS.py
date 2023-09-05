"""
Read the FITS files provided by SOLIS
"""

import numpy as np
import scipy.fft
from astropy.io import fits

def get_B_vec(fname):
	"""
	Read B_vector from FITS files (synoptic vector magnetograms), Fourier-transform it, and return it as an array. A pseudo-Cartesian coordinate system is used, where we map r,phi,mu=cos(theta) to x,y,z (a right-handed coordinate system).
	
	Arguments:
		fname: string, filename that can be handled by astropy.io.fits.open
	
	"""
	with fits.open(fname) as f:
		hdu = f[0]
		data = hdu.data
	
	data = np.nan_to_num(data)
	Br, Bt, Bp, _ = scipy.fft.fft2(data, norm='forward')
	#TODO: https://magmap.nso.edu/solis/v9g-int-maj_dim-180_cmp-phi-kc.html seems to show some artefacts at high latitudes. Need to cut them out?
	
	B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
	B_vec = np.swapaxes(B_vec, -1, -2) #The FITS files would've had spatial coordinates latitude,longitude.
	
	return B_vec

def get_B_vec_dbllat(fname):
	"""
	Read B_vector from FITS files (synoptic vector magnetograms), double the domain in the latitudinal directions, Fourier-transform it, and return it as an array. A pseudo-Cartesian coordinate system is used, where we map r,phi,mu=cos(theta) to x,y,z (a right-handed coordinate system).
	
	Arguments:
		fname: string, filename that can be handled by astropy.io.fits.open
	
	"""
	with fits.open(fname) as f:
		hdu = f[0]
		data = hdu.data
	data = np.nan_to_num(data)
	
	#TODO: https://magmap.nso.edu/solis/v9g-int-maj_dim-180_cmp-phi-kc.html seems to show some artefacts at high latitudes. I also see them while plotting the magnetic field from a single synoptic magnetogram. Need to cut them out?
	
	_, n_lat, n_lon = np.shape(data)
	if not n_lon == n_lat*2:
		raise RuntimeError(f"Unexpected FITS data size: {np.shape(data)}")
	if not n_lat%2 == 0:
		raise RuntimeError("n_lat is odd, so unclear how to split into two for stacking.")
	nlatb2 = int(n_lat/2)
	data = np.concatenate((data[:,nlatb2:,:], data, data[:,:nlatb2,:]), axis=1)
	
	Br, Bt, Bp, _ = scipy.fft.fft2(data, norm='forward')
	
	B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
	B_vec = np.swapaxes(B_vec, -1, -2) #The FITS files would've had spatial coordinates latitude,longitude.
	
	return B_vec
