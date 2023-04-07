"""
Read the FITS files provided by HMI
"""

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
	Read B_vector from FITS files, Fourier-transform it, and return it as an array.
	
	Arguments:
		fname: string of the form "hmi.b_synoptic_small.2267". The input file for Br should be called fname+".Br.fits" (and similar for Bt, Bp). Resulting filename may be anything that is handled by astropy.io.fits.open
	
	"""
	Br = get_data_fft(f"{fname}.Br.fits")
	Bt = get_data_fft(f"{fname}.Bt.fits")
	Bp = get_data_fft(f"{fname}.Bp.fits")
	
	B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
	B_vec = np.swapaxes(B_vec, -1, -2)
	
	return B_vec
