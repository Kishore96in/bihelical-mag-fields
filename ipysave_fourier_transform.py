from astropy.io import fits
from astropy.wcs import WCS
import scipy.fft
import numpy as np

def get_data_fft(fname):
	"""
	Given the name of a FITS file, return the Fourier transform of the data in it
	"""
	with fits.open(fname) as f:
		hdu = f[0]
		data = hdu.data
	data = np.nan_to_num(data)
	return scipy.fft.fft2(data)

def calc_spec(fname, K, get_fft=get_data_fft, L=None, shift_onesided=True):
	"""
	Arguments:
		fname: string of the form "hmi.b_synoptic_small.2267". The input file for Br should be called fname+".Br.fits" (and similar for Bt, Bp). Resulting filename may be anything that is handled by astropy.io.fits.open
		K: 2-element numpy array, large-scale wavevector to handle. The true wavevector is 2*pi*K/min(L)
		get_fft: function that when given the path to a FITS file, returns the Fourier transform of the data stored in it.
		L: 2-element numpy array, length of the domain along the latitudinal and longitudinal directions. Default: np.array([2*np.pi, 2*np.pi])
		shift_onesided: bool
			True: Mij = B_i(k+K) B_j^*(k) (the one used in Nishant's 2018 paper)
			False: Mij = B_i(k+K/2) B_j^*(k-K/2) (the correct definition)
	"""
	
	if L is None:
		L = np.array([2*np.pi, 2*np.pi])
	
	Br = get_fft(f"{fname}.Br.fits")
	Bt = get_fft(f"{fname}.Bt.fits")
	Bp = get_fft(f"{fname}.Bp.fits")
	
	B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
	B_vec = np.swapaxes(B_vec, -1, -2)
	
	#Approximate wavenumbers in both the directions
	_, n_lon, n_lat = np.shape(B_vec)
	L_lat, L_lon = L
	
	L_min = min(L) #We use this to make the wavevector an integer (to ease binning). All 'wavevectors' below then need to be multiplied by 2pi/L to get the actual wavevector.
	nk = int(np.min(np.floor(np.array([n_lat,n_lon])/2))) #Maximum 'magnitude' of the wavevectors
	k_lat = L_min*n_lat*scipy.fft.fftfreq(n_lat, d=L_lat)
	k_lon = L_min*n_lon*scipy.fft.fftfreq(n_lon, d=L_lon)
	
	k_lon_g, k_lat_g = np.meshgrid(k_lon, k_lat, indexing='ij')
	k_rad_g = np.zeros_like(k_lat_g)
	k_vec = np.stack([k_rad_g, k_lon_g, k_lat_g])
	k_mag = np.sqrt(np.sum(k_vec**2, axis=0))
	
	
	if shift_onesided:
		Mij = np.roll(B_vec, shift=np.round(-K).astype(int), axis=(1,2))[:,None,:,:]*np.conj(B_vec)[None,:,:,:]
	else:
		#NOTE: Below, if K=(0,1), it will just be rounded to 0,1, resulting in no shift being applied. The one-sided shift used above is a workaround for that
		Mij = np.roll(B_vec, shift=np.round(-K/2).astype(int), axis=(1,2))[:,None,:,:]*np.roll(np.conj(B_vec), shift=np.round(K/2).astype(int), axis=(1,2))[None,:,:,:]
	k_mag = np.sqrt(np.sum(k_vec**2, axis=0))
	
	k_mag_round = np.round(k_mag) #Used to bin the spectra
	
	E = np.zeros(nk, dtype=complex)
	H = np.zeros(nk, dtype=complex)
	
	E_integrand = np.einsum("ii...", Mij)
	H_integrand = 1j*np.einsum("ii...", np.cross(k_vec, Mij, axis=0))
	
	for k in range(nk):
		E[...,k] = np.sum(np.where(k_mag_round == k, E_integrand, 0), axis=(-1,-2))/2
		H[...,k] = np.sum(np.where(k_mag_round == k, H_integrand, 0), axis=(-1,-2))/(2*np.pi*k/L_min)
	
	k = (2*np.pi/L_min)*np.arange(nk)
	return k, E, H

def signed_loglog_plot(k, spec, ax, line_params=None):
	where_pos = np.where(spec>=0)[0]
	where_neg = np.where(spec<0)[0]
	spec = np.abs(spec)
	
	params_pos = {'facecolors':'none', 'edgecolors':'r', 'marker':'o'}
	params_neg = {'c':'b', 'marker':'o'}
	if line_params is None:
		line_params = dict()
	
	l1 = ax.loglog(k, spec, **line_params)[0]
	l2 = ax.scatter(k[where_pos], spec[where_pos], **params_pos, label="positive")
	l3 = ax.scatter(k[where_neg], spec[where_neg], **params_neg, label="negative")
	
	return [l1, l2, l3]

if __name__ == "__main__":
	L = np.array([2,360])
	k, E1, H1 = calc_spec("hmi.b_synoptic_small.2267", K=np.array([0,1]), L=L, shift_onesided=False)
