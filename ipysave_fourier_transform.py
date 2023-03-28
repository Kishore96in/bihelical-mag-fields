from astropy.io import fits
from astropy.wcs import WCS
import scipy.fft
import numpy as np

if __name__ == "__main__":
	def get_data_fft(fname):
		with fits.open(fname) as f:
			hdu = f[0]
			data = hdu.data
		return scipy.fft.fft2(data)
	
	Br = get_data_fft("hmi.b_synoptic_small.2267.Br.fits")
	Bt = get_data_fft("hmi.b_synoptic_small.2267.Bt.fits")
	Bp = get_data_fft("hmi.b_synoptic_small.2267.Bp.fits")
	
	B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
	B_vec = np.swapaxes(B_vec, -1, -2)
	
	#Approximate wavenumbers in both the directions
	_, n_lon, n_lat = np.shape(B_vec)
	L_lat = 2
	L_lon = 360
	
	L = min(L_lat, L_lon) #We use this to make the wavevector an integer (to ease binning). All 'wavevectors' below then need to be multiplied by 2pi/L to get the actual wavevector.
	k_lat = L*scipy.fft.fftfreq(n_lat, d=L_lat)
	k_lon = L*scipy.fft.fftfreq(n_lon, d=L_lon)
	
	k_lon_g, k_lat_g = np.meshgrid(k_lon, k_lat, indexing='ij')
	k_rad_g = np.zeros_like(k_lat_g)
	k_vec = np.stack([k_rad_g, k_lon_g, k_lat_g])
	k_mag = np.sqrt(np.sum(k_vec**2, axis=0))
	
	Mij = B_vec[:,None,:,:,None,None]*B_vec[None,:,None,None,:,:]
	k1_vec = k_vec[:,:,:,None,None]
	k2_vec = k_vec[:,None,None,:,:]
	K_vec = k1 + k2
	k_vec = (k1 - k2)/2
	k_mag = np.sqrt(np.sum(k_vec**2, axis=0))
	
	k_mag_round = np.round(k_mag) #Used to bin the spectra
	nk = int(np.min(np.floor(np.array([n_lat,n_lon])/2)))
	
	E = np.zeros([n_lat, n_long, nk])
	H = np.zeros([n_lat, n_long, nk])
	
	E_integrand = np.einsum("ii...", Mij)
	H_integrand = np.einsum("ii...", np.cross(k_vec, Mij, axis=0))
	
	for k in range(nk):
		E[...,k] = np.sum(np.where(k_mag_round == k, E_integrand, 0), axis=(-1,-2))/2
		H[...,k] = np.sum(np.where(k_mag_round == k, H_integrand, 0), axis=(-1,-2))/(2*(2*np.pi*k/L))
