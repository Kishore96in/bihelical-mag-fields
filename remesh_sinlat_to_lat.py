"""
Given a FITS file in which the data is equispaced in sine(latitude), remesh the data and save a FITS file where the data is equispaced in latitude
"""

import numpy as np
import os
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator as RGI

def remesh(fname, out):
	"""
	Handles a single FITS file
	
	Arguments:
		fname: str, path to the FITS file
		out: str, path to save the FITS file containing the remeshed data
	"""
	with fits.open(fname) as f:
		hdu = f[0]
		header = hdu.header
		data = hdu.data
	
	header['CUNIT2'] = "" #sine(latitude) is dimensionless, but the HMI maps erroneously have this field nonempty.
	
	i_sinlat = np.arange(np.shape(data)[0])
	i_lon = np.arange(np.shape(data)[1])
	
	w = WCS(header)
	_, sinlat = w.array_index_to_world_values(i_sinlat, np.zeros_like(i_sinlat))
	lon ,_ = w.array_index_to_world_values(np.zeros_like(i_lon), i_lon)
	
	#NOTE: removing nan because they will make everything nan on taking fft or interpolating.
	data = np.nan_to_num(data)
	lat = np.arcsin(sinlat)*180/np.pi #get latitude in degrees
	r = RGI((lat, lon), data, method='cubic',bounds_error=False)
	
	lat_new = np.linspace(lat[0], lat[-1], len(lat))
	new_grid = tuple(np.meshgrid(lat_new, lon, indexing='ij'))
	data_remeshed = r(new_grid)
	
	#Prepare header for the new FITS file
	header['CRPIX2'] = 0.5
	header.comments['CRPIX2'] = ""
	header['CRVAL2'] = lat_new[0]
	header.comments['CRVAL2'] = ""
	header['CDELT2'] = lat_new[1]-lat_new[0]
	header.comments['CDELT2'] = "[degree/pixel]"
	header['CUNIT2'] = "deg"
	header.comments['CTYPE2'] = "Latitude"
	header.add_comment("  Rebinned to have the y-axis equispaced in degrees by Kishore G. (kishore96@gmail.com).")
	
	fits.writeto(out, data_remeshed, header=header)

def gen_out_name(fname):
	"""
	Generate name of output file from the name of the input file. Just inserts the word 'rebinned' after the second dot in fname
	"""
	dirname = os.path.dirname(fname)
	out = os.path.basename(fname)
	
	out = out.split('.')
	out.insert(2, "rebinned")
	out = ".".join(out)
	
	out = os.path.join(dirname, out)
	
	assert out != fname #Just a sanity check
	return out

if __name__ == "__main__":
	fname = 'hmi.b_synoptic_small.2267.Br.fits'
	remesh(fname, gen_out_name(fname))