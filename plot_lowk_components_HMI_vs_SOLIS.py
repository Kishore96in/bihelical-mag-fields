"""
Fourier-filter the HMI and SOLIs synoptic maps for a particular CR and see if they look similar.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from read_FITS import HMIreader, SOLISreader
from kishore_backpack.spectrum import filter_fourier
from plot_E0_twopeak import get_fname_SOLIS

if __name__ == "__main__":
	cr = 2180
	
	read_h = HMIreader()
	read_s = SOLISreader()
	
	Bvec_h = read_h.get_Brtp(f"images/hmi.b_synoptic_small.rebinned.{cr}")
	Bvec_s = read_s.get_Brtp(get_fname_SOLIS(cr))
	
	Bvec_h_filt = filter_fourier(Bvec_h, 0, 1e-2, k_axes=[-1,-2], L=[2*np.pi*700, np.pi*700])
	Bvec_s_filt = filter_fourier(Bvec_s, 0, 1e-2, k_axes=[-1,-2], L=[2*np.pi*700, np.pi*700])
	
	fig = plt.figure()
	gs = mpl.gridspec.GridSpec(1,2, width_ratios=[0.95, 0.05])
	gsl = mpl.gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gs[0], hspace=0.3)
	ax0 = fig.add_subplot(gsl[0])
	ax1 = fig.add_subplot(gsl[1])
	ax_cbar = fig.add_subplot(gs[1])
	
	vmin = min(np.min(Bvec_h_filt), np.min(Bvec_s_filt))
	vmax = max(np.max(Bvec_h_filt), np.max(Bvec_s_filt))
	
	im_kwargs = {
		'origin': 'lower',
		'extent': [0,360,-90,90],
		'vmin': vmin,
		'vmax': vmax,
		'cmap': 'bwr',
		}
	
	im0 = ax0.imshow(Bvec_h_filt[0], **im_kwargs)
	ax0.set_title("HMI")
	ax0.set_ylabel(r"$\lambda$ (degrees)")
	
	im1 = ax1.imshow(Bvec_s_filt[0], **im_kwargs)
	ax1.set_title("SOLIS")
	ax1.set_ylabel(r"$\lambda$ (degrees)")
	ax1.set_xlabel(r"$\phi$ (degrees)")
	
	c = fig.colorbar(im1, cax=ax_cbar)
	c.set_label(r"$B_r$")
	
	fig.set_size_inches(4,4)
	fig.tight_layout()
	
	plt.show()
