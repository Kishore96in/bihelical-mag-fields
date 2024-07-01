"""
Fourier-filter the HMI and SOLIs synoptic maps for a particular CR and see if they look similar.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from read_FITS import HMIreader, SOLISreader
from kishore_backpack.spectrum import filter_fourier
from utils import fig_saver

if __name__ == "__main__":
	cr = 2180
	k_max = 1e-2
	savefig = True
	savedir=  "plots/test_twopeak"
	
	save = fig_saver(savefig, savedir)
	read_h = HMIreader()
	read_s = SOLISreader()
	
	Bvec_h = read_h.get_Brtp(read_h.get_fname(cr))
	Bvec_s = read_s.get_Brtp(read_s.get_fname(cr))
	
	Bvec_h_filt = filter_fourier(Bvec_h, 0, k_max, k_axes=[-1,-2], L=[2*np.pi*700, np.pi*700])
	Bvec_s_filt = filter_fourier(Bvec_s, 0, k_max, k_axes=[-1,-2], L=[2*np.pi*700, np.pi*700])
	
	fig = plt.figure()
	gs = mpl.gridspec.GridSpec(1,2, width_ratios=[0.95, 0.05])
	gsl = mpl.gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gs[0], hspace=0.5)
	ax0 = fig.add_subplot(gsl[0])
	ax1 = fig.add_subplot(gsl[1])
	ax_cbar = fig.add_subplot(gs[1])
	
	vmax = max(np.max(np.abs(Bvec_h_filt)), np.max(np.abs(Bvec_s_filt)))
	
	im_kwargs = {
		'origin': 'lower',
		'extent': [0,360,-90,90],
		'vmin': -vmax,
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
	
	assert k_max == 1e-2
	fig.suptitle(rf"CR: {cr:04d}, $0 \leq k < 10^{{-2}}$")
	fig.set_size_inches(4,4)
	fig.tight_layout()
	
	save(fig, "synoptic_lowpass.pdf")
