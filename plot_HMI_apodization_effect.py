"""
Check if the spectra calculated from HMI data for Carrington rotations 2177-2186 are significantly affected by discarding data at high latitudes or by masking the weak-field regions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from read_FITS import HMIreader_dbl, ExciseLatitudeMixin
from plot_hel_with_err import real, E0H1_dbl
from spectrum import calc_spec, signed_loglog_plot
from utils import jackknife, downsample_half, fig_saver

class HMIreader_dblexc(ExciseLatitudeMixin, HMIreader_dbl):
	pass

if __name__ == "__main__":
	cr_list = np.arange(2207,2217)
	max_lat = 60
	savefig = True
	savedir = "plots"
	
	save = fig_saver(savefig, savedir)
	
	read = HMIreader_dbl()
	read_apod = HMIreader_dblexc(max_lat=max_lat)
	
	r = E0H1_dbl(cr_list, read)
	r_apod = E0H1_dbl(cr_list, read_apod)
	
	#Compare HMI with SOLIS
	fig,axs = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'height_ratios': [2,1]})
	
	handles = signed_loglog_plot(r.k, r.k*r.nimH1, axs[0,0], {'label':"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,0].loglog(r.k, r.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,0].loglog(r.k, np.abs(r.nimH1)/r.nimH1_err, label="$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,0].loglog(r.k, r.E0/r.E0_err, label="$E(k,0)$")
	
	
	handles = signed_loglog_plot(r_apod.k, r_apod.k*r_apod.nimH1, axs[0,1], {'label':"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,1].loglog(r_apod.k, r_apod.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,1].loglog(r_apod.k, np.abs(r_apod.nimH1)/r_apod.nimH1_err, label="$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,1].loglog(r_apod.k, r_apod.E0/r_apod.E0_err, label="$E(k,0)$")
	
	axs[0,0].set_title("Full")
	axs[0,1].set_title(rf"$\left|\lambda\right| < {max_lat}^{{\circ}}$")
	fig.suptitle(f"CR {min(cr_list)}–{max(cr_list)}")
	
	for ax in axs[1]:
		ax.axhline(1, ls=':', c='k')
		ax.set_ylabel("|data/error|")
		ax.set_xlabel("k")
	
	fig.set_size_inches(6,4)
	fig.tight_layout()
	save(fig, "effect_HMI_apodization.pdf")
