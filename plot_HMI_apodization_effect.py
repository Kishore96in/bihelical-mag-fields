"""
Check if the spectra calculated from HMI data for Carrington rotations 2177-2186 are significantly affected by discarding data at high latitudes or by masking the weak-field regions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from read_FITS import HMIreader_dbl, ExciseLatitudeMixin
from plot_hel_with_err import real
from spectrum import calc_spec, signed_loglog_plot
from utils import jackknife, downsample_half, fig_saver

class HMIreader_dblexc(ExciseLatitudeMixin, HMIreader_dbl):
	pass

#TODO: This seems general enough that I might move it to another file. But note that the length is hardcoded. Can at least use in plot_hel_with_err, I guess.
def E0H1_dbl(cr_list, read):
	L = np.array([2*np.pi*700,2*np.pi*700]) #data will be doubled in the latitudinal direction.
	
	E0_list = []
	H1_list = []
	for cr in cr_list:
		B_vec = read(f"images/hmi.b_synoptic_small.rebinned.{cr}")
		k, E0, _ = calc_spec(B_vec, K=np.array([0,0]), L=L)
		_, _, H1 = calc_spec(B_vec, K=np.array([0,2]), L=L, shift_onesided=0)
		
		E0_list.append(E0)
		H1_list.append(H1)
	
	E0_list = np.array(E0_list)
	H1_list = np.array(H1_list)
	
	k, E0_list, H1_list = downsample_half(k, E0_list, H1_list, axis=1)
	
	E0, E0_err = jackknife(E0_list, axis=0)
	nimH1, nimH1_err = jackknife(-np.imag(H1_list), axis=0)
	
	#Avoid some annoying matplotlib warnings
	E0 = real(E0)
	E0_err = real(E0_err)
	nimH1 = real(nimH1)
	nimH1_err = real(nimH1_err)
	
	return result(k, E0, E0_err, nimH1, nimH1_err)

class result():
	def __init__(self, k, E0, E0_err, nimH1, nimH1_err):
		self.k = k
		self.E0 = E0
		self.E0_err = E0_err
		self.nimH1 = nimH1
		self.nimH1_err = nimH1_err

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
