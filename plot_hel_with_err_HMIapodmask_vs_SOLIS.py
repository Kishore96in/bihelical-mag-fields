"""
Similar to the plot produced by reproduce_singh18_HMI_dbl, but for different CR.
"""

import numpy as np
import matplotlib.pyplot as plt

from spectrum import calc_spec, signed_loglog_plot
from read_FITS import HMIreader_dbl
from utils import jackknife, downsample_half, fig_saver
from plot_hel_with_err import E0H1_dbl
from plot_E0_twopeak import E0H1_SOLISdbl, E0H1_HMIdblapodmsk

def plot_hel_with_err_compare(res_1, res_2):
	fig,axs = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'height_ratios': [2,1]})
	
	handles = signed_loglog_plot(res_1.k, res_1.k*res_1.nimH1, axs[0,0], {'label':"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,0].loglog(res_1.k, res_1.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,0].loglog(res_1.k, np.abs(res_1.nimH1)/res_1.nimH1_err, label="$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,0].loglog(res_1.k, res_1.E0/res_1.E0_err, label="$E(k,0)$")
	
	handles = signed_loglog_plot(res_2.k, res_2.k*res_2.nimH1, axs[0,1], {'label':"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,1].loglog(res_2.k, res_2.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,1].loglog(res_2.k, np.abs(res_2.nimH1)/res_2.nimH1_err, label="$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,1].loglog(res_2.k, res_2.E0/res_2.E0_err, label="$E(k,0)$")
	
	axs[0,0].set_title(res_1.title)
	axs[0,1].set_title(res_2.title)
	
	for ax in axs[1]:
		ax.axhline(1, ls=':', c='k')
		ax.set_ylabel("|data/error|")
		ax.set_xlabel("k")
	
	return fig

if __name__ == "__main__":
	savefig = True
	savedir = "plots/hel_with_err_compare_HMIapodmask_SOLIS"
	max_lat = 50 #For HMI apodization
	threshold = 200 #For HMI masking
	
	#TODO: In the SOLIS data which I have downloaded, CR 2153 and 2154 are missing. Why?
	#Just like Singh 2018, we exclude certain Carrington rotations.
	cr_exclude = [2099, 2107, 2127, 2139, 2152, 2153, 2154, 2155, 2163, 2164, 2166, 2167]
	cr_bins = list(np.arange(2097,2196,10))
	
	save = fig_saver(savefig, savedir)
	for i in range(len(cr_bins)-1):
		print(f"{cr_bins[i] = }") #debug
		cr_list = [cr for cr in range(cr_bins[i], cr_bins[i+1]) if cr not in cr_exclude]
		figname = f"{cr_bins[i]:04d}-{cr_bins[i+1]-1:04d}.pdf"
		
		res_h = E0H1_HMIdblapodmsk(cr_list, max_lat=max_lat, threshold=threshold)
		res_h.title = "HMI"
		
		res_s = E0H1_SOLISdbl(cr_list)
		res_s.title = "SOLIS"
		
		fig = plot_hel_with_err_compare(res_h, res_s)
		fig.suptitle(f"CR: {min(cr_list):04d}–{max(cr_list):04d}")
		fig.set_size_inches(6.3,4)
		fig.tight_layout()
		
		save(fig, figname)
