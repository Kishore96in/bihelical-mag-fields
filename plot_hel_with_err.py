"""
Similar to the plot produced by reproduce_singh18_HMI_dbl, but for different CR.
"""

import numpy as np
import matplotlib.pyplot as plt

from spectrum import calc_spec, signed_loglog_plot
from read_FITS import HMIreader_dbl
from utils import jackknife, downsample_half, fig_saver

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

def E0H1_list_from_CR_list(cr_list):
	L = np.array([2*np.pi*700,2*np.pi*700]) #data will be doubled in the latitudinal direction.
	read = HMIreader_dbl()
	
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
	
	return downsample_half(k, E0_list, H1_list, axis=1)

def test_small_im(arr):
	assert max(np.abs(np.imag(arr)/np.real(arr))) < 1e-10

def real(arr):
	test_small_im(arr)
	return np.real(arr)

def plot_hel_with_err(cr_list):
	read = HMIreader_dbl()
	res = E0H1_dbl(cr_list, read)
	
	fig,axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2,1]})
	
	handles = signed_loglog_plot(res.k, res.k*res.nimH1, axs[0], {'label':"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0].loglog(res.k, res.E0, label="$E(k,0)$")
	handles.extend(h)
	# axs[0].legend(handles=handles)
	
	axs[1].loglog(res.k, np.abs(res.nimH1)/res.nimH1_err, label="$-\mathrm{Im}(k H(k,K_1))$")
	axs[1].loglog(res.k, res.E0/res.E0_err, label="$E(k,0)$")
	axs[1].axhline(1, ls=':', c='k')
	axs[1].set_ylabel("|data/error|")
	axs[1].set_xlabel("k")
	# axs[1].legend()
	
	cr_list_int = [int(cr) for cr in cr_list]
	fig.suptitle(f"CR: {min(cr_list_int)}–{max(cr_list_int)}")
	fig.set_size_inches(4,4)
	fig.tight_layout()
	
	return fig

if __name__ == "__main__":
	savefig = True
	savedir = "plots/hel_with_err"
	
	save = fig_saver(savefig, savedir)
	
	cr_bins = np.arange(2097,2268,10)
	for i in range(len(cr_bins)-1):
		cr_list = [f"{cr}" for cr in range(cr_bins[i], cr_bins[i+1])]
		figname = f"{cr_bins[i]}-{cr_bins[i+1]-1}.pdf"
		save(plot_hel_with_err(cr_list), figname)
