"""
Look at Carrington rotations 2177-2186, and check the robustness of the two-peaked structure in the magnetic energy spectrum.
Main checks:
1. Is it also visible in SOLIS data?
2. Does it still persist when the weak-field regions are masked
3. Does it persist even when the domain is not doubled?

Secondary checks:
4. Do we recover the same helicity spectra from SOLIS and HMI?
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from read_FITS import get_B_vec as get_HMI
from mask_weak import get_B_vec as get_HMI_msk
from read_FITS_SOLIS import get_B_vec_dbllat as get_SOLIS2
from plot_hel_with_err import E0H1_list_from_CR_list as E0H1_list_from_CR_list_HMI2, real
from spectrum import calc_spec, signed_loglog_plot
from utils import jackknife, downsample_half, fig_saver

def E0H1_HMIdbl(cr_list):
	k, E0_list, H1_list = E0H1_list_from_CR_list_HMI2(cr_list)
	
	E0, E0_err = jackknife(E0_list, axis=0)
	nimH1, nimH1_err = jackknife(-np.imag(H1_list), axis=0)
	
	#Avoid some annoying matplotlib warnings
	E0 = real(E0)
	E0_err = real(E0_err)
	nimH1 = real(nimH1)
	nimH1_err = real(nimH1_err)
	
	return result(k, E0, E0_err, nimH1, nimH1_err)

def E0H1_HMIsgl(cr_list):
	L = np.array([2*np.pi*700,np.pi*700])
	
	E0_list = []
	H1_list = []
	for cr in cr_list:
		B_vec = get_HMI(f"images/hmi.b_synoptic_small.rebinned.{cr}")
		k, E0, _ = calc_spec(B_vec, K=np.array([0,0]), L=L)
		_, _, H1 = calc_spec(B_vec, K=np.array([0,1]), L=L)
		
		E0_list.append(E0)
		H1_list.append(H1)
	
	E0_list = np.array(E0_list)
	H1_list = np.array(H1_list)
	
	E0, E0_err = jackknife(E0_list, axis=0)
	nimH1, nimH1_err = jackknife(-np.imag(H1_list), axis=0)
	
	#Avoid some annoying matplotlib warnings
	E0 = real(E0)
	E0_err = real(E0_err)
	nimH1 = real(nimH1)
	nimH1_err = real(nimH1_err)
	
	return result(k, E0, E0_err, nimH1, nimH1_err)


def E0H1_HMIsglmsk(cr_list):
	L = np.array([2*np.pi*700,np.pi*700])
	
	E0_list = []
	H1_list = []
	for cr in cr_list:
		B_vec = get_HMI_msk(f"images/hmi.b_synoptic_small.rebinned.{cr}", threshold=150)
		k, E0, _ = calc_spec(B_vec, K=np.array([0,0]), L=L)
		_, _, H1 = calc_spec(B_vec, K=np.array([0,1]), L=L)
		
		E0_list.append(E0)
		H1_list.append(H1)
	
	E0_list = np.array(E0_list)
	H1_list = np.array(H1_list)
	
	E0, E0_err = jackknife(E0_list, axis=0)
	nimH1, nimH1_err = jackknife(-np.imag(H1_list), axis=0)
	
	#Avoid some annoying matplotlib warnings
	E0 = real(E0)
	E0_err = real(E0_err)
	nimH1 = real(nimH1)
	nimH1_err = real(nimH1_err)
	
	return result(k, E0, E0_err, nimH1, nimH1_err)

def get_fname_SOLIS(cr):
	"""
	Get the filename of the FITS file for a given Carrington rotation.
	
	Arguments:
		cr: int, Carrington rotation number
	"""
	img_loc = "images_SOLIS"
	
	match = lambda f: f[-30:] == f"c{cr:04d}_000_int-mas_dim-180.fits" and f[:5] == "kcv9g"
	files = [f for f in os.listdir(img_loc) if match(f)]
	if len(files) > 1:
		raise RuntimeError(f"Too many matches for {cr = }; {files = }")
	return os.path.join(img_loc, files[0])

def E0H1_SOLISdbl(cr_list):
	L = np.array([2*np.pi*700,2*np.pi*700]) #data will be doubled in the latitudinal direction.
	
	E0_list = []
	H1_list = []
	for cr in cr_list:
		B_vec = get_SOLIS2(get_fname_SOLIS(cr))
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
	cr_list = np.arange(2177,2187)
	savefig = True
	savedir = "plots/test_twopeak"
	
	save = fig_saver(savefig, savedir)
	
	r_h1 = E0H1_HMIsgl(cr_list)
	r_h1m = E0H1_HMIsglmsk(cr_list)
	r_h2 = E0H1_HMIdbl(cr_list)
	r_s2 = E0H1_SOLISdbl(cr_list)
	
	#Compare HMI with SOLIS
	fig,axs = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'height_ratios': [2,1]})
	
	handles = signed_loglog_plot(r_h2.k, r_h2.k*r_h2.nimH1, axs[0,0], {'label':"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,0].loglog(r_h2.k, r_h2.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,0].loglog(r_h2.k, np.abs(r_h2.nimH1)/r_h2.nimH1_err, label="$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,0].loglog(r_h2.k, r_h2.E0/r_h2.E0_err, label="$E(k,0)$")
	
	
	handles = signed_loglog_plot(r_s2.k, r_s2.k*r_s2.nimH1, axs[0,1], {'label':"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,1].loglog(r_s2.k, r_s2.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,1].loglog(r_s2.k, np.abs(r_s2.nimH1)/r_s2.nimH1_err, label="$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,1].loglog(r_s2.k, r_s2.E0/r_s2.E0_err, label="$E(k,0)$")
	
	axs[0,0].set_title("HMI")
	axs[0,1].set_title("SOLIS")
	fig.suptitle(f"CR {min(cr_list)}–{max(cr_list)}")
	
	for ax in axs[1]:
		ax.axhline(1, ls=':', c='k')
		ax.set_ylabel("|data/error|")
		ax.set_xlabel("k")
	
	save(fig, "compare_HMI_SOLIS.pdf")
	
	#See if domain-doubling changes the energy spectrum for the HMI data
	fig,ax = plt.subplots()
	ax.loglog(r_h1.k, r_h1.E0, label="undoubled")
	ax.loglog(r_h2.k, r_h2.E0, label="doubled", ls='--')
	ax.loglog(r_h1m.k, r_h1m.E0, label="masked", ls=':')
	ax.legend()
	ax.set_ylabel("$E(k,0)$")
	ax.set_xlabel("$k$")
	fig.suptitle(f"CR {min(cr_list)}–{max(cr_list)}")
	
	#TODO: Perhaps it is not surprising that the low-k peak disappears in the masked case. After all, the strong-field regions have very small length scales. But then, that means that this peak may just be an artefact of disambiguation.
	
	save(fig, "effect_double_mask.pdf")
