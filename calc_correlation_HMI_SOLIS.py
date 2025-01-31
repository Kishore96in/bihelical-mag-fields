"""
Statistical measures of how well the signs of the helicity spectra calculated from HMI and SOLIS synoptic magnetograms are correlated.

1. signed correlation coefficient (like Spearman; 1 if same sign at a particular wavenumber; else 0)

2. chi-squared estimator, which is then used to calculate a p value

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
from dataclasses import dataclass

from kishore_backpack.plotting import errorfill

from read_FITS import HMIreader_dbl, SOLISreader_dbl as SOLISreader_dbl_exc, ExciseLatitudeMixin
from utils import fig_saver
from plot_hel_with_err import E0H1_dbl

class HMIreader_dblexc(ExciseLatitudeMixin, HMIreader_dbl): pass

def calc_frachel(k, H_w_err, E_w_err):
	H, Herr = H_w_err
	E, Eerr = E_w_err
	
	frachel = k*H/E
	frachelerr = np.sqrt((Herr/H)**2 + (Eerr/E)**2)*frachel
	return frachel, frachelerr

def trunc(arr, k, k_min, k_max):
	if arr.ndim != 1: raise ValueError
	if k.ndim != 1: raise ValueError
	if len(arr) != len(k): raise ValueError
	
	ik_min = np.argmin(np.abs(k - k_min))
	ik_max = np.argmin(np.abs(k - k_max))
	return arr[ik_min:ik_max]

if __name__ == "__main__":
	savefig = True
	savedir = "plots/correlation"
	mpl.style.use("kishore.mplstyle")
	
	save = fig_saver(savefig, savedir)
	
	read_HMI = HMIreader_dblexc(max_lat=60)
	read_SOLIS = SOLISreader_dbl_exc(max_lat=60)
	
	#Just like Singh 2018, we exclude certain Carrington rotations.
	cr_exclude = [2099, 2107, 2127, 2139, 2152, 2153, 2154, 2155, 2163, 2164, 2166, 2167]
	cr_bins = np.arange(2097,2196,10)
	
	corr_sign_list = []
	corr_sign_werr_list = []
	chi2_list = []
	chi2r_list = []
	pval_list = []
	for i in range(len(cr_bins)-1):
		cr_list = [cr for cr in range(cr_bins[i], cr_bins[i+1]) if cr not in cr_exclude]
		
		res_h = E0H1_dbl(cr_list, read_HMI)
		res_s = E0H1_dbl(cr_list, read_SOLIS)
		
		k_min = max(res_h.k[1], res_s.k[1]) #k[0]==0, so we ignore that
		k_max = min(res_h.k[-1], res_s.k[-1])
		
		trunc_h = lambda arr: trunc(arr, res_h.k[1:], k_min, k_max)
		trunc_s = lambda arr: trunc(arr, res_s.k[1:], k_min, k_max)
		
		E_s = trunc_s(res_s.E0[1:])
		Eerr_s = trunc_s(res_s.E0_err[1:])
		H_s = trunc_s(res_s.nimH1[1:])
		Herr_s = trunc_s(res_s.nimH1_err[1:])
		k_s = trunc_s(res_s.k[1:])
		
		E_h = trunc_h(res_h.E0[1:])
		Eerr_h = trunc_h(res_h.E0_err[1:])
		H_h = trunc_h(res_h.nimH1[1:])
		Herr_h = trunc_h(res_h.nimH1_err[1:])
		k_h = trunc_h(res_h.k[1:])
		
		if not np.all(np.isclose(k_s, k_h)):
			raise RuntimeError("Wavenumbers are unequal")
		
		sign_matches = (H_s*H_h >= 0)
		corr_sign_list.append(
			np.average(np.where(sign_matches, 1, 0)**2),
			)
		
		sign_matches_werr = (H_s*H_h >= 0) | (Herr_s > abs(H_s)) | (Herr_h > abs(H_h))
		corr_sign_werr_list.append(
			np.average(np.where(sign_matches_werr, 1, 0)**2),
			)
		
		frachel_h, frachelerr_h = calc_frachel(k_h, (H_h, Herr_h), (E_h, Eerr_h))
		frachel_s, frachelerr_s = calc_frachel(k_s, (H_s, Herr_s), (E_s, Eerr_s))
		
		ndof = len(frachel_h)
		rv = scipy.stats.chi2(ndof)
		chi2 = np.sum((frachel_h - frachel_s)**2/(frachelerr_h**2 + frachelerr_s**2))
		
		chi2_list.append(chi2)
		chi2r_list.append(chi2/ndof)
		pval_list.append(1-rv.cdf(chi2))
	
	cr_bin_labels = (cr_bins[:-1] + cr_bins[1:])/2
	
	fig, ax = plt.subplots()
	ax.plot(cr_bin_labels, corr_sign_list)
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$\sigma_\mathrm{sign}$")
	ax.set_ylim(0,1)
	save(fig, "corr_sign.pdf")
	
	fig, ax = plt.subplots()
	ax.plot(cr_bin_labels, corr_sign_werr_list)
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$\sigma_\mathrm{sign}$")
	ax.set_ylim(0,1)
	save(fig, "corr_sign_werr.pdf")
	
	fig, ax = plt.subplots()
	ax.plot(cr_bin_labels, chi2_list)
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$\chi^2$")
	save(fig, "chi2.pdf")
	
	fig, ax = plt.subplots()
	ax.plot(cr_bin_labels, pval_list)
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$p$")
	# ax.set_yscale('log')
	save(fig, "pval.pdf")
