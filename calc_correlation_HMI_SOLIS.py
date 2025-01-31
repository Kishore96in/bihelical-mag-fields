"""
Statistical measures of how well the signs of the helicity spectra calculated from HMI and SOLIS synoptic magnetograms are correlated.

1. signed correlation coefficient (like Spearman; 1 if same sign at a particular wavenumber; else 0)

2. chi-squared estimator, which is then used to calculate a p value

Also computes the above in restricted wavenumber bands, to check if perhaps, e.g., the helicity spectrum at large wavenumbers is more reliable.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
from dataclasses import dataclass
import functools

from kishore_backpack.plotting import errorfill

from read_FITS import HMIreader_dbl, SOLISreader_dbl as SOLISreader_dbl_exc, ExciseLatitudeMixin
from utils import fig_saver
from plot_hel_with_err import E0H1_dbl as E0H1_dbl_uncached

E0H1_dbl = functools.cache(E0H1_dbl_uncached)

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

@dataclass
class res_for_kmax:
	kmax: float
	corr_sign_vs_cr: list
	corr_sign_werr_vs_cr: list
	chi2_vs_cr: list
	chi2r_vs_cr: list
	pval_vs_cr: list
	cr_labels: list

def calc_stats_for_kmax(kmax, cr_bins):
	"""
	kmax: float
	cr_bins: iterable of tuples; outer iterable indexes the bins, while the inner tuple is the list of CRs in each bin
	"""
	corr_sign_list = []
	corr_sign_werr_list = []
	chi2_list = []
	chi2r_list = []
	pval_list = []
	cr_labels = []
	for cr_list in cr_bins:
		cr_labels.append((cr_list[0] + cr_list[1])/2)
		
		res_h = E0H1_dbl(cr_list, read_HMI)
		res_s = E0H1_dbl(cr_list, read_SOLIS)
		
		k_min = max(res_h.k[1], res_s.k[1]) #k[0]==0, so we ignore that
		k_max = min(res_h.k[-1], res_s.k[-1], kmax)
		
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
	
	return res_for_kmax(
		kmax = k_max,
		corr_sign_vs_cr = corr_sign_list,
		corr_sign_werr_vs_cr = corr_sign_werr_list,
		chi2_vs_cr = chi2_list,
		chi2r_vs_cr = chi2r_list,
		pval_vs_cr = pval_list,
		cr_labels = cr_labels,
		)

if __name__ == "__main__":
	savefig = True
	savedir = "plots/correlation"
	mpl.style.use("kishore.mplstyle")
	
	save = fig_saver(savefig, savedir)
	
	read_HMI = HMIreader_dblexc(max_lat=60)
	read_SOLIS = SOLISreader_dbl_exc(max_lat=60)
	
	#Just like Singh 2018, we exclude certain Carrington rotations.
	cr_exclude = [2099, 2107, 2127, 2139, 2152, 2153, 2154, 2155, 2163, 2164, 2166, 2167]
	cr_bins_bounds = np.arange(2097,2196,10)
	k_max_list = [np.inf, 1e-1, 2e-2]
	
	n_bins = len(cr_bins_bounds)-1
	cr_bins = [tuple(cr for cr in range(cr_bins_bounds[i], cr_bins_bounds[i+1]) if cr not in cr_exclude) for i in range(n_bins)]
	
	res_list = [calc_stats_for_kmax(kmax, cr_bins) for kmax in k_max_list]
	
	fig, ax = plt.subplots()
	for res in res_list:
		ax.plot(res.cr_labels, res.corr_sign_vs_cr, label=f"{res.kmax:.2f}")
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$\sigma_\mathrm{sign}$")
	ax.set_ylim(0,1)
	ax.legend(title=r"$k_\mathrm{max}$")
	save(fig, "corr_sign.pdf")
	
	fig, ax = plt.subplots()
	for res in res_list:
		ax.plot(res.cr_labels, res.corr_sign_werr_vs_cr, label=f"{res.kmax:.2f}")
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$\sigma_\mathrm{sign}$")
	ax.set_ylim(0,1)
	ax.legend(title=r"$k_\mathrm{max}$")
	save(fig, "corr_sign_werr.pdf")
	
	fig, ax = plt.subplots()
	for res in res_list:
		ax.plot(res.cr_labels, res.chi2_vs_cr, label=f"{res.kmax:.2f}")
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$\chi^2$")
	ax.legend(title=r"$k_\mathrm{max}$")
	save(fig, "chi2.pdf")
	
	fig, ax = plt.subplots()
	for res in res_list:
		ax.plot(res.cr_labels, res.chi2r_vs_cr, label=f"{res.kmax:.2f}")
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$\chi^2/n$")
	ax.legend(title=r"$k_\mathrm{max}$")
	save(fig, "chi2r.pdf")
	
	fig, ax = plt.subplots()
	for res in res_list:
		ax.plot(res.cr_labels, res.pval_vs_cr, label=f"{res.kmax:.2f}")
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$p$")
	# ax.set_yscale('log')
	ax.legend(title=r"$k_\mathrm{max}$")
	save(fig, "pval.pdf")
