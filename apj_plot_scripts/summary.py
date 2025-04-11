"""
Plots that summarize the important findings.

1. HMI energy spectra show a peak at low k during cycle minima
2. This peak can be suppressed by either truncating high latitudes or masking weak-field regions.
3. Even after apodizing, the helicity spectra from HMI and SOLIS disagree at large k.
"""

import sys
import pathlib
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


from spectrum import signed_loglog_plot
from read_FITS import HMIreader_dbl, SOLISreader_dbl as SOLISreader_dbl_exc, ExciseLatitudeMixin, MaskWeakMixin
from utils import fig_saver, errorfill
from plot_hel_with_err import E0H1_dbl
from config import cr_SOLIS_bad

class HMIreader_dblexc(ExciseLatitudeMixin, HMIreader_dbl): pass
class HMIreader_dblmsk(MaskWeakMixin, HMIreader_dbl): pass

if __name__ == "__main__":
	savefig = True
	savedir = root/"plots/summary_apj"
	mpl.style.use(root/"kishore_apj.mplstyle")
	
	save = fig_saver(savefig, savedir)
	
	read_HMI = HMIreader_dbl()
	read_HMIapod = HMIreader_dblexc(max_lat=60)
	read_HMImask200 = HMIreader_dblmsk(threshold=200)
	read_HMImask50 = HMIreader_dblmsk(threshold=50)
	read_SOLIS = SOLISreader_dbl_exc(max_lat=60)
	
	cr_list_1 = list(range(2142, 2152)) #Near the maximum of cycle 24
	cr_list_2 = list(range(2187, 2197)) #Between the peaks of cycles 24 and 25
	cr_list_3 = list(range(2197, 2207)) #Between the peaks of cycles 24 and 25 # NOTE: no SOLIS magnetograms in this interval
	
	def filt_for_SOLIS(crs):
		"""
		Given a list of Carrington rotation numbers, return the subset for which the SOLIS magnetograms are good
		"""
		return [cr for cr in crs if cr not in cr_SOLIS_bad]
	
	#Plot HMI energy spectra from different Carrington rotations
	res_HMI_1 = E0H1_dbl(cr_list_1, read_HMI)
	res_HMI_2 = E0H1_dbl(cr_list_2, read_HMI)
	res_HMI_3 = E0H1_dbl(cr_list_3, read_HMI)
	
	fig, ax = plt.subplots()
	
	for res, cr_list, ls in [
		(res_HMI_1, cr_list_1, '-'),
		(res_HMI_2, cr_list_2, '-'),
		(res_HMI_3, cr_list_3, '--'),
		]:
		errorfill(ax, res.k[1:], res.E0[1:], res.E0_err[1:], marker='', label=f"{min(cr_list)}–{max(cr_list)}", ls=ls)
	
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.legend(title="Carrington rotation")
	ax.margins(x=0)
	
	ax.set_xlabel("$k$ (Mm$^{-1}$)")
	ax.set_ylabel(r"$\widetilde{E}(k,0)$ (erg cm$^{-3}$)")
	
	ylim = ax.get_ylim()
	ax.set_ylim(ylim[0]/(ylim[1]/ylim[0])**0.5, ylim[1])
	
	save(fig, "HMI_energy_spectra.pdf")
	
	#Compare unapodized, apodized, and masked HMI spectra
	fig, ax = plt.subplots()
	
	res_HMIapod_3 = E0H1_dbl(cr_list_3, read_HMIapod)
	res_HMImask200_3 = E0H1_dbl(cr_list_3, read_HMImask200)
	res_HMImask50_3 = E0H1_dbl(cr_list_3, read_HMImask50)
	
	for res, label, ls in [
		(res_HMI_3, "full", '-'),
		(res_HMIapod_3, r"$\left| \lambda \right| < 60^\circ$", '-'),
		(res_HMImask200_3, r"$\left| \vec{B} \right| > 200$ G", ':'),
		(res_HMImask50_3, r"$\left| \vec{B} \right| > 50$ G", '--'),
		]:
		errorfill(ax, res.k[1:], res.E0[1:], res.E0_err[1:], marker='', label=label, ls=ls)
	
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.margins(x=0)
	ax.legend()
	
	ax.set_xlabel("$k$ (Mm$^{-1}$)")
	ax.set_ylabel(r"$\widetilde{E}(k,0)$ (erg cm$^{-3}$)")
	
	save(fig, f"HMI_apodization_masking_effect_cr_{min(cr_list_3)}-{max(cr_list_3)}.pdf")
	
	#Compare helicity spectra from apodized HMI and SOLIS data
	#NOTE that some SOLIS magnetograms are excluded as in Singh 2018
	res_SOLIS_1 = E0H1_dbl(filt_for_SOLIS(cr_list_1), read_SOLIS)
	res_HMIapod_1 = E0H1_dbl(cr_list_1, read_HMIapod)
	
	fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
	H_label = r"$-\mathrm{Im}(k\,\widetilde{H}(k,K_1))$"
	
	for res, ax, title in [
		(res_HMIapod_1, axs[0], "HMI"),
		(res_SOLIS_1, axs[1], "SOLIS"),
		]:
		
		handles_H = signed_loglog_plot(
			res.k[1:],
			(res.k*res.nimH1)[1:],
			err = (res.k*res.nimH1_err)[1:],
			label = "abs",
			ax = ax,
			)
		
		ef = errorfill(
			ax,
			res.k[1:],
			res.E0[1:],
			res.E0_err[1:],
			marker='',
			label=r"$\widetilde{E}(k,0)$",
			)
		
		#Store for later
		ax.handles_E = [ef.line]
		ax.handles_H = handles_H
		
		ax.set_title(title)
	
	axs[0].set_ylabel(r"$\widetilde{E}(k,0)$ (erg cm$^{-3}$)")
	fig.supxlabel("$k$ (Mm$^{-1}$)", size='medium')
	axs[0].margins(x=0)
	
	leg_E = axs[-1].legend(handles=axs[-1].handles_E)
	
	ax = axs[0]
	h_abs, *h_pm = ax.handles_H
	[h_dum] = ax.plot([], [], linestyle='-', color='none', label=" ")
	handles = [h_abs, h_dum, *h_pm] #insert a blank entry to keep + and - in the same column when ncols=2
	leg_H = ax.legend(handles=handles, labels=[h.get_label() for h in handles], title=H_label, ncols=2)
	
	fig.set_size_inches(6.5,2.7)
	
	save(fig, f"compare_helspec_HMI_SOLIS_cr_{min(cr_list_1)}-{max(cr_list_1)}.pdf")
