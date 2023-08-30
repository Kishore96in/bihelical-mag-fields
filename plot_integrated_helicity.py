"""
Similar to figure 3 of Singh et al 2018.
Here, we double the domain to apply the two-scale method.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import trapezoid

from spectrum import calc_spec, signed_loglog_plot
from read_FITS import get_B_vec_dbllat
from utils import downsample_half, rebin

if __name__ == "__main__":
	cr_list = np.arange(2097,2269)
	
	L = np.array([2*np.pi*700,2*np.pi*700]) #data will be doubled in the latitudinal direction.
	
	E0_list = []
	H1_list = []
	for cr in cr_list:
		B_vec = get_B_vec_dbllat(f"images/hmi.b_synoptic_small.rebinned.{cr}")
		k, E0, _ = calc_spec(B_vec, K=np.array([0,0]), L=L)
		_, _, H1 = calc_spec(B_vec, K=np.array([0,2]), L=L, shift_onesided=0)
		
		E0_list.append(E0)
		H1_list.append(H1)
	
	E0_list = np.array(E0_list)
	H1_list = np.array(H1_list)
	
	_, H1_list = downsample_half(k, H1_list, axis=1)
	k, E0_list = downsample_half(k, E0_list, axis=1)
	
	Eint_list = trapezoid(E0_list, k, axis=1)
	nimHint_list = -np.imag(trapezoid(H1_list, k, axis=1))
	l_list = (np.pi/2) * trapezoid(E0_list[:,1:]/k[1:], k[1:], axis=1)/Eint_list
	r_list = nimHint_list/(2*l_list*Eint_list)
	
	fig = plt.figure()
	gs = mpl.gridspec.GridSpec(2,1, height_ratios=[1,4])
	gs1 = mpl.gridspec.GridSpecFromSubplotSpec(4,1, subplot_spec=gs[1], hspace=0)
	ax0 = fig.add_subplot(gs[0])
	axl = fig.add_subplot(gs1[3])
	
	axs = [
		ax0,
		fig.add_subplot(gs1[0], sharex=axl),
		fig.add_subplot(gs1[1], sharex=axl),
		fig.add_subplot(gs1[2], sharex=axl),
		axl,
		]
	
	assert nimHint_list.ndim == 1
	axs[0].hist(nimHint_list, bins=100)
	axs[0].set_xlabel(r"$- im(\mathcal{H}_M)$")
	
	handles = signed_loglog_plot(cr_list, nimHint_list, axs[1])
	axs[1].legend(handles=handles)
	axs[1].set_ylabel(r"$- im(\mathcal{H}_M)$")
	axs[1].set_xscale('linear')
	
	axs[2].semilogy(cr_list, Eint_list)
	axs[2].set_ylabel(r"$\mathcal{E}_M$")
	
	axs[3].plot(cr_list, l_list)
	axs[3].set_ylabel(r"$\mathcal{l}_M$")
	
	axs[4].plot(cr_list, np.abs(r_list))
	axs[4].set_ylabel(r"$\left| r_M \right|$")
	
	f = mpl.ticker.ScalarFormatter()
	f.set_scientific(False)
	axs[4].xaxis.set_major_formatter(f)
	
	for ax in axs[1:]:
		ax.label_outer()
		ax.tick_params(direction='inout', axis='x', which='major', top=True, bottom=True)
	
	fig.set_size_inches(6.4,8.4)
	fig.tight_layout()
	
	#TODO: Need to see how exactly Singh 2018 chose the scales at which to plot
	#TODO: is there a good heuristic to figure out if (and at which scale) a (noisy) helicity spectrum switches sign?
	#Similar to figure 4 of Singh et al 2018.
	bin_boundaries = np.array([0, 0.01, 0.1, 0.5])
	E0_rb = rebin(k, E0_list, bin_boundaries, axis=-1)
	nimkH1_rb = rebin(k, -np.imag(k*H1_list), bin_boundaries, axis=-1)
	
	fig,axs = plt.subplots(3, sharex=True, sharey=True)
	
	for i in range(len(bin_boundaries)-1):
		#TODO: Below, should I multiply by the bin width to account for the normalization?
		handles = signed_loglog_plot(cr_list, nimkH1_rb[:,i], axs[i])
		handles.extend(axs[i].semilogy(cr_list, E0_rb[:,i]))
		
		axs[i].set_title(rf"${bin_boundaries[i]} \leq k < {bin_boundaries[i+1]}$")
		axs[i].legend(handles=handles)
		# axs[i].set_ylabel(r"$\int E(k,0),\, -im(\int k H(k,1))$") # TODO: Figure out a nice way to denote what I mean
		axs[i].set_yscale('log')
		axs[i].set_xscale('linear')
	
	f = mpl.ticker.ScalarFormatter()
	f.set_scientific(False)
	axs[-1].xaxis.set_major_formatter(f)
	axs[-1].set_xlabel("CR")
	
	fig.set_size_inches(6.4,8.4)
	fig.tight_layout()
	
	plt.show()
