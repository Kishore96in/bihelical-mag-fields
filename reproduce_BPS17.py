"""
Reproduce Figure 8a of Brandenburg et al 2017 (https://doi.org/10.3847/1538-4357/836/1/21)
"""

import numpy as np
import matplotlib.pyplot as plt

from spectrum import get_B_vec, calc_spec, signed_loglog_plot

if __name__ == "__main__":
	#Figure 2
	cr_list = ["2161", "2162", "2163"]
	
	#NOTE: Nishant sets this as (2*R_sun, 4*R_sun) in his IDL scripts (search for `scl=!pi/Rphys`). This was apparently inherited from Gordon Petrie.
	L = np.array([np.pi*700,2*np.pi*700])
	
	E0_list = []
	H1_list = []
	for cr in cr_list:
		B_vec = get_B_vec(f"images/hmi.b_synoptic_small.rebinned.{cr}")
		k, E0, _ = calc_spec(B_vec, K=np.array([0,0]), L=L)
		_, _, H1 = calc_spec(B_vec, K=np.array([0,1]), L=L)
		
		E0_list.append(E0)
		H1_list.append(H1)
	
	E0_list = np.array(E0_list)
	H1_list = np.array(H1_list)
	
	E0 = np.average(E0_list, axis=0)
	E0_err = np.sqrt(np.average((E0_list - E0)**2, axis=0))
	
	H1 = np.average(H1_list, axis=0)
	H1_err = np.sqrt(np.average((H1_list - H1)**2, axis=0))
	
	fig,axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2,1]})
	
	handles = signed_loglog_plot(k, -np.imag(k*H1), axs[0], {'label':"-np.imag(k*H(k,1))"})
	h = axs[0].loglog(k, E0, label="E(k,0)")
	handles.extend(h)
	axs[0].legend(handles=handles)
	
	axs[1].loglog(k, E0_err, label="err, E")
	axs[1].loglog(k, k*H1_err, label="err, kH")
	axs[1].set_ylabel("Error")
	axs[1].set_xlabel("k")
	axs[1].legend()
	
	fig.set_size_inches(6.4,6.4)
	fig.tight_layout()
	
	plt.show()
