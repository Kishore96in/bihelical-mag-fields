import numpy as np
import matplotlib.pyplot as plt

from spectrum import calc_spec, signed_loglog_plot
from read_FITS import get_B_vec
from mask_weak import get_B_vec as get_B_vec_masked

if __name__ == "__main__":
	cr = "2148"
	threshold=150 #If the magnetic field magnitude is below this value, set it to zero.
	
	L = np.array([2*np.pi*700,np.pi*700])
	
	for cr in ["2148", "2180"]:
		B_vec = get_B_vec(f"images/hmi.b_synoptic_small.rebinned.{cr}")
		B_vec_masked = get_B_vec_masked(f"images/hmi.b_synoptic_small.rebinned.{cr}", threshold=threshold)
		
		k, E0, _ = calc_spec(B_vec, K=np.array([0,0]), L=L)
		_, _, H1 = calc_spec(B_vec, K=np.array([0,1]), L=L)
		_, E0m, _ = calc_spec(B_vec_masked, K=np.array([0,0]), L=L)
		_, _, H1m = calc_spec(B_vec_masked, K=np.array([0,1]), L=L)
		
		nimH1 = -np.imag(H1)
		nimH1m = -np.imag(H1m)
		
		fig,axs = plt.subplots(3, 1, sharex=True)
		
		handles = []
		handles.extend( signed_loglog_plot(k, k*nimH1, axs[0], {'label':"-imag(k*H(k,1))"}) )
		handles.extend( axs[0].loglog(k, E0, label="E(k,0)") )
		axs[0].legend(handles=handles)
		axs[0].set_title("Unmasked")
		
		handles = []
		handles.extend( signed_loglog_plot(k, k*nimH1m, axs[1], {'label':"-imag(k*H(k,1))"}) )
		handles.extend( axs[1].loglog(k, E0m, label="E(k,0)") )
		axs[1].legend(handles=handles)
		axs[1].set_title("Masked")
		
		axs[2].loglog(k, np.abs(nimH1m/nimH1), label="-imag(H), ")
		axs[2].loglog(k, np.abs(E0m/E0), label="E")
		axs[2].axhline(1, ls=':', c='k')
		axs[2].legend()
		axs[2].set_ylabel("|masked/unmasked|")
		
		axs[2].set_xlabel("k")
		
		fig.suptitle(f"Carrington rotation {cr}")
		fig.set_size_inches(6.4,9.6)
		fig.tight_layout()
	
	plt.show()
