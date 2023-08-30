"""
Test the two-scale method on a simulation where the sign of helicity is the same throughout the domain. Here, we extract the K=0 mode.
"""

import matplotlib.pyplot as plt
import pencil as pc
import numpy as np
import scipy.fft
import os

from spectrum import calc_spec, signed_loglog_plot
from utils import fig_saver

savefig = True #whether to save plots
simpath = "simulation/2"
savedir = os.path.join(simpath, "plots") #Where to save plots
iter_list = None

save = fig_saver(savefig, savedir)

sim = pc.sim.get(quiet=True, path=simpath)
av = pc.read.aver(quiet=True, iter_list=iter_list, datadir=sim.datadir, simdir=sim.path)
grid = pc.read.grid(trim=True, quiet=True, datadir=sim.datadir)

res = {}
varname = "var.h5"
var = pc.read.var(sim=sim, var_file=varname, trimall=True, quiet=True, magic='bb')
res[varname] = []
for ix in np.round(np.linspace(0,sim.dim.nx-1,7))[:-1].astype(int):
	Bvec = var.bb[:,:,:,ix]
	Bvec = np.swapaxes(Bvec, 1,2)
	Bvec_fft = scipy.fft.fft2(Bvec, norm='forward', axes=(-2,-1))
	
	L = [grid.Ly, grid.Lz]
	k, E0, H0 = calc_spec(Bvec_fft, K=np.array([0,0]), L=L)
	
	res[varname].append({'k':k, 'E0': E0, 'H0': H0})

H0av = np.average(np.array([d['H0'] for d in res['var.h5']]), axis=0)
E0av = np.average(np.array([d['E0'] for d in res['var.h5']]), axis=0)
k = res['var.h5'][0]['k']

fig,axs = plt.subplots(ncols=2)

axs[0].plot(grid.z, av.xy.abmz[-1])
axs[0].set_xlabel(r"$z$")
axs[0].set_ylabel(r"$\left< \vec{A}\cdot\vec{B} \right>_{xy}$")
axs[0].set_xlim(min(grid.z), max(grid.z))
axs[0].axhline(0, ls=':', c='k')

handles = []
handles.extend( signed_loglog_plot(k, k*np.real(H0av), axs[1], {'label':"real(k*H(k,0))"}) ) #for the K=0 mode, we want the real part
handles.extend( axs[1].loglog(k, E0av, label="E(k,0)") )
axs[1].legend(handles=handles)
axs[1].set_xlabel("k")

fig.set_size_inches(6.4,3)
fig.tight_layout()
save(fig, "check_helspec_calc.pdf")
