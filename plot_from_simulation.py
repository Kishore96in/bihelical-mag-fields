import matplotlib.pyplot as plt
import pencil as pc
import numpy as np
import scipy.fft
import os
import warnings

from spectrum import calc_spec, signed_loglog_plot

savefig = True #whether to save plots
simpath = "simulation/1"
savedir = os.path.join(simpath, "plots") #Where to save plots
iter_list = None

def save(fig, name, **kwargs):
	if not savefig:
		return
	
	if not os.path.exists(savedir):
		#Create directory if it does not exist
		os.makedirs(savedir)
	elif not os.path.isdir(savedir):
		raise FileExistsError(f"Save location {savedir} exists but is not a directory.")
	
	try:
		import git
		repo = git.Repo(path = os.path.dirname(__file__), search_parent_directories = True)
		
		if "metadata" in kwargs.keys():
			raise ValueError("Git was found. Do not specify metadata manually.")
		
		id_str = f"{os.path.basename(__file__)} at git commit {repo.head.object.hexsha}"
		
		if name[-4:] == ".pdf":
			kwargs['metadata'] = {'Creator': id_str}
		elif name[-4:] == ".png":
			kwargs['metadata'] = {'Software': id_str}
		else:
			raise ValueError(f"Could not infer file type for name {name}")
	except Exception as e:
		warnings.warn(f"Git info will not be saved in the image. {repr(e)}")
	
	loc = os.path.join(savedir, name)
	loc_dir = os.path.dirname(loc)
	if not os.path.exists(loc_dir):
		os.makedirs(loc_dir)
	fig.savefig(loc, **kwargs)

sim = pc.sim.get(quiet=True, path=simpath)
av = pc.read.aver(quiet=True, iter_list=iter_list, datadir=sim.datadir, simdir=sim.path)
grid = pc.read.grid(trim=True, quiet=True, datadir=sim.datadir)

res = {}
varname = "var.h5"
var = pc.read.var(sim=sim, var_file=varname, trimall=True, quiet=True, magic='bb')
res[varname] = []
for iy in np.round(np.linspace(0,sim.dim.ny-1,7))[:-1].astype(int):
	"""
	Coordinate system:
	r,phi,mu in synoptic maps correspond to y,x,z in our simulation.
	"""
	Bvec = var.bb[[1,0,2],:,iy,:]
	Bvec = np.swapaxes(Bvec, 1,2)
	Bvec_fft = scipy.fft.fft2(Bvec, norm='forward', axes=(-2,-1))
	
	L = [grid.Lx, grid.Lz]
	k, E0, _ = calc_spec(Bvec_fft, K=np.array([0,0]), L=L)
	_, _, H1 = calc_spec(Bvec_fft, K=np.array([0,1]), L=L)
	
	res[varname].append({'k':k, 'E0': E0, 'H1': H1})

H1av = np.average(np.array([d['H1'] for d in res['var.h5']]), axis=0)
E0av = np.average(np.array([d['E0'] for d in res['var.h5']]), axis=0)
k = res['var.h5'][0]['k']

fig,axs = plt.subplots(ncols=2)

axs[0].plot(grid.z, av.xy.abmz[-1])
axs[0].set_xlabel(r"$z$")
axs[0].set_ylabel(r"$\left< \vec{A}\cdot\vec{B} \right>$")
axs[0].set_xlim(min(grid.z), max(grid.z))
axs[0].axhline(0, ls=':', c='k')

handles = []
handles.extend( signed_loglog_plot(k, k*(-np.imag(H1av)), axs[1], {'label':"-imag(k*H(k,1))"}) )
handles.extend( axs[1].loglog(k, E0av, label="E(k,0)") )
axs[1].legend(handles=handles)
axs[1].set_xlabel("k")

fig.set_size_inches(6.4,3)
fig.tight_layout()
save(fig, "check_helspec_calc.pdf")
