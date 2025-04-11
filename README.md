# Scripts

## Plots for the ApJ paper
* `simulation/plot_apj.py`: figures 1,2,3
* `apj_plot_scripts/summary.py`: figures 4,5,7,8
* `lowk_components_HMI_vs_SOLIS.py`: figure 6
* `apj_plot_scripts/correlation_HMI_SOLIS.py`: figure 9
* `apj_plot_scripts/effect_azimuth.py`: figure 10

## Utilities
* `remesh_sinlat_to_lat.py`: remesh `sine-latitude*longitude` maps (downloaded from JSOC) to `latitude*longitude`.
* `read_*.py`: functions to read FITS and IDL sav files and get the vector magnetic field

## Other plots
* `reproduce_singh18_SOLIS.py`: try to reproduce figure 2 of [Singh 18] using SOLIS data from Nishant's IDL sav files.
* `simulation/plot_from_simulation*.py`: apply the two-scale technique to simulations (present in subdirectories of `simulation/`, and see if it is consistent with the sign of the helicity calculated directly from xy averages (similar to figure 6 of Brandenburg et al., 2017).

# Downloading and plotting data

## HMI synoptic maps
### Downloading
Can use either `hmi.b_synoptic` (20MB per image) or `hmi.b_synoptic_small` (1MB per image).
Prime key is `CAR_ROT` for both.
Available segments are `Br, Bt, Bp, epts`

To download required FITS files into the current directory:
```python
import drms
client = drms.Client(email=REGISTERED_EMAIL_HERE, verbose=True)
q = client.export('hmi.b_synoptic_small[2093-2268]{Br,Bt,Bp}', protocol='fits')
q.download(".")
```

### Malformed FITS headers
FITS headers in the files are malformed, so cannot use `astropy.wcs.WCS` directly.
Need to do something like
```python
from astropy.io import fits
from astropy.wcs import WCS

f = fits.open("hmi.b_synoptic.2267.Br.fits")
f[0].header['CUNIT2'] = "" #sine(latitude) is dimensionless

w = WCS(f[0].header)
```

Handcoded one (to get the coordinate arrays) is like
```pseudocode
h = f[0].header
coord = lambda i: (i+1-h['CRPIX?'])*h['CDELT?'] + h['CRVAL?']
```

## SOLIS synoptic maps
Data download links are provided at <https://solis.nso.edu/0/solis_data.html>. Synoptic vector magnetograms only seem to be available through the "Alternative Interface".
Naming scheme for files is described in <https://solis.nso.edu/pubkeep/DATAINFO_VSM.txt> (saved in `images_SOLIS/`).
If we are interested in Carrington rotations 2177–2186, we need to look at the date range 2016-05-10 to 2017-02-06.
Links to the maps we require seem to be directly available at <https://magmap.nso.edu/solis/v9g-int-maj_dim-180_cmp-phi-kc.html>.

As per <https://solis.nso.edu/pubkeep/DATAINFO_VSM.txt>, we need `kcv9g*t*_int-mas_dim-180.fits.gz`.
These maps are in coordinates latitude vs longitude.
An example of such a file is <https://magmap.nso.edu/solis//SV/v9g/201710/kcv9g171023/kcv9g171023t1526c2196_000_int-mas_dim-180.fits.gz>.
A single file contains all three components of the magnetic field (see `hdu.header['IMTYPE{1,2,3}']`.
`hdu.data` is a 4x180x360 array, with `[[0,1,2],:,:]` being r,theta,phi components.

# Miscellany
## Date for a given Carrington rotation
Sunpy provides the function `sunpy.coordinates.sun.carrington_rotation_time`.

## West and east
On the Sun, 'west' and 'east' are swapped as compared to the ones on the Earth (see <https://astronomy.stackexchange.com/questions/2203/how-are-east-and-west-defined-on-other-bodies-of-our-solar-system/2207#2207>).
This is why many sources (e.g. the headers of the SOLIS synoptic maps) say that the coordinate $\phi$ increases in the westward direction (which is the prograde direction).
$r, \theta, \phi$ thus form a right-handed coordinate system, as expected.
