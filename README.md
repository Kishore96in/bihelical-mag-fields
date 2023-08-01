# Scripts
* `remesh_sinlat_to_lat.py`: remesh `sine-latitude*longitude` maps (downloaded from JSOC) to `latitude*longitude`.
* `read_*.py`: functions to read FITS and IDL sav files and get the vector magnetic field
* `reproduce_singh18_SOLIS.py`: try to reproduce figure 2 of [Singh 18] using SOLIS data from Nishant's IDL sav files.

# Downloading and plotting data
## Synoptic map

Can use either `hmi.b_synoptic` (20MB per image) or `hmi.b_synoptic_small` (1MB per image).
Prime key is `CAR_ROT` for both.
Available segments are `Br, Bt, Bp, epts`

