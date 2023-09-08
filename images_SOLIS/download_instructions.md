# Commands to download synoptic maps.

```bash
#Recursively list all files in the directory https://magmap.nso.edu/solis/SV/v9g . We do this because the integrated synoptic maps seem to sit in unpredictable subdirectories. This takes a lot of time.
lftp -c find --ls https://magmap.nso.edu/solis/SV/v9g > magmap_v9g_file_list.txt

#Get only the names of the required fits.gz files from the recursive filelist generated above
grep -e 'kcv9g.*_int-mas_dim-180.fits.gz$' magmap_v9g_file_list.txt |\
	tr -s ' ' |\
	cut -d ' ' -f 5 > magmap_v9g_files_to_download.txt

wget --wait 1 --random-wait -i magmap_v9g_files_to_download.txt
gunzip -k *.fits.gz
```
