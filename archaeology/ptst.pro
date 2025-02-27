;br=+reform(bslice[*,*,2])
;bt=-reform(bslice[*,*,1])
;bp=+reform(bslice[*,*,0])
;
nKK=2
nk=288
nslice=6
tEk_all=complexarr(nk,nslice,nKK)
kHk_all=complexarr(nk,nslice,nKK)
for i=0,nslice-1 do begin
  restore,'bslice'+str(i)+'.sav'
  br=+reform(bslice[*,*,0])
  bt=-reform(bslice[*,*,2])
  bp=+reform(bslice[*,*,1])
  ;
  for KKmax=0,1 do begin
    helspec_two,br,bt,bp,tEk,kHk,k,k1,KKmax,/single
    tEk_all[*,i,KKmax]=tEk
    kHk_all[*,i,KKmax]=kHk
    plot,k,k*float(kHk),xr=[0,20],yr=[-1,1]*3.5e-3
    oplot,k,+k*imaginary(kHk),col=122
    wait,.5
  endfor
endfor
;
save,file='slice_spectra.sav',k,nslice,nk,tEk_all,kHk_all
END
