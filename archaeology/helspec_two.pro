pro helspec_two,br,bt,bp,tEk,kHk,k,k1,KKmax,single=single,debug=debug,count=count,kx0_count=kx0_count,axisym=axisym
;
;  compute wavenumbers
;
ii=complex(0.,1.)
s=size(br) & nx=s[1] & ny=s[2]
n=nx*ny
nk=min([nx,ny]/2)
dkx=2*nk/float(nx)
dky=2*nk/float(ny)
print,'dkx,dky=',dkx,dky
;
;  shift, because we want equator in the middle
;
b1k=fft(+shift(bp,0,ny/2),-1)
b2k=fft(-shift(bt,0,ny/2),-1)
b3k=fft(+shift(br,0,ny/2),-1)
;
;  shift to corner
;
b1k=shift(b1k,nx/2,ny/2)
b2k=shift(b2k,nx/2,ny/2)
b3k=shift(b3k,nx/2,ny/2)
;
;  compute wavevector
;
kx=dkx*(findgen(nx)-nx/2.)
ky=dky*(findgen(ny)-ny/2.)
kx=reform(kx,nx,1)
ky=reform(ky,1,ny)
kx=rebin(kx,nx,ny)
ky=rebin(ky,nx,ny)
kr=sqrt(kx^2+ky^2)
;
;  unit vector, compute 1/kr
;
kr1=kr
orig=where(kr1 eq 0.)
kr1(orig)=1.
kr1=1./kr1
kr1(orig)=0.
;
kx1=kx*kr1
ky1=ky*kr1
;
nKK=KKmax+1
k=fltarr(nk)
count=fltarr(nk)
kx0_count=fltarr(nk)
;
if keyword_set(single) then begin
  KKstart=KKmax
  tEk=complexarr(nk)
  kHk=complexarr(nk)
endif else begin
  KKstart=0.
  tEk=complexarr(nk,nKK)
  kHk=complexarr(nk,nKK)
endelse
;
for KK=KKstart,KKmax do begin
  if KKmax ne 1 then print,'KK=',KK
  b1kp=shift(b1k,0,-KK)
  b2kp=shift(b2k,0,-KK)
  b3kp=shift(b3k,0,-KK)
  ;
  for ik=0,nk-1 do begin
    good=where((kr ge ik-.5) and (kr lt ik+.5))
    ;
    tmp1=(b1kp(good)*conj(b1k(good)))
    tmp2=(b2kp(good)*conj(b2k(good)))
    tmp3=(b3kp(good)*conj(b3k(good)))
    ;
    scr2=+ii*b2kp(good)*conj(b3k(good))*kx1(good) $
         -ii*b3kp(good)*conj(b2k(good))*kx1(good)
    scr3=+ii*b3kp(good)*conj(b1k(good))*ky1(good) $
         -ii*b1kp(good)*conj(b3k(good))*ky1(good)
    ;
    if keyword_set(single) then begin
      tEk(ik)=total(tmp1+tmp2+tmp3)
      kHk(ik)=total(scr2+scr3)
    endif else begin
      if keyword_set(axisym) then begin
        tmp=tmp1+tmp2+tmp3
        scr=scr2+scr3
        kx_zero=where(kx(good) eq 0.)
        tEk(ik,KK)=total(tmp(kx_zero))
        kHk(ik,KK)=total(scr(kx_zero))
      endif else begin
        tEk(ik,KK)=total(tmp1+tmp2+tmp3)
        kHk(ik,KK)=total(scr2+scr3)
      endelse
    endelse
    ;
    k(ik)=mean(kr(good))
    count(ik)=n_elements(good)
    kx0_count(ik)=n_elements(where(kx(good) eq 0.))
    if keyword_set(debug) then begin
      if ik ge 9 then stop
    endif
  endfor
endfor
;
k1=k
k1[0]=1.
k1=1./k1
k1[0]=0.
;
END
