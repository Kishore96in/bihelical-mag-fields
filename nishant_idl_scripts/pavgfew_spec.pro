if !d.name eq 'PS' then begin
  device,filename='avgfew_spec.ps',xsize=18,ysize=18.,yoffset=3
  !p.charthick=3 & !p.thick=3 & !x.thick=3 & !y.thick=3
end
;
; mv avgfew_spec.ps ~/tex/nishant/twoscl_helspec_Sun/fig/
;
!p.charsize=1.7
!x.margin=[10.8,.5]
!y.margin=[3.2,.5]
tilde='!9!s!aA!n!r!6'
siz=2.0
si2=1.5
;
Rphys=700. ;(Mm)
scl=!pi/Rphys
;
restore,'all_data_analyzed.sav'
;        k,k1,CR,alltEk0,alltEk1,allkHk0,allkHk1
good=where(CR ge 2148 and CR le 2151)
ngd=n_elements(good)
gdtEk0=float(alltEk0[*,good])
gdkHk1=-imaginary(allkHk1[*,good])
gdtEk0=smooth(gdtEk0,[0,2],/edge_mirror)
gdkHk1=smooth(gdkHk1,[0,2],/edge_mirror)
;
; mean over time-dim ... as fn of k
tEk0mk=total(gdtEk0,2)/ngd
kHk1mk=total(gdkHk1,2)/ngd
;
; ERROR ESTIMATION - RMS
;
nk=n_elements(k)
etEk0=make_array(nk,ngd,/float,value=0.)
ekHk1=make_array(nk,ngd,/float,value=0.)
for i=0,ngd-1 do begin
 etEk0[*,i]=gdtEk0[*,i]-tEk0mk
 ekHk1[*,i]=gdkHk1[*,i]-kHk1mk
 ;
 ;ekHk1[*,i]=abs(gdkHk1[*,i])-abs(kHk1mk)
 ;ekHk1[*,i]=abs(gdkHk1[*,i]-kHk1mk)
endfor
eetEk0=sqrt(total(etEk0^2.,2)/ngd)
eekHk1=sqrt(total(ekHk1^2.,2)/ngd)
;
;eekHk1=total(ekHk1,2)/ngd
;=======================
; Golden ratio (~1.618) for heights of different panels
; htot = hbot + gr*hbot
;
gr=(1.+sqrt(5))/2.
hgap=0.06 & filler=0.08+hgap+0.01
htot=1.-filler
hbot=htot/(1.+gr) & htop=gr*hbot
;
hh1=0.08+hbot & hh2=hh1+hgap & hh3=hh2+htop
;=======================
;
!x.title='!6'
!x.title='!8k!6 [Mm!u-1!n]'
!y.title='!62'+tilde+'!8E!6!dM!n and !9!!!6Im!8k'+tilde+'!8H!6!dM!n!9!!!6 [G!u2!nMm]'
;
xr=[3e-3,1.2]
xr=[3e-3,0.8]
yrE=[8e-1,2e4]
; for -5/3
xx1=.1 & xx2=0.4 & xl1=.22 & yl1=5e3 & fE=2.6e2
; for -8/3
xx3=.2 & xx4=0.45 & xl2=.3 & yl2=8e1 & fH=2e0
; for 3/2
xx5=.009 & xx6=0.035 & xl3=.016 & yl3=7e2 & fk2=2e5
;
;
plot_oo,scl*k,tEk0mk/scl,xr=xr,yr=yrE,pos=[0.19,hh2,0.99,hh3],xtit='',thick=4
oplot,scl*k,abs(kHk1mk)/scl,li=2,thick=4
;oplot,scl*k,abs(kHk1mk)/(k*scl*scl),li=2,thick=4
circ_sym,1.4,1 & oplot,scl*k,-kHk1mk/scl,col=55,ps=8
circ_sym,1.2,0 & oplot,scl*k,+kHk1mk/scl,col=122,ps=8
;
;errplot,scl*k,(abs(kHk1mk)-eekHk1)/scl,(abs(kHk1mk)+eekHk1)/scl,li=2
;
xp1=0.005 & yp1=6e0 & yn2=2e0
circ_sym,1.3,0 & legend,xp1,0.,yp1,0,ps=8,col=122,'  pos.'
circ_sym,1.3,1 & legend,xp1,0.,yn2,0,ps=8,col=55,'  neg.'
xl4=.017 & yl4=8.
ll4='!13<!6-Im!8k'+tilde+'!8H!6!dM!n(!8K!6!d0!n,!8k!6)!13>!6!dCR!n'
xyouts,siz=si2,xl4,yl4,ll4
xl5=.0043 & yl5=5e3
xyouts,siz=si2,xl5,yl5,'!13<!62'+tilde+'!8E!6!dM!n(0,!8k!6)!13>!6!dCR!n'
xl6=.023 & yl6=1.5
xyouts,siz=si2,xl6,yl6,'!6Average Spectra'
;
xx=[xx1,xx2] & oplot,xx,fE/xx^1.667
xyouts,xl1,yl1,'!8k!6!u-5/3!n'
xx=[xx3,xx4] & oplot,xx,fH/xx^2.667
xyouts,xl2,yl2,'!8k!6!u-8/3!n'
xx=[xx5,xx6] & oplot,xx,fk2*xx^(3./2.)
xyouts,xl3,yl3,'!8k!6!u3/2!n'
;
;=======
; the error plot
;
!y.title='!6Error [G!u2!nMm]'
yr2=[4e-1,5e3]
plot_oo,scl*k,eetEk0/scl,xr=xr,yr=yr2,thick=4,$
 pos=[0.19,0.12,0.99,hh1],/noerase
oplot,scl*k,eekHk1/scl,li=2,thick=4
;
xyouts,0.005,100,'!7r!6!d2!8E!n!6'
xyouts,0.02,8,'!7r!8!dkH!n!6'
;
xx=[.09,.45] & oplot,xx,2.8*xx^(-5./3.)
xyouts,.18,52,'!8k!6!u-5/3!n',siz=si2
;
STOP
END
