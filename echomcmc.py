import matplotlib
matplotlib.use('Agg')
import math as m
import numpy as np
import pylab as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import scatter as sc
from myfig import *
from scipy import optimize as opt
from scipy import stats

import emcee
import sys

# disk_echo_export_deltas_only_snr/kplr009222909/kplr009222909_flare_composite.npy
usedir = "disk_echo_export_deltas_only_snr"

seed = 99952345
testing = False
usefake = False
mdiskfake = 0.1
ainfake = 30.0
aoutfake = 60.0
dosearch = False
noecho = False
kic = ''

_c_ = 2.99792458e10            # speed-o-light...
deg = np.pi/180.0
AU = 1.495978707e13 # 1 AU in cgs (units used here...)
micron = 1e-4
mm = 0.1
Mearth = 5.972e27

# fitting parameter limits...
#  global AinminAU, AinmaxAU, AoutmaxAU, MdmaxMearth
Tmin = 60 * 45 # don't use first two bins in Kepler lc data.
Tmax = 24 * 60 * 60 # num hours, default
MdmaxMearth = 10.0
MassExp = 0.35
MassPrior = 3 # 1 is uniform > 0, 2 is exp(-m/MassExp), 3 use tau=1 cut
MdminMearth = 0.0 if MassPrior else -MdmaxMearth

def resetlimits(tmin,tmax,massprior = 1): # will add more...
    global Tmin,Tmax,AinminAU,AinmaxAU,AoutmaxAU,MdmaxMearth,MdminMearth
    Tmax = tmax
    Tmin = tmin
    AinminAU = _c_*tmin/AU
    AoutmaxAU = 0.5*_c_*Tmax/AU # based on edge-on, 2x ltravel time to disk
    AinmaxAU = AoutmaxAU 
    MdminMearth = 0.0 if massprior else -MdmaxMearth
    print('reset',AoutmaxAU)

resetlimits(Tmin,Tmax)

rhop = 2.0
rmin = 1.0*micron
rmax = 1.0*mm
g = 0.429       # underlying assumption about dust. not fitting this...
nu = 0.114      # Draine 03 parameterization.
hscale = 0.05   # scale height param for disk

n = 240       # underlying cyindrical grid dim, nxn approx.
NWalkers = 30

#fin = 'kplr009222909_flare_composite.npy'
fin = ''
faadd = ''

argv = [sys.argv[0]]
i = 1
argc = len(sys.argv)
while i<argc:
    ar = sys.argv[i]
    if ar[0] == '-':
        if ar == '-fit':
            dosearch = True
        elif ar == '-test':
            testing = True
        elif ar == '-fake':
            usefake = True
        elif ar == '-aoutmax' and i+1<argc:
            AoutmaxAU = float(sys.argv[i+1])
            i += 1
        elif ar == '-massprior' and i+1<argc:
            MassPrior = int(sys.argv[i+1])
            i += 1
        elif ar == '-noecho' or ar == '-no_echo':
            noecho = True
        elif ar == '-kic' and i+1<argc:
            i += 1
            kic = sys.argv[i]
        elif ar == '-ngrid' and i+1<argc:
            i += 1
            n = int(sys.argv[i])
        elif ar == '-nwalkers' and i+1<argc:
            i += 1
            NWalkers = int(sys.argv[i])
        elif ar == '-seed' and i+1<argc:
            i += 1
            seed = int(sys.argv[i])
        else:
            print(' unrecognized option',ar,' or missing parameter.')
            quit()
    else:
        argv.append(sys.argv[i])
    i += 1

# be careful here...
resetlimits(AinminAU*AU/_c_,2*AoutmaxAU*AU/_c_,massprior=MassPrior)




if usefake:
    if kic != '':
        print('confused, both kic and usefake?')
        quit()
    kic = 'fake'
elif kic != '':
    if "kplr" != kic[:4]:
        print(' hmmmm, try kplr as a prefix to the KIC #, kplr009...')

if testing:
    usefake = True
    noecho = False
    kic = 'test'
    g = 0
    nu = 0

if len(argv)==2:
    if usefake:
        print(' usefake is set, yet there is an input file?')
        quit()
    elif kic != '':
        print(' warning: kic is set to',kic+', yet there is an input file.')
        print(' using kic as a label for output files.')
    fin = argv[1]
elif kic == '':
    print('usage:',argv[0],'[-fit -fake -no_echo -kic ###] kicfile')
    quit()

ofilbase="echomcmc"
if kic != '': ofilbase = ofilbase+'_'+kic
if noecho: ofilbase = ofilbase+'_noecho'
fnpy = ofilbase+'_psamp.npy'
ftxt = ofilbase+'_lc.txt'
fplt = ofilbase+'.pdf'

def genfdat(mdisk,ain,aout,inc,rhop,rmin,rmax,g,nu,n):
    global _c_ # really?
    relwid = (aout-ain)/aout
    relcirc = 2.0*np.pi;
    na = int(n*np.sqrt(relwid/relcirc))
    if na<3: na=3
    nh = n**2//na
    ntot = (na*nh)
    da, dh = (aout-ain)/na, 2*np.pi/nh
    aa = np.linspace(ain+da/2,aout-da/2,na)
    hh = np.linspace(-np.pi+dh/2,np.pi-dh/2,nh)
    rmap,hmap = np.meshgrid(aa,hh)
    rmap = rmap.flatten()
    hmap = hmap.flatten()
    dA = dh*rmap*da
    tmap = np.zeros(ntot)
    bmap = np.ones(ntot)*1e-45;
    # working in an image plane inclined to disk... 
    x = rmap*np.cos(hmap)
    y = rmap*np.sin(hmap)
    z = 0.
    tmap = (rmap - np.sin(inc)*x)/_c_
    xo = np.sin(inc); yo = 0; zo = np.cos(inc) # defines observer dir
    scatang = np.arccos((xo*x+yo*y+zo*z)/rmap)
    # bmap = (rmap/rin)**(-2-gamma)*sc.phase_function_HG(scatang,g)
    # note: bmap expects q=3.5 dust model....
    X = np.sqrt(rmax/rmin)
    C = mdisk * 3 / (8*np.pi*rhop*rmin*X*(aout-ain))
    bmap = C * dA / rmap**3 * sc.phase_function_D03(scatang,g,nu)
    return rmap,hmap,tmap,bmap

def maskring(rmap,rin,rout,tmap,tmin,tmax):
    msk = (rmap >= rin) & (rmap < rout) & (tmap >= tmin) & (tmap < tmax)
    return msk;  

def maskringang(rmap,rin,rout,hmap,hmin,hmax,tmap,tmin,tmax):
    msk = (rmap >= rin) & (rmap < rout) & (tmap >= tmin) & (tmap < tmax)
    if ((hmin >= hmax) or (hmax-hmin >= 2*np.pi)): return msk
    hminuse = hmin; hmaxuse = hmax
    flag = 0
    if (hmin <= -np.pi):
        hminuse += 2*np.pi
        flag += 1
    if (hmax >= np.pi):
        hmaxuse -= 2*np.pi
        flag += 1
    if (flag==1):        
        msk = msk & ((hmap < hmaxuse) | (hmap >= hminuse))
    elif (flag==0):
        msk = msk & ((hmap >= hminuse) & (hmap < hmaxuse))
    return msk;  

def genlc(ta,ain,aout,hmin,hmax,rmap,hmap,tmap,bmap,phiweight=False):
    # generates a lightcurve normalized so it sums to one.
    # returns this, and the sum of flux bins prior to normalization.
    tbeg, dt, nt = ta[0], ta[1]-ta[0], len(ta)
    tend = ta[-1]+dt
    tbins = np.append(ta,[tend])
    msk = maskringang(rmap,ain,aout,hmap,hmin,hmax,tmap,tbeg,tend)
    wei = bmap[msk]
    if phiweight == True:
        wei *= 1-np.abs(hmap[msk])/np.pi    
    fa,_ = np.histogram(tmap[msk],bins=tbins,weights=wei)
    return fa

def genmodel(t,parms,rhop,rmin,rmax,g,nu,n): 
    # runs a full model....careful w/units!!
    md = parms[0]*Mearth
    ain = parms[1]*AU
    aout = ain*parms[2]
    inc = parms[3]
    rm, hm, tm, bm = genfdat(md,ain,aout,inc,rhop,rmin,rmax,g,nu,n)
    ftry = genlc(t,ain,aout,-np.pi,np.pi,rm,hm,tm,bm,phiweight=True)
    return ftry

def mthresh(ain,aout,rhop,rmin,rmax,h): # cgs
    X = np.sqrt(rmax/rmin)
    return 32*np.pi/3*h*rhop*X*rmin*ain*aout

def chi2ff(parms,t,f,s,rhop,rmin,rmax,g,nu,n): # chi2 = |(amp*fmodel-f)|^2
    global AinminAU, AinmaxAU, AoutmaxAU, MdminMearth, MdmaxMearth
    md = parms[0]
    ain = parms[1]
    aout = ain*parms[2]
    inc = parms[3]
    if ain <AinminAU or ain>AinmaxAU: return 1e99  # ain
    if aout<1.02*ain or aout>AoutmaxAU: return 1e99       # aout/ain
    if inc<0. or inc>np.pi/2: return 1e99  # inc
    # if md<0 or md>MdmaxMearth: return 1e99 
    if (md>MdmaxMearth) or (md<MdminMearth): return 1e99 
    ftry = genmodel(t,parms,rhop,rmin,rmax,g,nu,n)
    chi2 = np.sum(((ftry-f)/s)**2)
    #print(f'chi2: {f}, reduced: {f}'.format(chi2,chi2/(len(t)-len(parms)))
    return chi2

def genfake(mdisk,ain,aout,inc,rhop,rmin,rmax,g,nu,n):
    global Tmin, Tmax
    Tmax = 24*60*60
    # AinmaxAU = c_*Tmax/AU) # ? 
    dt = 30.*60
    Tmin = 0.5*dt
    resetlimits(Tmin,Tmax)
    rm, hm, tm, bm = genfdat(mdisk,ain,aout,inc,rhop,rmin,rmax,g,nu,n)
    nflares = 30
    ta = np.arange(-dt,Tmax,dt)
    Tmin = 0.5*dt # use first non-flare bin in fake data
    nt = len(ta)
    fall = np.zeros(nt)    
    if True:
        for i in range(nflares):
            rndang = np.random.uniform(-np.pi/2,np.pi/2)
            hmin = rndang-0.5*np.pi; hmax = rndang+0.5*np.pi
            fa = genlc(ta,ain,aout,hmin,hmax,rm,hm,tm,bm)
            fall += fa
        fall /= nflares # work with average
    else:
        fall = genlc(ta,ain,aout,-np.pi,np.pi,rm,hm,tm,bm,phiweight=True)
    print('in fake, refltot:',np.sum(fall))
    falltrue = np.copy(fall)    
    # errors add in quadrature.
    # sig1 = 0.75*max(falltrue)
    sig1 = 0.00075
    #print('sigma: ',sig1)
    sig = sig1/np.sqrt(nflares)
    noi = np.random.normal(0,sig,nt); noi[0] = 0
    fall += noi
    ferr = np.ones(len(fall))*sig
    return ta, fall, ferr, falltrue

def pvalue_noflare(t,f,s):
    dof = len(t)-1 # because we fit the mean and subtracted it off..
    chi2 = np.sum((f/s)**2)
    return stats.chi2(dof).sf(chi2)


np.random.seed(seed)

if usefake:
    md,ain,aout,inc = mdiskfake*Mearth,ainfake*AU,aoutfake*AU,20*deg
    ta,fall,sig,falltrue = genfake(md,ain,aout,inc,rhop,rmin,rmax,g,nu,n)
else:
    # disk_echo_export_deltas_only_snr/kplr009222909/kplr009222909_flare_composite.npy
    if kic != '' and fin == '':
        fin = usedir+'/'+kic+'/'+kic+'_flare_composite.npy'
    X = np.load(fin)
    ta,fall,sig = X[0],X[1],X[2]
    ta *= 60.0
    dt = (ta[-1]-ta[0])/(len(ta)-1)
    resetlimits(2.1*dt,ta[-1])

if noecho:
    xxx = np.random.uniform(3) # fuck with the RNG stream a little
    fall = np.random.randn(len(ta))*sig
    falltrue = np.zeros(len(fall))

if testing:
    print(' testing!!')
    print(' expect:',3*mdiskfake*Mearth/(32*np.pi*rhop*np.sqrt(rmax*rmin)*ainfake*AU*aoutfake*AU))
    print(" calc'd:",np.sum(falltrue))
    quit()

nt = len(ta)
dt = ta[1]-ta[0]
tmsk = ta>1.1*dt
tmsk = (ta>= Tmin) & (ta <= Tmax) 

pnada = pvalue_noflare(ta[tmsk],fall[tmsk],sig[tmsk])
if pnada<0.01: print(pnada,'pval no flare excess @ 99% conf!',fin)
else: print(pnada,'pval no flare.',fin)

if dosearch == False:
    quit()

xmsk = (ta > Tmin) & (ta < AoutmaxAU*AU/_c_)
zz = fall[xmsk]/sig[xmsk]
ii = np.argmax(zz)
aguess = ta[xmsk][ii]*_c_/AU
if aguess < 15: aguess = 15.
print('t peak:',ta[xmsk][ii]/60.0,' a guess:',aguess)

print('optimizing...')

mdiskguess = 0.05
guess = [mdiskguess,aguess,1.5,20*deg]
args = (ta[tmsk],fall[tmsk],sig[tmsk],rhop,rmin,rmax,g,nu,n)
res = opt.minimize(chi2ff,guess,args=args, method='Nelder-Mead')

#res = opt.minimize(chi2ff,[40,1.6,20*deg,10.0],args=(ta[tmsk],fall[tmsk],sig[tmsk],g,n), method='Nelder-Mead')

mdopt, ainopt, aratopt, incopt = res.x
aoutopt = ainopt * aratopt
print('done.',mdopt,'M_Earth,',ainopt,'AU,',aoutopt,'AU,',incopt/deg,'deg')
print(' mthresh: ',mthresh(ainopt*AU,aoutopt*AU,rhop,rmin,rmax,hscale)/Mearth)

mdopt *= Mearth; ainopt *= AU; aoutopt *= AU

print(' reduced chi2',res.fun/(np.sum(tmsk)-len(res.x)))

if False:
    plt.errorbar(ta[tmsk]/60,fall[tmsk],yerr=sig[tmsk],ls='none')
    plt.plot(ta[tmsk]/60,faopt[tmsk])
    ofil="echodetect.pdf"
    plt.savefig(ofil)
    www = "public_html"
    os.system("convert "+ofil+" $HOME/"+www+"/tmp.jpg")
    quit()

def lnprob(x,t,f,s,rho,rmin,rmax,g,nu,h,n):
    mass, ain, arat, inc = x # units: earth masses, AU, AU, radians
    aout = ain*arat
    if (inc<=0 or inc>np.pi/2): return -np.inf
    if (ain<AinminAU) or (ain>aout) or (aout>AoutmaxAU): return -np.inf
    lnprior = np.log(np.sin(inc))
    if (MassPrior):
        if (mass < 0.0): return -np.inf
        if (MassPrior == 2):
            lnprior -= mass/MassExp  # in Earth masses
        if (MassPrior == 3):
            X = np.sqrt(rmax/rmin)
            mth = mthresh(ain*AU,aout*AU,rhop,rmin,rmax,h)/Mearth
            lnprior  += -(mass/mth) + np.log(mth)
            #lnprior  += -0.5*(mass/mth)**2 + 0.5*np.log(2*np.pi*mth**2)
    c2 = chi2ff(x,t,f,s,rho,rmin,rmax,g,nu,n)
    if (c2>=1e99): return -np.inf
    return -0.5*(c2 + np.sum(np.log(2*np.pi*s**2))) + lnprior
    
# reseed so emcee starts fresh after optimizer....
np.random.seed(seed)

ndim = 4
nwalkers = NWalkers
bf = np.copy(res.x)
if bf[3] < 1e-3: bf[3] = 2*deg
p0 = [bf*(1+0.1*np.random.randn(ndim)) for i in range(nwalkers)]
p0[0] = res.x

mcmcargs = (ta[tmsk],fall[tmsk],sig[tmsk],rhop,rmin,rmax,g,nu,hscale,n)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=mcmcargs)

print('emcee burn-in...')
pos, prob, state = sampler.run_mcmc(p0, 15*ndim)
sampler.reset()
print('emcee run...')
sampler.run_mcmc(pos, ndim*500)
# lnprobsamp = sampler.get_log_prob(flat=True)
lnprobsamp = sampler.flatlnprobability
# psamp = sampler.get_chain(flat=True)
psamp = sampler.flatchain 
print('...and done.')
#tau = sampler.get_autocorr_time()
#print(tau)

msk = np.isfinite(lnprobsamp)
lnprobsamp = lnprobsamp[msk]
psamp = psamp[msk,:]

idx = np.argsort(lnprobsamp)
lnprobsamp = lnprobsamp[idx]
psamp = psamp[idx,:]
mdsamp = psamp[:,0]
ainsamp = psamp[:,1]
aratsamp = psamp[:,2]; aoutsamp = aratsamp*ainsamp
incsamp = psamp[:,3]

np.save(fnpy,np.array([mdsamp,ainsamp,aoutsamp,incsamp,lnprobsamp]))
# data = np.load(fnpy)
# print(data.shape)
# mm,aaii,aaoo,ii,llpp = data
# print(np.sum(np.abs(llpp-lnprobsamp)))
# quit()
     
ii = np.argmax(lnprobsamp)
if lnprobsamp[ii] > res.fun:
    mostprob = psamp[ii,:]
else:
    mostprob = res.x
mdmp, ainmp, aratmp,incmp = mostprob
aoutmp = ainmp*aratmp

chi2mpsamp = chi2ff(psamp[ii,:],*args)
print('compare bfs:',res.fun,chi2mpsamp,ii,len(lnprobsamp))

mdmin,mdmax = np.quantile(mdsamp,[0.025,0.975])
ainmin,ainmax = np.quantile(ainsamp,[0.025,0.975])
aoutmin,aoutmax = np.quantile(aoutsamp,[0.025,0.975])
incmin,incmax = np.quantile(incsamp,[0.025,0.975])

# print(f'mdisk {mdmp},  {mdmin}--{mdmax}')
print('mdisk %f, [%f, %f]' % (mdmp,mdmin,mdmax))
#print(f'ain {ainmp},  {ainmin}--{ainmax} AU')
print('ain %f, [%f, %f]' % (ainmp,ainmin,ainmax))
#print(f'aout {aoutmp},  {aoutmin}--{aoutmax} AU')
#print(f'inc {incmp/deg},  {incmin/deg}--{incmax/deg}')

if not MassPrior:
    mdx = mdsamp[mdsamp>=0]
    mdminx,mdmaxx = np.quantile(mdx,[0.025,0.975])
    print('mdisk %f, [%f, %f]' % (mdmp,mdminx,mdmaxx))

# assume delta fn flare....

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
if False:
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)
    ax4.set_box_aspect(1)

# plot the model vs data
taa = ta/3600
fac = 100

famp = genmodel(ta,mostprob,rhop,rmin,rmax,g,nu,n)
with open(ftxt,'w') as myf:
    myf.write('# time (sec) flux_obs sig_obs flux_model\n')
    myf.write('# pval noecho: %16.13g\n'%(pnada))
    myf.write('# models parm: %16.13g %16.13g %16.13g\n'%(mostprob[0],mostprob[1],mostprob[2]))
    for i in range(len(ta)):
        myf.write('%16.13g %16.13g %16.13g %16.13g\n'%(ta[i],fall[i],sig[i],famp[i]))
    myf.close()
ax1.errorbar(taa[tmsk],fac*fall[tmsk],yerr=fac*sig[tmsk],c='k',zorder=0,ls='none')
ax1.plot(taa[tmsk],fac*fall[tmsk],'ok',ms=5,mew=0,zorder=1)

ax1.plot(taa[tmsk],fac*famp[tmsk],'-r',zorder=999,lw=1)
if usefake:
    ax1.plot(taa[tmsk],fac*falltrue[tmsk],'-b',lw=2,zorder=99)
ax1.set(ylabel="relative flux (%)",xlabel="echo delay time (hours)")

if True:
    fc = '#000055'
    #fc = '#888888'
    cm = 'plasma'
    cm = 'inferno'
    # cm = 'YlOrBr'
    siz=3
    thin = 10
    x = lnprobsamp # np.exp(-lnprobsamp)
    lpo = (x-min(x))/(max(x)-min(x))
    lpo = lpo**2
    lpo = lpo[::10]
    ax2.set_facecolor(fc)
    ax2.scatter(ainsamp[::thin],incsamp[::thin]/deg,s=siz,c=lpo,cmap=cm)
    ax2.set(xlabel="inner disk radius (AU)",ylabel="inclination (degrees)")
    ax3.set_facecolor(fc)
    ax3.scatter(mdsamp[::thin],aoutsamp[::thin],s=siz,c=lpo,cmap=cm)
    ax3.set(xlabel=r"disk mass (M$_\oplus$)",ylabel="outer disk radius (AU)")
    ax4.set_facecolor(fc)
    ax4.scatter(ainsamp[::thin],aoutsamp[::thin],s=siz,c=lpo,cmap=cm)
    ax4.set(xlabel="inner disk radius (AU)",ylabel="outer disk radius (AU)")


if not 'fake' in kic:
    fig.suptitle(kic)

plt.tight_layout()

ofil="echomcmc"
if kic != '': ofil = ofil+'_'+kic
if noecho: ofil = ofil+'_noecho'
ofil = ofil+'.pdf'

plt.savefig(fplt)

os.system("convert "+fplt+" tmp.jpg")
os.system("cp tmp.jpg ~/www/")
quit()


