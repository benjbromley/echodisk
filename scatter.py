# routines for light scattering calculations, including a few
# parameterized phase functions and polarization fractions.

# phase_function_HG(Theta,g): returns phase function for scattering
# angle Theta (0=>forward scattering), based on
# Henyey-Greenstein. parameter g=-1..1 runs from full back-scattering
# (g= -1) to forward scattering (g=1), g=0 is isotropic.Normalized so
# that the integral over all solid angles is unity.

# phase_function_D03(Theta,g,nu): returns phase function for
# scattering angle Theta (0=>forward scattering), based on Draine
# 2003. Parameter nu=0 is HG, while 1 is Rayleigh.
# Normalization is so that integration over all solid angles is unity.

# phase_function_F19(Theta): returns phase function for scattering
# angle Theta (0=>forward scattering), based on Frattin et
# al. 2019. Normalized so that the integral over all solid angles is
# unity.  [int dOmega ... = int 2*pi*sin(theta)*dtheta...]
# Helper function is phase_function_F19_setup(). Estimated from Fig 4.
# in Frattin et al. in lab experiments w/cometary dust.

# polarization_frac(pmax,Theta): polarization fraction as a function
# of scattering angle Theta. This simple sin(Theta)**2/[1+cos(Theta)**2].
# Paramenter pmax gives the value of the peak polarization, e.g., 0.1=10%.

# polarization_frac_F19(pmax,Theta): polarization fraction as a
# function of scattering angle Theta. This is estimated from Frattin
# et al. 2019 plots.  Paramenter pmax gives the value of the peak
# polarization, pmax=0.1 corresponds to 10%.

# scatter_test() runs through a few examples of usage.

#
# bcb 2020-04-16
#

import math as m
import numpy as np
from scipy import integrate
from scipy import interpolate

# angular arguments are phase angles(!!) = pi - scattering_angles

def phase_function_F19_setup(verbose=False):
    # data loosely from Frattin et al. (2019)
    pf = np.array([0.2, 0.16,
                   0.13, 0.12, 0.115, 0.12,
                   0.135, 0.16, 0.2, 0.32,
                   0.5, 0.65, 1, 2.5, 5, 7.3, 10, 35])  # from fig 4.
    pa = np.array([  0, 3, 10, 20, 30, 50.,
                     75, 90,  106, 123,
                     135, 142, 150, 165, 172, 175, 177, 180]) # degrees
    sa = 180-pa # scattering angle
    m = 6
    sa *= np.pi/180.
    kind='cubic'
    pfit = interpolate.interp1d(sa,pf,kind=kind)
    def ppff(x): return 2*np.pi*np.sin(x)*pfit(x)
    # want integral_over_solid ang pfit = 1....
    norm = integrate.quad(ppff,0,np.pi)[0]
    pfit = interpolate.interp1d(sa,pf/norm,kind=kind)
    if verbose == True:
        print('phase function F19 fit with interp1d...')
    if False:
        import matplotlib as mpl
        mpl.use('Agg')
        import pylab as pl
        xx = np.linspace(sa[0],sa[-1],1000)
        pl.semilogy(180-180*xx/np.pi,norm*pfit(xx),'k-')
        pl.semilogy(180-180*sa/np.pi,pf,'bo')
        pl.savefig('tmp.png')
    return pfit

def polarization_frac_F19_setup(verbose=False):
    nsamp = 37
    phaseang = np.linspace(0.,180.,nsamp)
    # this is hacked up to make the fit look good.
    degpol = np.array([0,-1.3,-2.5,-3,-2.9,-2.5,-1.3,0.0,1.2,2.5,3.8,5.0,6.5,7.9,8.9,9.5,9.9,10.0,9.8,9.4,8.6,7.8,7.1,6.4,5.8,5.2,4.7,4.1,3.7,3.6,3.5,3.3,2.9,2.6,1.9,1.0,-0.0])
    # the above is units of degrees and percent...
    polfrac = 0.1*degpol # so pmax is 1
    phaseang *= np.pi/180.0
    sa = np.pi - phaseang
    pfit = interpolate.interp1d(sa,polfrac)
    if verbose:
        print('polarization degree fit using interp1d...')
    if False:
        import matplotlib as mpl
        mpl.use('Agg')
        import pylab as pl
        pl.plot(phaseang*180/np.pi,pfit(sa),'k-')
        pl.plot(phaseang*180/np.pi,polfrac,'bo')
        pl.savefig('tmp.png')
    return pfit

def phase_function_F19(Theta):
    if phase_function_F19.func == 0:
        phase_function_F19.func = phase_function_F19_setup()
    return phase_function_F19.func(Theta)
phase_function_F19.func = 0

def polarization_frac_F19(pmax,Theta):
    if polarization_frac_F19.func == 0:
        polarization_frac_F19.func = polarization_frac_F19_setup()
    return pmax*polarization_frac_F19.func(Theta)
polarization_frac_F19.func = 0


def phase_function_HG(Theta,g):
    # Theta is scattering angle, Theta=0 is formard
    y = (1-g*g)/(1+g*g-2*g*np.cos(Theta))**1.5
    # y *= 1./(1-g*g)/(1+g*g-2*g*np.cos(np.pi*150./180.))
    y *= 1.0/(4*np.pi)
    return y

def phase_function_D03(Theta,g,nu): # Draine 2003; nu=0 (HG)->1 (Rayleigh)
    g2 = g*g
    ch = np.cos(Theta)
    y = (1-g2)/(1+nu*(1+2*g2)/3)*((1+nu*ch*ch)/(1+g2-2*g*ch)**(1.5))/4/np.pi
    # y *= 1./(1-g*g)/(1+g*g-2*g*np.cos(np.pi*150./180.))
    return y

def polarization_frac(pmax,Theta): # scattering angle, radians
    return pmax*np.sin(Theta)**2/(1+np.cos(Theta)**2)


def scatter_test():
    deg = np.pi/180.0

    print('*** phase functions ***')
    print('phase function [Henyey-Greenstein], 0,90,180 degrees:')
    g = 0.7
    print('using g =',g)
    print(phase_function_HG(0*deg,g))
    print(phase_function_HG(90*deg,g))
    print(phase_function_HG(180*deg,g))
    def pf(x): return 2*np.pi*np.sin(x)*phase_function_HG(x,g)
    print('phase fn norm:',integrate.quad(pf,0,np.pi)[0])
    
    print('phase function [Draine 2003], 0,90,180 degrees:')
    g,nu = 0.7,0.123
    print('using g, nu =',g,nu)
    print(phase_function_D03(0*deg,g,nu))
    print(phase_function_D03(90*deg,g,nu))
    print(phase_function_D03(180*deg,g,nu))
    def pf(x): return 2*np.pi*np.sin(x)*phase_function_D03(x,g,nu)
    print('phase fn norm:',integrate.quad(pf,0,np.pi)[0])
    
    print('phase function [Frattin 2019], 0,90,180 degrees:')
    print(phase_function_F19(0*deg))
    print(phase_function_F19(90*deg))
    print(phase_function_F19(180*deg))
    def pf(x): return 2*np.pi*np.sin(x)*phase_function_F19(x)
    print('phase fn norm:',integrate.quad(pf,0,np.pi)[0])

    print(' ')
    print('*** polarization fractions ***')

    print('polarization frac [Rayleigh-like] at 0,90,180 degrees:')
    print(polarization_frac(1,0*deg))
    print(polarization_frac(1,90*deg))
    print(polarization_frac(1,180*deg))

    print('polarization frac [Frattin 2019] at 0,90,180 degrees:')
    print(polarization_frac_F19(1,0*deg))
    print(polarization_frac_F19(1,90*deg))
    print(polarization_frac_F19(1,180*deg))

# scatter_test()
