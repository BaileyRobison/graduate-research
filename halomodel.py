
#ANISOTROPIC version of halomodel.py

#import relevant modules
import numpy as np
from astropy.cosmology import FlatLambdaCDM
#from hankel import HankelTransform
#from hankelConvolve import hankelConvolve
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.integrate import quad


class halomodel():
    #class parameters
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3) #cosmology
    h = 0.7 #Hubble constant
    G = 6.67e-11 #gravitational constant in (m^3)(kg^-1)(s^-2)
    A200 = 3.93 #A parameter, for calculating c200, from Duffy et al. (2015)
    B200 = -0.097 #B parameter, for calculating c200, from Duffy et al. (2015)
    Mpivot = (2.0e12)/h #pivot mass in solar masses, for calculating c200, from Duffy et al. (2015)
    c_factor = 1.0 #concentration factor of host halo for OG term
    tau_factor = 10.0 #relation between tau and c200
    fsat = 0.53 #satellite fraction for group

    #create mass parameters
    M200 = None #virial mass of halo
    Mgroup = None #virial mass of host halo
    Ecrit = None #critical density

    def __init__(self, z_lens, Mstar, M200=None): #takes z_lens and Mstar, come from data files
        self.z_lens = z_lens
        self.Mstar = Mstar
        self.pc = self.cosmo.critical_density(z_lens).si.value #critical density in (kg)(m^-3)
        if M200 is not None: self.setMass(M200) #if mass given, set mass

    #function that calculates useful parameters from halo mass
    def setMass(self, M200):
        c200 = self.A200*((M200/self.Mpivot)**self.B200) #halo concentration
        R200 = ( (3.0/4.0/np.pi/200.0/self.pc)*M200*(2.0e30) )**(1.0/3.0) #virial radius R200 in m
        R200 = R200/(3.086e16) #R200 in pc
        rs = R200/c200 #scale radius in pc
        dc = (200.0/3.0)*(c200**3.0)/(np.log(1.0+c200) - c200/(1.0+c200)) #overdensity
        ps = dc*self.pc/(2e30)*((3.086e16)**3.0) #critical density in M_solar/pc^3

        #set as class variables
        self.M200 = M200
        self.c200 = c200
        self.R200 = R200
        self.rs = rs
        self.ps = ps

    def setHostMass(self, Mgroup): #set host halo mass
        self.Mgroup = Mgroup
        
    def setEcrit(self, Ecrit): #set critical density
        self.Ecrit = Ecrit

    #truncated NFW functions
    def fxmore(self, x): #x>1 case for F(x)
        return np.divide(np.arccos(np.divide(1.0, x)), np.sqrt(np.square(x)-1.0))
    def fxless(self, x): #x<1 case for F(x)
        n = -1.0*np.log(np.divide(1.0, x) - np.sqrt(np.divide(1.0, np.square(x)) - 1.0))
        return np.divide(n, np.sqrt(1.0-np.square(x)))
    def fx(self, x):
        return np.piecewise(x, [x==1.0, x>1.0, x<1.0], [1.0, self.fxmore, self.fxless])

    def lx(self, x, tau):
        return np.log( np.divide(x, np.sqrt(tau*tau + np.square(x)) + tau) )

    def Estar(self, r): #stellar mass term
        return np.divide(self.Mstar, np.pi*np.square(r*1000.0))

    def Enfw(self, r, **kwargs): #NFW profile
        if 'c' in kwargs: #if the user specifies a value
            c200 = kwargs['c']
            rs = self.R200/c200
            dc = (200.0/3.0)*(c200**3.0)/(np.log(1.0+c200) - c200/(1.0+c200))
            ps = dc*self.pc/(2e30)*((3.086e16)**3.0)
        else:
            c200 = self.c200
            rs = self.rs
            ps = self.ps
        x = r*1000.0/rs #radius ratios (pc/pc)
        tau = self.tau_factor*c200
        M0 = 4.0*np.pi*ps*rs*rs*rs

        Ea = np.divide(tau*tau+1.0, np.square(x)-1.0)*(1.0-self.fx(x)) + (2.0*self.fx(x)) - np.pi/(np.sqrt(tau*tau+np.square(x)))
        Eb = np.divide(tau*tau-1.0, tau*np.sqrt(tau*tau+np.square(x)))*self.lx(x,tau)
        E = (M0/(rs*rs)) * (tau*tau/(2.0*np.pi*(tau*tau+1.0)**2.0)) * (Ea+Eb)
        return E

    def Mproj(self, r, **kwargs): #projected enclosed mass
        if 'c' in kwargs: #if the user specifies a value
            c200 = kwargs['c']
            rs = self.R200/c200
            dc = (200.0/3.0)*(c200**3.0)/(np.log(1.0+c200) - c200/(1.0+c200))
            ps = dc*self.pc/(2e30)*((3.086e16)**3.0)
        else:
            c200 = self.c200
            rs = self.rs
            ps = self.ps
        x = r*1000.0/rs #radius ratios (pc/pc)
        tau = self.tau_factor*c200
        M0 = 4.0*np.pi*ps*rs*rs*rs

        Ma = (tau*tau + 1.0 + 2.0*(np.square(x)-1.0))*self.fx(x) + tau*np.pi + (tau*tau-1.0)*np.log(tau)
        Mb = np.sqrt(tau*tau+np.square(x))*(-1.0*np.pi + self.lx(x,tau)*(tau*tau-1.0)/tau)
        Mp = (M0*tau*tau)/((tau*tau + 1.0)**2.0)*(Ma+Mb)
        return Mp

    def delEnfw(self, r, **kwargs): #delta NFW profile
        return np.subtract( np.divide(self.Mproj(r, **kwargs), np.pi*np.square(r*1000.0)), self.Enfw(r, **kwargs) )

    def E1h(self, r): #1-halo term
        return self.Estar(r)+self.delEnfw(r)


    #remove offset group term for display purposes
    #it is unlikely that someone will have the Hankel package
    '''
    def delEOG(self, r): #offset group term      #this doesn't work right now
        M200 = self.M200 #save M200 of halo
        self.setMass(self.Mgroup) #set group mass as mass
        norm = self.Mproj(r[-1], c=self.c_factor*self.c200) #normalization factor

        f1 = lambda x : self.Enfw(x)
        f2 = lambda x : self.Enfw(x, c=self.c_factor*self.c200)/norm #probability function
        ks = np.logspace(-5,1,100)
        hcon = hankelConvolve(f1, f2, k=ks, N=2000, h=0.01) #create convolution object
        offset = 1000.0*1000.0*self.fsat*hcon.convolve(r)
        self.setMass(M200) #reset mass to halo mass

        EOG = spline(r, offset)
        EOGint = lambda x : EOG(x)*x #integrate for projected mass
        Mproj = lambda x : 2*np.pi*quad(EOGint,0,x)[0] #projected mass

        dEOG=[]
        for rad in r: #calculate delta sigma for each radius
            dEOG.append( Mproj(rad)/(np.pi*rad*rad) - EOG(rad) )

        return dEOG
    '''


    def delE(self, r): #total density profile
        return self.E1h(r) + self.delEOG(r)

    #anisotropic functions
    def fprimemore(self, x): #x>1 case for F'(x)
        n = np.sqrt(x*x - 1) - x*x*np.arccos(1/x)
        return n/(x*np.power(x*x-1, 3./2.))
    def fprimeless(self, x): #x<1 case for F'(x)
        c = x*x - (x**3)*np.sqrt(1/(x*x)-1)*np.log(1/x - np.sqrt(1/(x*x)-1)) - 1
        return c/(x*np.square(x*x-1))
    def fprime(self, x): #F'(x)
        return np.piecewise(x, [x==1.0, x>1.0, x<1.0], [0.0, self.fprimemore, self.fprimeless])

    def lprime(self, x, tau): #L'(x)
        return tau/(x*np.sqrt(tau*tau + x*x))

    def E0(self, r, **kwargs): #monopole
        return self.Enfw(r, **kwargs)

    def n0(self, rs, **kwargs): #derivative
        return self.Eprime(rs, **kwargs)*(rs/self.E0(rs, **kwargs))

    def E2(self, rs, e, **kwargs): #quadrupole
        return (-0.5*e)*self.n0(rs, **kwargs)*self.E0(rs, **kwargs)

    def Eaniso(self, rs, theta, e, **kwargs): #total anisotropic model
        return self.E0(rs,**kwargs)+self.E2(rs, e,**kwargs)*np.cos(2*theta)

    def Eprime(self, r, **kwargs): #derivative of E_NFW
        if 'c' in kwargs: #if the user specifies a value
            c200 = kwargs['c']
        else:
            c200 = self.c200
        x = r*1000.0/self.rs #radius ratios (pc/pc)
        tau = self.tau_factor*c200
        M0 = 4.0*np.pi*self.ps*self.rs*self.rs*self.rs

        #derivatives of individual NFW profile terms
        t1 = -2*x*(tau*tau+1)/np.square(x*x-1)
        t2 = (tau*tau+1)/(x*x-1) * (2*x*self.fx(x)/(x*x-1) - self.fprime(x))
        t3 = 2*self.fprime(x)
        t4 = np.pi*x / np.power(tau*tau + x*x, 1.5)
        t5 = (tau*tau - 1)/(tau*np.sqrt(tau*tau + x*x)) * (self.lprime(x, tau) - (x*self.lx(x, tau))/(tau*tau + x*x))

        return (M0/(self.rs*self.rs)) * (tau*tau/(2.0*np.pi*(tau*tau+1.0)**2.0)) * (t1+t2+t3+t4+t5) * (1000/self.rs)

    #quadrupole estimators
    def check_iterable(self, test): #check if object is iterable
        iterable=False
        try:
            iterator = iter(test)
            iterable=True
        except:
            pass
        return iterable
    
    def I1(self, R):
        iterable = self.check_iterable(R) #whether or not R is iterable
        
        if iterable: #if iterable
            ints = []
            for radius in R:
                integrand = lambda x : (x**3)*self.E0(x)*self.n0(x) #function to integrate
                integrated = quad(integrand, 0.01, radius)[0]
                ints.append( integrated*(3/(radius**4)) )
            return np.array(ints)
        else: #if not iterable
            integrand = lambda x : (x**3)*self.E0(x)*self.n0(x) #function to integrate
            integrated = quad(integrand, 0.01, R)[0]
            return integrated*(3/(R**4))
        
    def I2(self, R):
        iterable = self.check_iterable(R) #whether or not R is iterable
        
        if iterable: #if iterable
            ints = []
            for radius in R:
                integrand = lambda x : self.E0(x)*self.n0(x)/x #function to integrate
                integrated = quad(integrand, radius, np.inf)[0]
                ints.append( integrated )
            return np.array(ints)
        else: #if not iterable
            integrand = lambda x : self.E0(x)*self.n0(x)/x #function to integrate
            integrated = quad(integrand, R, np.inf)[0]
            return integrated
    
    #CJ estimators
    def E1plus(self, R, e, const):
        a = 2.*self.I1(R) - self.E0(R)*self.n0(R)
        b = 2.*self.I2(R)
        c = self.E0(R)*self.n0(R)
        return (e/np.pi)*(0.5*a + (np.pi/4.)*b - (np.pi/4.)*c) + const
    
    def E2plus(self, R, e, const):
        a = 2.*self.I1(R) - self.E0(R)*self.n0(R) 
        return (e/np.pi)*(0.5*a) + const
    
    def E1minus(self, R, e, const):
        a = 2.*self.I1(R) - self.E0(R)*self.n0(R)
        b = 2.*self.I2(R)
        c = self.E0(R)*self.n0(R)
        return (e/np.pi)*(-0.5*a + (np.pi/4.)*b - (np.pi/4.)*c) + const
    
    def E2minus(self, R, e, const):
        a = 2.*self.I1(R) - self.E0(R)*self.n0(R) 
        return (e/np.pi)*(-0.5*a) + const
    
    #SCH estimators
    def f_sigma(self, R, e, const):
        a = self.E0(R)*self.n0(R) - self.I1(R) - self.I2(R)
        return (e/4.)*a + const
    
    def f45_sigma(self, R, e, const):
        a = self.I2(R) - self.I1(R)
        return (e/4.)*a + const



