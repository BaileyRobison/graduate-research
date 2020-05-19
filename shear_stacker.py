
#class that performs shear stacking
#stores information about sources, number of lenses, and other important parameters
#performs stacking for each lens and returns output
#user can specify which estimators: Schrabback or Clampitt-Jain

# a version that uses a simulated single-lens system

#import relevant modules
import numpy as np
import astropy.table as table
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import gc


class Stacker():

    #general physical parameters
    Efactor = 1.66e12 #(c^2/4piG) in units of (Mo/pc)
    sigma_e = 0.28 #error in each of the ellipticity measurements
    cosmo = FlatLambdaCDM(H0=70., Om0=0.3) #cosmology

    def __init__(self, radii, estimator, data_path, source_path):
        self.radii = radii #radial bins
        self.numBins = (len(radii)-1) #number of bins

        if estimator != 'SCH' and estimator != 'CJ':
            raise ValueError('Incorrect estimator specified. Accepted estimators are: \'SCH\' or \'CJ\'')
        self.estimator = estimator
        
        self.source_path = data_path + source_path

        #sigma crit inverse function
        sigma_crit_table = table.Table.read(data_path + 'sigma_crit_inv.csv', format='ascii.csv')
        self.sigma_crit_inv = spline(sigma_crit_table['zl'],sigma_crit_table['sigma_crit_inv'])

        #initialize sum lists
        #monopole
        self.ERnums = np.zeros(self.numBins) #list of sums of E(R) numerators
        self.ERdems = np.zeros(self.numBins) #list of sums of E(R) denomimators
        self.ERNerrs = np.zeros(self.numBins) #list of errors on E(R) numerators

        #quadrupole      
        if self.estimator == 'SCH': #Schrabback estimators
            self.ERnumsQ1 = np.zeros(self.numBins) #list of sums of E(R) quadrupole numerators
            self.ERdemsQ1 = np.zeros(self.numBins) #list of sums of E(R) quadrupole denomimators
            self.ERNerrsQ1 = np.zeros(self.numBins) #list of errors on E(R) quadrupole numerators
            self.ERnumsQ2 = np.zeros(self.numBins) #list of sums of E(R) cross quadrupole numerators
            self.ERdemsQ2 = np.zeros(self.numBins) #list of sums of E(R) cross quadrupole denomimators
            self.ERNerrsQ2 = np.zeros(self.numBins) #list of errors on E(R) cross quadrupole numerators
        elif self.estimator == 'CJ': #CJ estimators
            #quadrupole numerators
            self.ERnums1P = np.zeros(self.numBins)
            self.ERnums1M = np.zeros(self.numBins)
            self.ERnums2P = np.zeros(self.numBins)
            self.ERnums2M = np.zeros(self.numBins)
            #quadrupole demoninators
            self.ERdems1P = np.zeros(self.numBins)
            self.ERdems1M = np.zeros(self.numBins)
            self.ERdems2P = np.zeros(self.numBins)
            self.ERdems2M = np.zeros(self.numBins)
            #quadrupole numerator errors
            self.ERNerrs1P = np.zeros(self.numBins)
            self.ERNerrs1M = np.zeros(self.numBins)
            self.ERNerrs2P = np.zeros(self.numBins)
            self.ERNerrs2M = np.zeros(self.numBins)

        #initialize counters
        self.ns = 0 #initialize number of sources
        self.npr = 0 #initialize number of lens-source pairs

    #calculate angular separation between two points (from SkyCoord)
    def angular_separation(self, lon1, lat1, lon2, lat2):
        sdlon = np.sin(lon2 - lon1)
        cdlon = np.cos(lon2 - lon1)
        slat1 = np.sin(lat1)
        slat2 = np.sin(lat2)
        clat1 = np.cos(lat1)
        clat2 = np.cos(lat2)

        num1 = clat2 * sdlon
        num2 = clat1 * slat2 - slat1 * clat2 * cdlon
        denominator = slat1 * slat2 + clat1 * clat2 * cdlon

        return np.arctan2(np.hypot(num1, num2), denominator) #return result in rad

    #calculate position angle betweent two points (from SkyCoord)
    #modified to produce angle CCW from x-axis
    def position_angle_sky(self, lon1, lat1, lon2, lat2):
        deltalon = lon2 - lon1
        colat = np.cos(lat2)

        x = np.sin(lat2) * np.cos(lat1) - colat * np.sin(lat1) * np.cos(deltalon)
        y = np.sin(deltalon) * colat

        pos_angle = np.arctan2(y, x) + np.pi/2
        return ( pos_angle + (2*np.pi) ) % (2*np.pi) #return result in rad

    #convert RA and DEC to x and y
    def convert_xy(self, RA1, DEC1, RA2, DEC2):
        RA1more = RA1-RA2 > np.pi #if spanning 360-0 gap
        RA2more = RA2-RA1 > np.pi #if spanning 0-360 gap
            
        RA2 = RA2 + RA1more*(2*np.pi)
        RA2 = RA2 - RA2more*(2*np.pi)

        x = -(RA2-RA1)*np.cos(DEC1)
        y = DEC2-DEC1
        return x,y

    #convert x and y to RA and DEC
    def convert_radec(self, RA1, DEC1, sx, sy):
        RAprime = RA1 - ( sx/np.cos(DEC1) )
        RAprime = (RAprime + (2*np.pi)) % (2*np.pi)
        DECprime = sy + DEC1

        return RAprime, DECprime

    #rotate coordinate frame (e1, e2, RA, and DEC)
    def rotate_coords(self, l, s, sx, sy, ths):
        numSource = len(s['RA']) #number of sources    

        #rotate e1, e2 by -2s
        coss = np.cos(-2*ths)*np.ones(numSource)
        sins = np.sin(-2*ths)*np.ones(numSource)
        Rs = np.array([coss,-1.0*sins,sins,coss])
        Rs = Rs.T.reshape((numSource,2,2))

        es = [s['e1'], s['e2']] #create arrays of ellipticities  
        es = np.transpose(es)
        eRs = [np.dot(rot,ell) for rot,ell in zip(Rs,es)] #perform rotation
        eRs = np.transpose(eRs)
        e1prime = eRs[0]
        e2prime = eRs[1]

        #rotate x, y by s
        coss = np.cos(-1*ths)*np.ones(numSource)
        sins = np.sin(-1*ths)*np.ones(numSource)
        Rs = np.array([coss,-1.0*sins,sins,coss])
        Rs = Rs.T.reshape((numSource,2,2))

        es = [sx, sy] #array of x's and y's
        es = np.transpose(es)
        eRs = [np.dot(rot,ell) for rot,ell in zip(Rs,es)] #perform rotation
        eRs = np.transpose(eRs)
        sxprime = eRs[0]
        syprime = eRs[1]

        return e1prime, e2prime, sxprime, syprime
        
    #perform shear stack for single lens
    def stack_shear(self, l):
        #read in sources
        sources = table.Table.read(self.source_path, format='ascii.csv')

        #calculate distance to lens
        l_dist = np.array(self.cosmo.angular_diameter_distance(l['z']))*1000.0 #distance to lens galaxy in kpc

        #convert RA and DEC from degrees to radians
        l['RA']*=(np.pi/180)
        l['DEC']*=(np.pi/180)
        sources['RA']*=(np.pi/180)
        sources['DEC']*=(np.pi/180)

        #calculate distances and restrict source sample
        separations = self.angular_separation(l['RA'], l['DEC'], sources['RA'], sources['DEC'])
        r_dists = separations * l_dist

        r_inds = np.digitize(r_dists, self.radii)-1 #determine indices of radial bins for distances
        sources = sources[r_inds < self.numBins] #eliminate furthest bin from source sample
        r_dists = r_dists[r_inds < self.numBins] #eliminate distances that are too far
        r_inds = r_inds[r_inds < self.numBins] #eliminate indices that are too far
        
        self.npr += len(sources) #increment number of lens-source pairs

        if len(sources) > 0: #if there are any sources                
            #calculate lens positon angle
            pos_angle = l['THETA']
            theta_shift = pos_angle*(np.pi/180)

            #rotate entire frame to align with x-axis

            #convert RA and DEC to x and y
            sx, sy = self.convert_xy(l['RA'],l['DEC'],sources['RA'],sources['DEC'])

            #rotate e1, e2, x, and y
            e1prime, e2prime, sxprime, syprime = self.rotate_coords(l,sources,sx,sy,theta_shift)
            sources['e1'] = e1prime
            sources['e2'] = e2prime

            #calculate position angle of sources (x-axis)
            thetas = np.arctan2(syprime, sxprime)
            
            e1 = sources['e1']
            e2 = sources['e2']
            eR1s = -e1*np.cos(2*thetas) - e2*np.sin(2*thetas)
            eR2s = e1*np.sin(2*thetas) - e2*np.cos(2*thetas)

            #calculate E_crit
            Ecrit = 1./self.sigma_crit_inv(l['z'])
            Wcrit = Ecrit**(-2)

            #calculate monopole
            histWeight = eR1s*sources['weight']
            ERnumsH = Ecrit*Wcrit*np.histogram(r_dists,bins=self.radii,weights=histWeight)[0] #add to E(R) numerator
            histWeight = sources['weight']
            ERdemsH = Wcrit*np.histogram(r_dists,bins=self.radii,weights=histWeight)[0] #add to E(R) denominator
            histWeight = Ecrit*Wcrit*sources['weight']*self.sigma_e
            histWeight = np.square(histWeight)
            ERNerrsH = np.histogram(r_dists,bins=self.radii,weights=histWeight)[0] #add to E(R) errors

            #sum histogram results to lists
            self.ERnums = self.ERnums + ERnumsH
            self.ERdems = self.ERdems + ERdemsH
            self.ERNerrs = self.ERNerrs + ERNerrsH

            #calculate quadrupole
            eg = abs( (l['A']-l['B']) / (l['A']+l['B']) ) #galaxy ellipticity

            if eg!=0: #CG quadrupole estimators
                
                if self.estimator == 'SCH':

                    #Schrabback estimators
    
                    #add E(R) tangential and cross results to sums
                    #numerators
                    histWeight = eR1s*sources['weight']*np.cos(2*thetas)
                    ERnumsH1 = Ecrit*Wcrit*np.histogram(r_dists,bins=self.radii,weights=histWeight)[0] #add to E(R) numerator
                    histWeight = eR2s*sources['weight']*np.sin(2*thetas)
                    ERnumsH2 = Ecrit*Wcrit*np.histogram(r_dists,bins=self.radii,weights=histWeight)[0] #add to E(R) numerator
                    
                    #denominators
                    histWeight = sources['weight']*np.cos(2*thetas)*np.cos(2*thetas)
                    ERdemsH1 = 2*Wcrit*np.histogram(r_dists,bins=self.radii,weights=histWeight)[0] #add to E(R) denominator
                    histWeight = sources['weight']*np.sin(2*thetas)*np.sin(2*thetas)
                    ERdemsH2 = 2*Wcrit*np.histogram(r_dists,bins=self.radii,weights=histWeight)[0] #add to E(R) denominator
                    
                    #error numerators
                    histWeight = Ecrit*Wcrit*sources['weight']*np.cos(2*thetas)
                    histWeight = np.square(histWeight)
                    ERNerrsH1 = np.histogram(r_dists,bins=self.radii,weights=histWeight)[0] #add to E(R) errors
                    histWeight = Ecrit*Wcrit*sources['weight']*np.sin(2*thetas)
                    histWeight = np.square(histWeight)
                    ERNerrsH2 = np.histogram(r_dists,bins=self.radii,weights=histWeight)[0] #add to E(R) errors
    
                    #sum histogram results to lists
                    self.ERnumsQ1 = self.ERnumsQ1 + ERnumsH1
                    self.ERdemsQ1 = self.ERdemsQ1 + ERdemsH1
                    self.ERNerrsQ1 = self.ERNerrsQ1 + ERNerrsH1
                    self.ERnumsQ2 = self.ERnumsQ2 + ERnumsH2
                    self.ERdemsQ2 = self.ERdemsQ2 + ERdemsH2
                    self.ERNerrsQ2 = self.ERNerrsQ2 + ERNerrsH2

                elif self.estimator == 'CJ':

                    # CJ estimator
                    
                    th1 = (thetas+(np.pi/8)) % (np.pi/2) #reduce to single quadrant, for gamma_1
                    th2 = thetas % (np.pi/2) #reduce to single quadrant, for gamma_2
                    
                    #E_1^+ estimator
                    inds = np.logical_and(th1 >= 0, th1 < (np.pi/4))
                    shears = sources['e1']*inds
                    weights = sources['weight']*inds
                    histResult = Ecrit*Wcrit*np.histogram(r_dists, bins=self.radii, weights=shears*weights)[0]
                    self.ERnums1P = self.ERnums1P + histResult
                    histResult = Wcrit*np.histogram(r_dists, bins=self.radii, weights=weights)[0]
                    self.ERdems1P = self.ERdems1P + histResult
                    histResult = Ecrit*Ecrit*Wcrit*Wcrit*np.histogram(r_dists, bins=self.radii, weights=np.square(weights))[0]
                    self.ERNerrs1P = self.ERNerrs1P + histResult
    
                    #E_1^- estimator
                    inds = np.logical_and(th1 >= (np.pi/4), th1 < (np.pi/2))
                    shears = sources['e1']*inds
                    weights = sources['weight']*inds
                    histResult = Ecrit*Wcrit*np.histogram(r_dists, bins=self.radii, weights=shears*weights)[0]
                    self.ERnums1M = self.ERnums1M + histResult
                    histResult = Wcrit*np.histogram(r_dists, bins=self.radii, weights=weights)[0]
                    self.ERdems1M = self.ERdems1M + histResult
                    histResult = Ecrit*Ecrit*Wcrit*Wcrit*np.histogram(r_dists, bins=self.radii, weights=np.square(weights))[0]
                    self.ERNerrs1M = self.ERNerrs1M + histResult
    
                    #E_2^+ estimator
                    inds = np.logical_and(th2 >= 0, th2 < (np.pi/4))
                    shears = sources['e2']*inds
                    weights = sources['weight']*inds
                    histResult = Ecrit*Wcrit*np.histogram(r_dists, bins=self.radii, weights=shears*weights)[0]
                    self.ERnums2P = self.ERnums2P + histResult
                    histResult = Wcrit*np.histogram(r_dists, bins=self.radii, weights=weights)[0]
                    self.ERdems2P = self.ERdems2P + histResult
                    histResult = Ecrit*Ecrit*Wcrit*Wcrit*np.histogram(r_dists, bins=self.radii, weights=np.square(weights))[0]
                    self.ERNerrs2P = self.ERNerrs2P + histResult
    
                    #E_2^- estimator
                    inds = np.logical_and(th2 >= (np.pi/4), th2 < (np.pi/2))
                    shears = sources['e2']*inds
                    weights = sources['weight']*inds
                    histResult = Ecrit*Wcrit*np.histogram(r_dists, bins=self.radii, weights=shears*weights)[0]
                    self.ERnums2M = self.ERnums2M + histResult
                    histResult = Wcrit*np.histogram(r_dists, bins=self.radii, weights=weights)[0]
                    self.ERdems2M = self.ERdems2M + histResult
                    histResult = Ecrit*Ecrit*Wcrit*Wcrit*np.histogram(r_dists, bins=self.radii, weights=np.square(weights))[0]
                    self.ERNerrs2M = self.ERNerrs2M + histResult

        #run garbage collection to free memory
        del sources[:]
        gc.collect()

    #final calculations and output
    def return_stack(self):
        #convert 0s to 1s to avoid divide-by-zero errors
        self.ERdems[self.ERdems==0.0]=1.0
        if self.estimator == 'SCH':
            self.ERdemsQ1[self.ERdemsQ1==0.0]=1.0
            self.ERdemsQ2[self.ERdemsQ2==0.0]=1.0
        elif self.estimator == 'CJ':
            self.ERdems1P[self.ERdems1P==0.0]=1.0
            self.ERdems1M[self.ERdems1M==0.0]=1.0
            self.ERdems2P[self.ERdems2P==0.0]=1.0
            self.ERdems2M[self.ERdems2M==0.0]=1.0

        #divide by weight sums to determine weighted average for each bin
        ERAvgs = self.ERnums / self.ERdems #list of weighted average excess mass density
        ERAvge = np.sqrt(self.ERNerrs) / self.ERdems #list of errors in average excess mass density

        #quadrupole estimators and errors
        if self.estimator == 'SCH':
            ERAvgsQ1 = self.ERnumsQ1 / self.ERdemsQ1 #list of weighted average excess mass density in quadrupole
            ERAvgsQ2 = self.ERnumsQ2 / self.ERdemsQ2 #list of weighted average excess mass density in cross quadrupole
    
            ERAvgeQ1 = self.sigma_e * np.sqrt(self.ERNerrsQ1) / self.ERdemsQ1
            ERAvgeQ2 = self.sigma_e * np.sqrt(self.ERNerrsQ2) / self.ERdemsQ2
    
            return ERAvgs, ERAvge, ERAvgsQ1, ERAvgeQ1, ERAvgsQ2, ERAvgeQ2, self.ns, self.npr
        
        elif self.estimator == 'CJ':
            ERAvgs1P = self.ERnums1P / self.ERdems1P
            ERAvge1P = self.sigma_e * np.sqrt(self.ERNerrs1P) / self.ERdems1P
    
            ERAvgs1M = self.ERnums1M / self.ERdems1M
            ERAvge1M = self.sigma_e * np.sqrt(self.ERNerrs1M) / self.ERdems1M
    
            ERAvgs2P = self.ERnums2P / self.ERdems2P
            ERAvge2P = self.sigma_e * np.sqrt(self.ERNerrs2P) / self.ERdems2P
    
            ERAvgs2M = self.ERnums2M / self.ERdems2M
            ERAvge2M = self.sigma_e * np.sqrt(self.ERNerrs2M) / self.ERdems2M
    
            return ERAvgs, ERAvge, ERAvgs1P, ERAvge1P, ERAvgs1M, ERAvge1M, ERAvgs2P, ERAvge2P, ERAvgs2M, ERAvge2M, self.ns, self.npr





