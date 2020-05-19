
#calculate 2D grid of shear
#output table to be used for stickplot.py

#import relevant modules
import numpy as np
import astropy.table as table
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import time
import gc
import json
import argparse


#convert RA and DEC to x and y
def convert_xy(RA1, DEC1, RA2, DEC2):
	RA1more = RA1-RA2 > np.pi #if spanning 360-0 gap
	RA2more = RA2-RA1 > np.pi #if spanning 0-360 gap
	    
	RA2 = RA2 + RA1more*(2*np.pi)
	RA2 = RA2 - RA2more*(2*np.pi)

	x = -(RA2-RA1)*np.cos(DEC1)
	y = DEC2-DEC1
	return x,y

#rotate coordinate frame (e1, e2, RA, and DEC)
def rotate_coords(l, s, sx, sy, ths):
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




#command line arguments
parser = argparse.ArgumentParser(description='Stack shears',
                                 
                                 epilog='Parameters in file \n' +                       
                                        '(parameter) \t (type) \t (default) \t\t (description) \n' +
                                        'bin_range \t float \t\t 100 \t\t\t extent of grid in kpc \n' +
                                        'num_bins \t float \t\t 7 \t\t\t number of radial bins \n' +
                                        'scale \t\t float \t\t 1000 \t\t\t scale factor for shear stickplots \n' +
                                        'data_path \t string \t \"data/\" \t\t path to input files \n'  +
                                        'source_path \t string \t \"sources_sim.csv\" \t path to source data \n' + 
                                        'lens_path \t string \t \"lens_sim.csv\" \t path to lens data \n' + 
                                        'output_path \t string \t \"results/\" \t\t path to results folder \n' + 
                                        'plot_path \t string \t \"plots/\" \t\t path to plots folder',
                                        
                                 formatter_class=argparse.RawTextHelpFormatter)
#parser arguments
parser.add_argument('-v',
                       '--verbose',
                       action='store_true',
                       help='verbosity (optional)')
args = parser.parse_args()


json_file_path = 'parameters2D.json'


#read hyper parameters from json file
f = open(json_file_path, 'r')
params = json.load(f)
f.close()

bin_range = params['bin_range']
num_bins = params['num_bins']
data_path = params['data_path']
source_path = params['source_path']
lens_path = params['lens_path']
output_path = params['output_path']


#general parameters
sigma_e = 0.28 #error in each of the ellipticity measurements
cosmo = FlatLambdaCDM(H0=70., Om0=0.3) #cosmology

#create grid
step = 2 * bin_range / num_bins
grid_bins = np.arange(-bin_range,bin_range+step,step)
binmid = [(a + b) /2 for a,b in zip(grid_bins[:-1], grid_bins[1:])]

#bins for monopole
radii = np.logspace(np.log10(50.), np.log10(bin_range), 15)
rPlot = [(a + b) /2 for a,b in zip(radii[:-1], radii[1:])] #radial data point positions

#sigma crit inverse
sigma_crit_table = table.Table.read(data_path + 'sigma_crit_inv.csv', format='ascii.csv')
sigma_crit_inv = spline(sigma_crit_table['zl'],sigma_crit_table['sigma_crit_inv'])


#main script

print('Starting...')
#read in lens data
lenses = table.Table.read(data_path + lens_path, format='ascii.csv')

#initialize weighted average grids
count_hist = np.zeros(( len(binmid) , len(binmid) )) #number of sources in each bin
e1_hist_num = np.zeros(( len(binmid) , len(binmid) )) #numerator for e1 average
e2_hist_num = np.zeros(( len(binmid) , len(binmid) )) #numerator for e2 average
e1_hist_dem = np.zeros(( len(binmid) , len(binmid) )) #denominator for e1 average
e2_hist_dem = np.zeros(( len(binmid) , len(binmid) )) #denominator for e2 average
mono_num = np.zeros(len(rPlot)) #numerator for monopole average
mono_dem = np.zeros(len(rPlot)) #denominator for monopole average
mono_err = np.zeros(len(rPlot)) #numerator for monopole error

start_time = time.time()
for i in range(len(lenses)): #repeat for all lenses
	l = lenses[i]

	sources = table.Table.read(data_path + source_path)
	l_dist = np.array(cosmo.angular_diameter_distance(l['z']))*1000.0 #distance to lens galaxy in kpc

	if len(sources)>0:
		#convert RA and DEC from degrees to radians
		l['RA']*=(np.pi/180)
		l['DEC']*=(np.pi/180)
		sources['RA']*=(np.pi/180)
		sources['DEC']*=(np.pi/180)

		#transform to LRG-aligned Cartesian coordinate system
		#Ecrit = 1/sigma_crit_inv(l['z'])
		pos_angle = l['THETA']
		theta_shift = pos_angle*(np.pi/180)
		sx, sy = convert_xy(l['RA'],l['DEC'],sources['RA'],sources['DEC']) #convert RA and DEC to x and y			
		e1prime, e2prime, sxprime, syprime = rotate_coords(l,sources,sx,sy,theta_shift) #rotate e1, e2, x, and y

		#convert x,y and create weights
		sxprime *= l_dist
		syprime *= l_dist
		sources['e1'] = e1prime
		sources['e2'] = e2prime
		source_weight = sources['weight']

		###calculate 2d map

		#e1,e2 numerator and denominator of weighted avg
		count_hist += np.histogram2d(sxprime, syprime, grid_bins)[0] #counts histogram
		e1_hist_num += np.histogram2d(sxprime, syprime, grid_bins, weights = source_weight*sources['e1'])[0] #e1 numerator histogram
		e2_hist_num += np.histogram2d(sxprime, syprime, grid_bins, weights = source_weight*sources['e2'])[0] #e2 numerator histogram
		e1_hist_dem += np.histogram2d(sxprime, syprime, grid_bins, weights = source_weight)[0] #e1 denominator histogram
		e2_hist_dem += np.histogram2d(sxprime, syprime, grid_bins, weights = source_weight)[0] #e2 denominator histogram

		###calculate monopole

		#calculate position angle of sources (x-axis)
		thetas = np.arctan2(syprime, sxprime)
		alphas = -2.0 * thetas
		r_dists = np.sqrt(sxprime**2 + syprime**2)

		#rotation matrices for monopole
		coss = np.cos(alphas) #array of cosines
		sins = np.sin(alphas) #array of sines
		Rs = np.array([coss,-1.0*sins,sins,coss])
		Rs = Rs.T.reshape((len(alphas),2,2)) #array of 2x2 rotation matrices

		#rotate monopole shear
		es = [ sources['e1'],sources['e2'] ]
		es = np.transpose(es)
		eRs = [np.dot(r,e) for r,e in zip(Rs,es)]
		eRs = np.transpose(eRs)
		eR1s = -1.0*eRs[0]
		eR2s = -1.0*eRs[1]

		#calculate monopole
		mono_num += np.histogram(r_dists,bins=radii,weights=source_weight*eR1s)[0] #add to E(R) numerator
		mono_dem += np.histogram(r_dists,bins=radii,weights=source_weight)[0] #add to E(R) denominator
		mono_err += np.histogram(r_dists,bins=radii,weights=np.square(sigma_e*source_weight))[0] #add to E(R) errors


	del sources[:]
	gc.collect()

	if args.verbose and i%100 == 0: #print out progress occasionally
		print('{0:.2f}'.format(100 * i/len(lenses))+'%'+' - '+'{0:.3f}'.format(time.time()-start_time)+' s')


#when ending script
print(time.time()-start_time)

        
###write 2d map output

#convert 0s -> 1s in denominators
e1_hist_dem[e1_hist_dem==0]=1
e2_hist_dem[e2_hist_dem==0]=1

#calculate weighted averages of e1 and e2
e1_hist_total = e1_hist_num / e1_hist_dem #2d weights average e1
e2_hist_total = e2_hist_num / e2_hist_dem #2d weights average e2

#write results to total table
bin_x, bin_y = np.meshgrid(binmid , binmid, indexing='ij')
e_hist_table = table.Table([bin_x.flatten(),bin_y.flatten(),e1_hist_total.flatten(),e2_hist_total.flatten(),count_hist.flatten()],
			     names=('x','y','e1','e2','n'))
e_hist_table.write(output_path+'stickplot_output.csv', format='ascii.csv')

###write monopole output

#convert 0s to 1s to avoid divide-by-zero errors
mono_dem[mono_dem==0]=1

#divide by weight sums to determine weighted average for each bin
ERAvgs = mono_num / mono_dem #list of weighted average excess mass density
ERAvge = np.sqrt(mono_err) / mono_dem #list of errors in average excess mass density

mono_table = table.Table([rPlot, ERAvgs, ERAvge], names=('Radius','AvgE','AvgEErr'))
mono_table.write(output_path + 'stickplot_mono.csv', format='ascii.csv')













