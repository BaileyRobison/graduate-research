
#plot results of stickplot_table.py
#creates two plots
#   1: 4 panel plot of 2D shear map, 1D monopole fit, 2D monopole fit, residual shear map
#   2: 3 panel plot of convergence map, monopole convergence fit, residual convergence

#import modules
import numpy as np
import scipy.ndimage
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.cm as cm
import astropy.table as table
from scipy.optimize import curve_fit
from halomodel import halomodel
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import json
import argparse


#plot a stick depending on e1 and e2
def plot_stick(x, y, e1, e2, scale_factor=None):
    theta = np.arctan2(e2,e1)/2 #angle of line
    emag = np.sqrt(e1**2 + e2**2) #length of line

    xstick = emag/2*np.cos(theta) #endpoint in x
    ystick = emag/2*np.sin(theta) #endpoint in y
    if scale_factor is not None: #scale if factor given
        xstick *= scale_factor
        ystick *= scale_factor

    xs = [x-xstick, x+xstick] #xs to plot
    ys = [y-ystick, y+ystick] #ys to plot

    plt.plot(xs,ys,'k', linewidth=2, color='k')

#kappa from KSB
def kappaKaiserSquires(g1, g2):

    pad_length = 50

    #set up kx, ky grid
    k_xaxis = np.fft.fftfreq(g1.shape[1] + 2*pad_length) * 2. * np.pi
    k_yaxis = np.fft.fftfreq(g1.shape[0] + 2*pad_length) * 2. * np.pi

    kx, ky = np.meshgrid(k_xaxis, k_yaxis, indexing='ij')

    kz = kx + ky*1j

    ksq = kz*np.conj(kz)

    #adjust denominator for kz=0 to avoid division by 0
    ksq[0,0] = 1.
    exp2ipsi = kz*kz/ksq

    #complex g = g1 + i g2
    gz = g1 + g2*1j

    #go to fourier space
    gz = np.pad(gz, pad_length, mode='constant')

    gz_k = np.fft.fft2(gz)

    #Kaiser & Squires (1993)
    kz_k = np.conj(exp2ipsi)*gz_k

    #back to real space
    kz = np.fft.ifft2(kz_k)

    kz = kz[pad_length:-pad_length , pad_length:-pad_length]

    kappaE = np.real(kz)
    kappaB = np.imag(kz)
    return kappaE, kappaB

#plot convergence
def plot_kappa(x, y, g_1, g_2, gauss_rad=1.5, limit=2., plotcbar=True):
    #run KSB and produce smoothed map

    #smooth shear
    g1 = scipy.ndimage.filters.gaussian_filter(g_1, gauss_rad, 0)
    g2 = scipy.ndimage.filters.gaussian_filter(g_2, gauss_rad, 0)

    #get kappa
    k_e, k_b = kappaKaiserSquires(g1, g2)

    #contour levels
    levels=np.linspace(0,0.003,5)
    
    ax = plt.gca()
    CT = ax.contour(x, y, k_e, levels=levels)
    CS = ax.imshow(k_e.T, origin='lower', extent=(np.amin(x),np.amax(x),np.amin(y),np.amax(y)), vmin=-0.003, vmax=0.003, cmap=cm.CMRmap )
    #plt.pcolormesh(x,y,k_e)

    if plotcbar:
        cb = plt.colorbar(CS)
        cb.set_label(r'$\kappa$', fontsize=20)

    plt.xlim([-limit , limit])
    plt.ylim([-limit , limit])
    mpl.rc('xtick', labelsize=15)
    mpl.rc('ytick', labelsize=15)
    plt.xlabel(r'$x$ $(Mpc)$', fontsize=20)
    plt.ylabel(r'$y$ $(Mpc)$', fontsize=20)
    
    plt.draw()


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
args = parser.parse_args()


json_file_path = 'parameters2D.json'


#read hyper parameters from json file
f = open(json_file_path, 'r')
params = json.load(f)
f.close()

bin_range = params['bin_range']
scale = params['scale']
data_path = params['data_path']
lens_path = params['lens_path']
output_path = params['output_path']
plot_path = params['plot_path']


#global variables
outfile = output_path + 'stickplot_output.csv' #2D shear output
monofile = output_path + 'stickplot_mono.csv' #monopole output
usemonofile = True #whether or not to use monopole file

#import results
e_table = table.Table.read(outfile, format='ascii.csv')

grid_bins = np.array(list(set(e_table['x'])))
grid_bins.sort()
limit = float(bin_range) #size in kpc


plt.figure(figsize=(12,12))

#plot shear map
plt.subplot(221)

for e in e_table:
    plot_stick(e['x'], e['y'], e['e1'], e['e2'], scale_factor=scale)

#plot center
plt.plot(0,0,marker='x',color='r')

plt.xlim([-limit,limit])
plt.ylim([-limit,limit])
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x (kpc)', fontsize=16)
plt.ylabel('y (kpc)', fontsize=16)
plt.title('BOSS LRGs total shear', fontsize=16)

#fit monopole in 1D
plt.subplot(222)

#get critical density
lenses = table.Table.read(data_path + lens_path, format='ascii.csv')
z_lens = np.mean(lenses['z'])
model = halomodel(z_lens, 1e11) #create halo model object

sigma_crit_table = table.Table.read(data_path + 'sigma_crit_inv.csv', format='ascii.csv')
sigma_crit_inv = spline(sigma_crit_table['zl'],sigma_crit_table['sigma_crit_inv'])
Ecrit = 1./sigma_crit_inv(z_lens)

if usemonofile:
    #read in monopole data
    mono_table = table.Table.read(monofile, format='ascii.csv')
    rPlot = mono_table['Radius']
    shear_avg = Ecrit * mono_table['AvgE']
    shear_err = Ecrit * mono_table['AvgEErr']

    plt.errorbar(rPlot, shear_avg, yerr=shear_err, fmt='ko')
else:
    #radial bins
    radii = np.logspace(np.log10(100.), np.log10(4000.), 7)
    rPlot = [(a + b) /2 for a,b in zip(radii[:-1], radii[1:])] #radial data point positions

    #rotate shears
    thetas = np.arctan2(e_table['y'], e_table['x'])
    alphas = -2.0 * thetas

    #rotation matrices for monopole
    coss = np.cos(alphas) #array of cosines
    sins = np.sin(alphas) #array of sines
    Rs = np.array([coss,-1.0*sins,sins,coss])
    Rs = Rs.T.reshape((len(alphas),2,2)) #array of 2x2 rotation matrices

    #rotate monopole shear
    es = [ e_table['e1'],e_table['e2'] ]
    es = np.transpose(es)
    eRs = [np.dot(r,e) for r,e in zip(Rs,es)]
    eRs = np.transpose(eRs)
    eR1s = -1.0*eRs[0]
    eR2s = -1.0*eRs[1]

    #bin shears in radial bins
    r_dists = np.sqrt(e_table['x']**2 + e_table['y']**2)
    shear_sum = np.histogram(r_dists, bins=radii, weights=eR1s)[0]
    shear_count = np.histogram(r_dists, bins=radii)[0]
    shear_count[shear_count==0]=1
    shear_avg = Ecrit * shear_sum / shear_count

    plt.plot(rPlot,shear_avg,'ko')


#fit model to monopole
def modelfit(rs,M200,c200): #function for fitting
    model.setMass(M200) #set 1-halo mass
    return model.delEnfw(rs,c=c200)


par,cov = curve_fit(modelfit,rPlot,shear_avg,p0=[1e14,2.0])
err = np.sqrt(np.diag(cov))
model.setMass(par[0])

plot_radii = np.logspace(np.log10(rPlot[0]),np.log10(rPlot[-1]),100)
plt.plot(plot_radii, model.delEnfw(plot_radii,c=par[1]), 'k') #plot NFW profile

#plot labels from fit
ax = plt.gca()
mass_exp = ('{:.2g}'.format(par[0])[-2:])
mass_int = 10**int(mass_exp)
mass_string = r'$M_{200}$: '+'{:.2f}'.format(par[0]/mass_int)+r'$\pm$'+'{:.2f}'.format(err[0]/mass_int)
mass_string += 'e+'+mass_exp+r' $M_{\odot}$'
plt.text(0.95,0.90,mass_string,fontsize=14,ha='right',va='bottom',transform=ax.transAxes) #display M200
c_string = r'$c_{200}: $'+'{:.2f}'.format(par[1])+r'$\pm$'+'{:.2f}'.format(err[1])
plt.text(0.95,0.80,c_string,fontsize=14,ha='right',va='bottom',transform=ax.transAxes) #display c200

#plot and label
plt.xlabel(r'$Radial$ $Distance$ $(kpc)$', fontsize=14)
plt.ylabel(r'$\langle\Sigma(R)\rangle$ $(M_{\odot}/pc^2)$', fontsize=14)
plt.title('monopole shear', fontsize=14)
plt.xscale('log')
plt.yscale('log')


#generate monopole for subtraction
plt.subplot(223)

#generate pre-rotated monopole shear
mono_radii = np.sqrt(e_table['x']**2 + e_table['y']**2)
mono_eR1 = model.delEnfw(mono_radii,c=par[1]) / Ecrit
mono_eR2 = np.zeros(len(mono_eR1)) / Ecrit

#rotate monopole shears
#rotate shears
thetas = np.arctan2(e_table['y'], e_table['x'])
alphas = 2.0 * thetas #negative of previous angles

#rotation matrices for monopole
coss = np.cos(alphas) #array of cosines
sins = np.sin(alphas) #array of sines
Rs = np.array([coss,-1.0*sins,sins,coss])
Rs = Rs.T.reshape((len(alphas),2,2)) #array of 2x2 rotation matrices

#rotate monopole shear
es = [ mono_eR1,mono_eR2 ]
es = np.transpose(es)
eRs = [np.dot(r,e) for r,e in zip(Rs,es)]
eRs = np.transpose(eRs)
mono_e1s = -1.0*eRs[0]
mono_e2s = -1.0*eRs[1]

#plot monopole grid
for i in range(len(e_table)):
    plot_stick(e_table[i]['x'], e_table[i]['y'], mono_e1s[i], mono_e2s[i], scale_factor=scale)

#plot center
plt.plot(0,0,marker='x',color='r')

plt.xlim([-limit,limit])
plt.ylim([-limit,limit])
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x (kpc)', fontsize=16)
plt.ylabel('y (kpc)', fontsize=16)
plt.title('monopole shear fit', fontsize=16)


#plot subtracted shears
plt.subplot(224)

e1_residual = e_table['e1'] - mono_e1s
e2_residual = e_table['e2'] - mono_e2s

for i in range(len(e_table)):
    plot_stick(e_table[i]['x'], e_table[i]['y'], e1_residual[i], e2_residual[i], scale_factor=scale)

#plot center
plt.plot(0,0,marker='x',color='r')

plt.xlim([-limit,limit])
plt.ylim([-limit,limit])
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x (kpc)', fontsize=16)
plt.ylabel('y (kpc)', fontsize=16)
plt.title('residual shears', fontsize=16)

plt.savefig(plot_path + 'stickplot_gamma',bbox_inches='tight')




#plot kappa maps
plt.clf()
plt.figure(figsize=(18,6))

#plot observed kappa
plt.subplot(131)

e_table = table.Table.read(outfile, format='ascii.csv')

e1s = e_table['e1']
e2s = e_table['e2']
xs = e_table['x']/1000.
ys = e_table['y']/1000.

#reshape to 2D grids
e_length = int(np.sqrt(len(e1s)))
e1s = np.reshape(e1s, (e_length,e_length))
e2s = np.reshape(e2s, (e_length,e_length))
xs = np.reshape(xs, (e_length,e_length))
ys = np.reshape(ys, (e_length,e_length))

plot_kappa(xs, ys, e1s, e2s, limit=limit/1000, plotcbar=False)
plt.title('Total Shear', fontsize=16)


#plot monopole kappa
plt.subplot(132)
mono_e1s = np.reshape(mono_e1s, (e_length,e_length))
mono_e2s = np.reshape(mono_e2s, (e_length,e_length))

plot_kappa(xs, ys, mono_e1s, mono_e2s, limit=limit/1000, plotcbar=False)
plt.title('Monopole Shear', fontsize=16)


#plot residual kappa
plt.subplot(133)
e1_residual = np.reshape(e1_residual, (e_length,e_length))
e2_residual = np.reshape(e2_residual, (e_length,e_length))

plot_kappa(xs, ys, e1_residual, e2_residual, limit=limit/1000)
plt.title('Residual Shear', fontsize=16)

plt.savefig(plot_path + 'stickplot_kappa',bbox_inches='tight')







