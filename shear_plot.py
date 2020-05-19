
#plot results from shear_cfis.py

###import relevant modules
import numpy as np
import matplotlib.pyplot as plt
import astropy.table as table
from scipy.optimize import curve_fit
from halomodel import halomodel
import json
import argparse


#command line arguments
parser = argparse.ArgumentParser(description='Stack shears',
                                 
                                 epilog='Parameters in file \n' +                       
                                        '(parameter) \t (type) \t (default) \t\t (description) \n' +
                                        'estimators \t string \t CJ \t\t\t estimators to use: either CJ or SCH \n' +
                                        'r_inner \t float \t\t 50 \t\t\t inner radius of radial bins \n' +
                                        'r_outer \t float \t\t 4000 \t\t\t outer radius of radial bins \n' +
                                        'h_factor \t Boolean \t true \t\t\t whether or not to use units with h \n' +
                                        'data_path \t string \t \"data/\" \t\t path to input files \n'  +
                                        'source_path \t string \t \"sources_sim.csv\" \t path to source data \n' + 
                                        'lens_path \t string \t \"lens_sim.csv\" \t path to lens data \n' + 
                                        'output_path \t string \t \"results/\" \t\t path to results folder \n' + 
                                        'plot_path \t string \t \"plots/\" \t\t path to plots folder',
                                        
                                 formatter_class=argparse.RawTextHelpFormatter)
args = parser.parse_args()


json_file_path = 'parameters.json'


#read hyper parameters from json file
f = open(json_file_path, 'r')
params = json.load(f)
f.close()

estimators = params['estimators']
radii = np.logspace(np.log10(params['r_inner']), np.log10(params['r_outer']), params['num_bins']) #radial bins for stacking
h_factor = params['h_factor']
data_path = params['data_path']
lens_path = params['lens_path']
output_path = params['output_path']
plot_path = params['plot_path']

lenses = table.Table.read(data_path + lens_path, format='ascii.csv')
z_lens = np.median(lenses['z'])

results = table.Table.read(output_path+'shear'+estimators+'_output.csv', format='ascii') #read in results from shear_fast.py


#divide error bars to sim more data
#account for file size limitations
#increase by N of simulated data -> decrease by sqrt{N} in error bars
factorExtra = 100
if estimators=='SCH':
    cols = ['AvgEErr','AvgEErr_Q1','AvgEErr_Q2']
elif estimators=='CJ':
    cols = ['AvgEErr','AvgE1Perr','AvgE1Merr','AvgE2Perr','AvgE2Merr']
for c in cols:
    results[c] /= np.sqrt(factorExtra)
    

h = 0.7
if estimators == 'SCH':
    convert_cols = ['AvgE','AvgEErr','AvgE_Q1','AvgEErr_Q1' ,'AvgE_Q2','AvgEErr_Q2']
elif estimators == 'CJ':
    convert_cols = ['AvgE','AvgEErr','AvgE1P','AvgE1Perr','AvgE1M','AvgE1Merr','AvgE2P','AvgE2Perr','AvgE2M','AvgE2Merr']
if h_factor:
    results['Radius'] *= h
    for col in convert_cols:
        results[col] /= h

#plot sigma profile
plt.figure(figsize=(18,6))
plt.subplot(131) #plot monopole

plt.errorbar(results['Radius'],results['AvgE'],yerr=results['AvgEErr'],fmt='ko') #plot monopole data

model = halomodel(z_lens, 1e11) #create halo model object

def modelfit(rs,M200,c200): #function for fitting
	model.setMass(M200) #set 1-halo mass
	return model.delEnfw(rs,c=c200)

#fit and plot model
par,cov = curve_fit(modelfit,results['Radius'],results['AvgE'],sigma=results['AvgEErr'],p0=[1e12,2.0])
err = np.sqrt(np.diag(cov))
model.setMass(par[0])

plot_radii = np.logspace(np.log10(results['Radius'][0]),np.log10(results['Radius'][-1]),100)
plt.plot(plot_radii, model.delEnfw(plot_radii,c=par[1]), 'k') #plot NFW profile

#format plot and output
if h_factor:
    plt.xlabel(r'$Radial$ $Distance$ $(kpc/h)$', fontsize=14)
    plt.ylabel(r'$\langle\Sigma(R)\rangle$ $(M_{\odot}h/pc^2)$', fontsize=14)
else:
    plt.xlabel(r'$Radial$ $Distance$ $(kpc)$', fontsize=14)
    plt.ylabel(r'$\langle\Sigma(R)\rangle$ $(M_{\odot}/pc^2)$', fontsize=14)

plt.title('monopole shear', fontsize=14)
plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
mass_exp = ('{:.2g}'.format(par[0])[-2:])
mass_int = 10**int(mass_exp)
mass_string = r'$M_{200}$: '+'{:.2f}'.format(par[0]/mass_int)+r'$\pm$'+'{:.2f}'.format(err[0]/mass_int)
mass_string += 'e+'+mass_exp+r' $M_{\odot}$'
plt.text(0.95,0.90,mass_string,fontsize=14,ha='right',va='bottom',transform=ax.transAxes) #display M200
c_string = r'$c_{200}: $'+'{:.2f}'.format(par[1])+r'$\pm$'+'{:.2f}'.format(err[1])
plt.text(0.95,0.80,c_string,fontsize=14,ha='right',va='bottom',transform=ax.transAxes) #display c200


if h_factor:
    if estimators=='SCH':
        f_sigma_fit = lambda r,e,c : model.f_sigma(r/h,e,c)/h
        f45_sigma_fit = lambda r,e,c : model.f45_sigma(r/h,e,c)/h
    elif estimators=='CJ':
        E1plus_fit = lambda r,e,c : model.E1plus(r/h,e,c)/h
        E2plus_fit = lambda r,e,c : model.E2plus(r/h,e,c)/h
        E1minus_fit = lambda r,e,c : model.E1minus(r/h,e,c)/h
        E2minus_fit = lambda r,e,c : model.E2minus(r/h,e,c)/h
else:
    if estimators=='SCH':
        f_sigma_fit = lambda r,e,c : model.f_sigma(r,e,c)
        f45_sigma_fit = lambda r,e,c : model.f45_sigma(r,e,c)
    elif estimators=='CJ':
        E1plus_fit = lambda r,e,c : model.E1plus(r,e,c)
        E2plus_fit = lambda r,e,c : model.E2plus(r,e,c)
        E1minus_fit = lambda r,e,c : model.E1minus(r,e,c)
        E2minus_fit = lambda r,e,c : model.E2minus(r,e,c)


plt.subplot(132) #plot quadrupole plus estimators

#plot data
if estimators == 'SCH':
    plt.errorbar(results['Radius'],results['AvgE_Q1'],yerr=results['AvgEErr_Q1'],fmt='ko')
elif estimators == 'CJ':
    p1=plt.errorbar(results['Radius'],results['AvgE1P'],yerr=results['AvgE1Perr'],fmt='ko')
    p2=plt.errorbar(results['Radius'],results['AvgE2P'],yerr=results['AvgE2Perr'],fmt='go')

#format plot and output
if h_factor:
    plt.xlabel(r'$Radial$ $Distance$ $(kpc/h)$', fontsize=14)
    plt.ylabel(r'$\langle\Sigma(R)\rangle$ $(M_{\odot}h/pc^2)$', fontsize=14)
else:    
    plt.xlabel(r'$Radial$ $Distance$ $(kpc)$', fontsize=14)
    plt.ylabel(r'$\langle\Sigma(R)\rangle$ $(M_{\odot}/pc^2)$', fontsize=14)
plt.xscale('log')

#fitting
if estimators=='SCH':
    #perform e fit
    fitID = 0
    par, cov = curve_fit(f_sigma_fit,results['Radius'][fitID:],results['AvgE_Q1'][fitID:],sigma=results['AvgEErr_Q1'][fitID:],p0=[0.3, 1.0])
    e1 = par[0]
    const1 = par[1]
    
    #calculate model
    ax = plt.gca()
    xlims = ax.get_xlim()
    rPlot = np.logspace(np.log10(xlims[0]), np.log10(xlims[-1]), 100)
    sigma1 = f_sigma_fit(rPlot, e1, const1)
    
    #plot model
    plt.plot(rPlot, sigma1, 'k')
    
    plt.text(0.95,0.20,r'$e$: '+'{0:.2f}'.format(e1),fontsize=14,ha='right',va='bottom',transform=ax.transAxes)
elif estimators=='CJ':
    fitID = 1
    par, cov = curve_fit(E1plus_fit, results['Radius'][fitID:],results['AvgE1P'][fitID:],sigma=results['AvgE1Perr'][fitID:],p0=[0.3, 1.0])
    e1 = par[0]
    const1 = par[1]
    fitId = 0
    par, cov = curve_fit(E2plus_fit, results['Radius'][fitID:],results['AvgE2P'][fitID:],sigma=results['AvgE2Perr'][fitID:],p0=[0.3, 1.0])
    e2 = par[0]
    const2 = par[1]
    
    #calculate model
    ax = plt.gca()
    xlims = ax.get_xlim()
    rPlot = np.logspace(np.log10(xlims[0]), np.log10(xlims[-1]), 100)
    Sigma1plus = E1plus_fit(rPlot, e1, const1)
    Sigma2plus = E2plus_fit(rPlot, e2, const2)
    
    #plot model
    plt.plot(rPlot, Sigma1plus, 'k')
    plt.plot(rPlot, Sigma2plus, 'g')
    
    plt.text(0.95,0.70,r'$e_1$: '+'{0:.2f}'.format(e1),fontsize=14,ha='right',va='bottom',transform=ax.transAxes)
    plt.text(0.95,0.60,r'$e_2$: '+'{0:.2f}'.format(e2),fontsize=14,ha='right',va='bottom',transform=ax.transAxes)

ax = plt.gca()
xlims = ax.get_xlim()
plt.plot(xlims, [0,0], 'r--')
ax.set_xlim(xlims)

if estimators == 'SCH':
    plt.title(r'$f(r)\Delta\Sigma$', fontsize=14)
if estimators == 'CJ':
    plt.title('quadrupole shear', fontsize=14)
    plt.legend([p1,p2],(r'$\Delta\Sigma_1^{(+)}$',r'$\Delta\Sigma_2^{(+)}$'),fontsize=16)


plt.subplot(133) #plot quadrupole minus estimators

#plot data
if estimators == 'SCH':
    plt.errorbar(results['Radius'],results['AvgE_Q2'],yerr=results['AvgEErr_Q2'],fmt='ko')
elif estimators == 'CJ':
    p1=plt.errorbar(results['Radius'],results['AvgE1M'],yerr=results['AvgE1Merr'],fmt='ko')
    p2=plt.errorbar(results['Radius'],results['AvgE2M'],yerr=results['AvgE2Merr'],fmt='go')

#format plot and output
if h_factor:
    plt.xlabel(r'$Radial$ $Distance$ $(kpc/h)$', fontsize=14)
    plt.ylabel(r'$\langle\Sigma(R)\rangle$ $(M_{\odot}h/pc^2)$', fontsize=14)
else:    
    plt.xlabel(r'$Radial$ $Distance$ $(kpc)$', fontsize=14)
    plt.ylabel(r'$\langle\Sigma(R)\rangle$ $(M_{\odot}/pc^2)$', fontsize=14)
plt.xscale('log')

#fitting
if estimators=='SCH':
    fitID = 0
    par, cov = curve_fit(f45_sigma_fit,results['Radius'][fitID:],results['AvgE_Q2'][fitID:],sigma=results['AvgEErr_Q2'][fitID:],p0=[0.3, 0.0])
    e2 = par[0]
    const2 = par[1]
    
    
    #calculate model
    ax = plt.gca()
    xlims = ax.get_xlim()
    rPlot = np.logspace(np.log10(xlims[0]), np.log10(xlims[-1]), 100)
    sigma2 = f45_sigma_fit(rPlot, e2, const2)
    
    #plot model
    plt.plot(rPlot, sigma2, 'k')
    
    plt.text(0.95,0.20,r'$e$: '+'{0:.2f}'.format(e2),fontsize=14,ha='right',va='bottom',transform=ax.transAxes)
elif estimators=='CJ':
    fitID = 0
    par, cov = curve_fit(E1minus_fit, results['Radius'][fitID:],results['AvgE1M'][fitID:],sigma=results['AvgE1Merr'][fitID:],p0=[0.3, 1.0])
    e1 = par[0]
    const1 = par[1]
    fitId = 0
    par, cov = curve_fit(E2minus_fit, results['Radius'][fitID:],results['AvgE2M'][fitID:],sigma=results['AvgE2Merr'][fitID:],p0=[0.3, 1.0])
    e2 = par[0]
    const2 = par[1]
    
    #calculate model
    ax = plt.gca()
    xlims = ax.get_xlim()
    rPlot = np.logspace(np.log10(xlims[0]), np.log10(xlims[-1]), 100)
    Sigma1minus = E1minus_fit(rPlot, e1, const1)
    Sigma2minus = E2minus_fit(rPlot, e2, const2)
    
    #plot model
    plt.plot(rPlot, Sigma1minus, 'k')
    plt.plot(rPlot, Sigma2minus, 'g')
    
    plt.text(0.95,0.20,r'$e_1$: '+'{0:.2f}'.format(e1),fontsize=14,ha='right',va='bottom',transform=ax.transAxes)
    plt.text(0.95,0.10,r'$e_2$: '+'{0:.2f}'.format(e2),fontsize=14,ha='right',va='bottom',transform=ax.transAxes)

ax = plt.gca()
xlims = ax.get_xlim()
plt.plot(xlims, [0,0], 'r--')
ax.set_xlim(xlims)

if estimators == 'SCH':
    plt.title(r'$f_{45}(r)\Delta\Sigma$', fontsize=14)
if estimators == 'CJ':
    plt.title('quadrupole shear', fontsize=14)
    plt.legend([p1,p2],(r'$\Delta\Sigma_1^{(-)}$',r'$\Delta\Sigma_2^{(-)}$'),fontsize=16)

#save plot
if estimators == 'SCH':
    plt.savefig(plot_path+'shearSCH',bbox_inches='tight')
if estimators == 'CJ':
    plt.savefig(plot_path+'shearCJ',bbox_inches='tight')  
