
#script that performs the shear stacking using the shear_stacker.py class
#uses either Yang BCGs or BOSS LRGs as lenses
#uses CFIS galaxies as sources
#outputs monopole and quadrupole shear in a table
#uses either Schrabback or CJ estimators

# a version that uses a simulated single-lens system

#import relevant modules
import numpy as np
import astropy.table as table
from shear_stacker import Stacker
import time
import json
import argparse

#command line arguments
parser = argparse.ArgumentParser(description='Stack shears',
                                 
                                 epilog='Parameters in file \n' +                       
                                        '(parameter) \t (type) \t (default) \t\t (description) \n' +
                                        'estimators \t string \t CJ \t\t\t estimators to use: either CJ or SCH \n' +
                                        'r_inner \t float \t\t 50 \t\t\t inner radius of radial bins \n' +
                                        'r_outer \t float \t\t 4000 \t\t\t outer radius of radial bins \n' +
                                        'num_bins \t float \t\t 7 \t\t\t number of radial bins \n' +
                                        'h_factor \t Boolean \t true \t\t\t whether or not to use units with h \n' +
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


json_file_path = 'parameters.json'


#read hyper parameters from json file
f = open(json_file_path, 'r')
params = json.load(f)
f.close()

estimators = params['estimators']
radii = np.logspace(np.log10(params['r_inner']), np.log10(params['r_outer']), params['num_bins']) #radial bins for stacking
h_factor = params['h_factor']
data_path = params['data_path']
source_path = params['source_path']
lens_path = params['lens_path']
output_path = params['output_path']

#prepare radial bins
h = 0.7
radii = radii/h #convert to kpc/h
rPlot = [(a + b) /2 for a,b in zip(radii[:-1], radii[1:])] #radial data point positions

start_time = time.time()

print('Starting...')
#read in lens data
lenses = table.Table.read(data_path + lens_path, format='ascii.csv')

#intialize counters
nl = len(lenses) #initialize number of lenses
nlq = len(lenses[lenses['A']!=lenses['B']]) #initialize number of quadrupole lenses

shear_stacker = Stacker(radii, estimators, data_path, source_path) #create Stacker object

for i in range(len(lenses)): #repeat for all lenses

    shear_stacker.stack_shear(lenses[i]) #perform stacking for this lens
    
    if args.verbose and i%100 == 0: #print out progress occasionally
        print('{0:.2f}'.format(100 * float(i)/len(lenses))+'%'+' - '+'{0:.3f}'.format(time.time()-start_time)+' s')

print(time.time()-start_time)

#final calculations and output
if estimators == 'SCH':
    ERAvgs, ERAvge, ERAvgsQ1, ERAvgeQ1, ERAvgsQ2, ERAvgeQ2, ns, npr = shear_stacker.return_stack()
    results = table.Table([rPlot,ERAvgs,ERAvge,ERAvgsQ1,ERAvgeQ1,ERAvgsQ2,ERAvgeQ2],
                           names=('Radius','AvgE','AvgEErr','AvgE_Q1','AvgEErr_Q1' ,'AvgE_Q2','AvgEErr_Q2'))
if estimators == 'CJ':
    ERAvgs,ERAvge,ERAvgs1P,ERAvge1P,ERAvgs1M,ERAvge1M,ERAvgs2P,ERAvge2P,ERAvgs2M,ERAvge2M,ns,npr = shear_stacker.return_stack()
    results = table.Table([rPlot,ERAvgs,ERAvge,ERAvgs1P,ERAvge1P,ERAvgs1M,ERAvge1M,ERAvgs2P,ERAvge2P,ERAvgs2M,ERAvge2M],
    names=('Radius','AvgE','AvgEErr','AvgE1P','AvgE1Perr','AvgE1M','AvgE1Merr','AvgE2P','AvgE2Perr','AvgE2M','AvgE2Merr'))

results.write(output_path+'shear'+estimators+'_output.csv', format='ascii.csv')


 






