# graduate-research
Measurement of elliptical dark matter halos from weak gravitational lensing

A sample of scripts and plots from my graduate research. The goal of this project is to use weak lensing to detect and measure dark matter halo ellipticity. Dark matter clusters around galaxies in halos that are often assumed to be spherical. However, we expect there to be a significant amount of anisotropy in the distribution of dark matter. Dark matter halos are interacted upon by graviational forces which may perturb them into an elliptical shape.

In this project, use weak lensing to observe the distribution of dark matter around galaxies. Dark matter is difficult to detect. It doesn't interact with electromagnetic forces, meaning it cannot be observed directly. However, we can statistically detect dark matter using gravitational lensing. The graviational forces of a massive object will bend and distort the background light of the objects behind it. For a round halo, this will result in a tangential shear around the center of the mass distribution. However, this effect is insignificant for a single galaxy. Instead we stack hundred of thousands of foreground galaxies. If we assume there is no prefered orientation for background galaxies, a statistically significant detection of tangential alignment reveals the foreground mass distribution. This can be extended further to study the shape of dark matter halos in general.

In the folder labelled 'data' I have included simulated data of sources (background galaxies) and lenses (background galaxies). In this project, I am using data from the Canada-France Imaging Survey (CFIS) which covers over 5000 square degrees on the sky and contains hundreds of millions of objects. The small subset of this data that I could upload to github wouldn't reveal any statistically significant results. Instead I have constructed simulated data of an ideal elliptical dark matter halo to demonstrate how the scripts work and the expected results.

The 'results' folder contains data that is produced by some of the scripts and used by the plotting scripts.

The 'plots' folder contains results from the potting scripts using the 'results' folder. The plots 'shearCJ.png' and 'shearSCH.png' contain radial density profiles of the dark matter halo. The leftmost panel is the spherical monopole term, and the two other panels contain various estimators used for measuring the quadrupole term (which describes the ellipticity). The plot 'stickplot_gamma.png' displays the observed shear (stretching from lensing), a 1D monopole fit, a 2D monopole fit, and the residual from the quadrupole. The plot 'stickplot_kappa.png' displays the reconstructed mass map from the shear. The panels display the elliptical dark matter halo, a monopole fit, and the residual quadrupole mass distribution.

There are two main functions that the sample scripts in this repository perform. The first is creating the radial density plots, which is done using `shear_cfis.py` and `shear_plot.py` . The second function is creating the 2D shear and mass maps, which is done using `stickplot_table.py` and `stickplot_plot.py` .

To generate data used for creating the radial density plots, run

`python shear_cfis.py [-h] [-v]`

The script loops through the lens sample and identifies the appropriate background sources. For each lens, the sources and coordinate frame are rotated to align with the major axis of the lens, the shear is calculcated, and then stacked into radial bins. The stacking is done by a Stacker object which is described in `shear_stacker.py` . This script is controlled by parameters in the file 'parameters.json' . Running the script in help mode with `-h` in the command line will describe each of these parameters. Adding `-v` will run the script in verbose mode. This will display progress after every 100 lenses, but this is not necessary with the small simulated dataset.

To display these results, run

`python shear_plot.py [-h]`

This will plot the results from the radial shear stacking. A model will be fit to the data, which is described in `halomodel.py` . This script is controlled by the parameter file 'parameters.json' . Running the script in help mode with `-h` in the command line will describe each of these parameters.

To generate data used for creating 2D shear and mass maps, run

`python stickplot_table.py [-h] [-v]`

This script will perform the same shear stacking process as `shear_cfis.py` but the data will be binned in 2D instead of radially. This script is controlled by parameters in the file 'parameters2D.json' . Running the script in help mode with `-h` in the command line will describe each of these parameters. Adding `-v` will run the script in verbose mode. This will display progress after every 100 lenses, but this is not necessary with the small simulated dataset.

To display these results, run

`python stickplot_plot.py [-h]`

This will plot the 2D shear map and 2D mass map. A model will be fit to the data, which is described in `halomodel.py` . This script is controlled by the parameter file 'parameters2D.json' . Running the script in help mode with `-h` in the command line will describe each of these parameters.
