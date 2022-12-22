Code for FUV to MIR Spectroscopic R(V) Relationship paper
=========================================================

Routines for FUV to MIR spectroscopic R(V) depedent relationship for
Milky Way dust extinction work.
This paper is Gordon, et al. (in prep).

In Development!
---------------

Active development.
Data still in changing.
Use at your own risk.

Contributors
------------
Karl Gordon

License
-------

This code is licensed under a 3-clause BSD style license (see the
``LICENSE`` file).

Data
----

Original data processed to be in a uniform format and have values and uncertainties
for E(B-V), A(V), and R(V) using Utils/process_(gordon09, fitzpatrick19, gordon21).py.

Linear of 1/R(V) versus A(lambda)/A(V): python Figs/fit_irv.py

Results of different fitting functions/methods: python Figs/plot_diff_fits.py datafile
   datafile from plot_rep_waves.py

Plot R(V) relationships (including the one derived in this paper) with measured
extinction curves for a narrow range of R(V) values: python Figs/plot_rep_curves.py

Figures
-------

1. A(V) versus R(V): python Figs/plot_samp_prop.py
  - also computes n_sightlines, A(V), R(V), and sample overlaps

2. 1/R(V) relationship at representative wavelengths: python Figs/plot_rep_waves.py

3. Example 2DCORR fitting technique: python Figs/plot_fitting_example.py

4. Fit parameters versus 1/R(V) in the UV: python Figs/plot_irv_params_waveregion.py --wavereg=uv

5. Fit parameters versus 1/R(V) in the optical: python Figs/plot_irv_params_waveregion.py --wavereg=opt

6. Fit parameters versus 1/R(V) in the IR: python Figs/plot_irv_params_waveregion.py --wavereg=ir

7. Fit parameters versus 1/R(V) from UV to IR: python Figs/plot_irv_params_waveregion.py --wavereg=all

8. Comparison of R(V) relationship to previous work: python Figs/plot_select_rv.py

9. Example deviate curves: python Figs/plot_deviate_curves.py
