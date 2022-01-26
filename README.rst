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

Linear fits (using hyperfit) of 1/R(V) versus A(lambda)/A(V): python Utils/fit_irv.py

Figures
-------

1. A(V) versus R(V): python Figs/plot_samp_prop.py
  - also computes n_sightlines, A(V), R(V), and sample overlaps

2. 1/R(V) relationship at representative wavelengths: python Figs/plot_rep_waves.py

3. Fit parameters versus 1/R(V) in the UV: python Figs/plot_irv_params_waveregion.py --wavereg=uv

4. Fit parameters versus 1/R(V) in the UV: python Figs/plot_irv_params_waveregion.py --wavereg=opt

5. Fit parameters versus 1/R(V) in the UV: python Figs/plot_irv_params_waveregion.py --wavereg=ir

?. Fit parameters versus 1/R(V): python Figs/plot_irv_params.py


Tables
------

1.
