# pyScarpFit
This is a python program to perform a 1 event or steady state uplift diffusion model on a fault scarp. See Hanks (2000) or Hanks et al., 1984 for details. 

The most novel thing about this code is how it finds the midpoint of a scarp (essential for these models) and determines the upper and lower (which may be different) far-field slope of the scarp, and the vertical offset (throw) of the scarp. The actual diffusion parameters are solved by a simple grid search. 

(c) Sean G Polun, University of Missouri (2021)
