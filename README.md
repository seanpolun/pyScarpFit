# pyScarpFit
This is a python program to perform a 1 event or steady state uplift diffusion model on a fault scarp. See Hanks (2000) or Hanks et al., 1984 for details. 

The most novel thing about this code is how it finds the midpoint of a scarp (essential for these models) and determines the upper and lower (which may be different) far-field slope of the scarp, and the vertical offset (throw) of the scarp. The actual diffusion parameters are solved by a simple grid search. 

## Installation
Note: this program uses numba to accelerate computations, and the functionality of numba-scipy is essential. Create a fresh conda environment, and first: 
```
conda install -c numba numba-scipy
```
then, install the remaining dependencies in requirements.txt. Do not allow conda to downgrade numba-scipy. 

Install this code using: 
```
python -m pip install git+https://github.com/seanpolun/pyScarpFit.git
```
(c) Sean G Polun, University of Missouri (2021)
