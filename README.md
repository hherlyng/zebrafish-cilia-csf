# zebrafish-cilia-csf
Modeling of cerebrospinal fluid (CSF) flow and solute transport in zebrafish brain ventricles.
CSF flow is modeled by the Stokes equations, and solute transport is modeled by an advection-diffusion
equation. The equations are discretized and solved numerically with finite element methods using
DOLFINx.

The Stokes equations are discretized with Brezzi-Douglas-Marini elements for the velocity and
discontinuous Galerkin elements for the pressure. The advection-diffusion equation is solved
with a discontinuous Galerkin method. See the paper for details.

## Getting started
Begin by installing the required dependencies using the `environment.yml` file. This file
can be used to create a `conda` environment by running
```
conda env create -f environment.yml
```
in a terminal. This installs all of the dependencies in a `conda` environment named
'zfish-cilia-csf-env'. The dependencies can alternatively be installed via `spack`.

## Test the setup by running the verification code
The script `simulate_mms_verification.py` is used to run convergence tests for the 
Stokes equations discretization. Test your setup by running the script on the 
cylinder meshes:
```
python simulate_mms_verification.py
```
The printed tables of error norms and convergence rates should
match those in table X of the paper, i.e., a linear convergence rate in the 
$H^1$ semi-norm for the velocity error and linear convergence rate in the
$L^2$ norm for the pressure error.