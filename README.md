# Modeling of cerebrospinal fluid (CSF) flow and solute transport in zebrafish brain ventricles
CSF flow is modeled by the Stokes equations, and solute transport is modeled by an advection-diffusion
equation. The equations are discretized and solved numerically with finite element methods using
DOLFINx. The code is parallelized and uses MPI communcation.

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
match those in Table 3 of the paper, i.e., a linear convergence rate in the 
$H^1$ semi-norm for the velocity error and linear convergence rate in the
$L^2$ norm for the pressure error.

## Run CSF flow simulations
Flow simulations can be run on `N` processors with
```
mpirun -np N python simulate_flow.py mesh_version cilia_scenario
```
where `mesh_version` is an integer argument:
- 0 = original mesh
- 1 = fore_shrunk mesh
- 2 = middle_shrunk mesh
- 3 = hind_shrunk mesh
- 4 = fore_middle_hind shrunk mesh

The same goes for `cilia_scenario`:
- 0 = all cilia
- 1 = remove anterior cilia
- 2 = remove dorsal cilia
- 3 = remove ventral cilia

## Run solute transport simulations
Transport simulations can be run on `N` processors with
```
mpirun -np N python simulate_transport.py molecule mesh_version cilia_scenario
```
where `molecule` is an integer argument:
- 1 = $D_1$: diffusion coefficient of extracellular vesicles (150 nm radius)
- 2 = $D_2$: diffusion coefficient of Starmaker-Green Fluorescent Protein
- 3 = $D_3$: diffusion coefficient of Dendra2 Fluorescent Protein

The arguments `mesh_version` and `cilia_scenario` take the same inputs
as for the flow simulation script.