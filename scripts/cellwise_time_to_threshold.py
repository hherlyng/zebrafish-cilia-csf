import numpy as np
import dolfinx as dfx
import adios4dolfinx as a4d

from mpi4py import MPI
from basix.ufl import element


k = 1 # Finite element degree of concentration data
comm = MPI.COMM_WORLD # MPI communicator
gm = dfx.mesh.GhostMode.shared_facet # Mesh MPI ghost mode

# Model version and filename
model_version = 'C'
input_filename = f"/Users/hherlyng/zebrafish-cilia-csf/src/output/transport/model_{model_version}_D3_DG1_pressureBC/injection_site_middle_dorsal_posterior/checkpoints/concentration/"

# Load mesh
mesh = a4d.read_mesh(comm=comm, filename=input_filename, engine="BP4", ghost_mode=gm)

# Create finite element function in DG_k space
el_k = element("DG", mesh.basix_cell(), k)
DG_k = dfx.fem.functionspace(mesh, el_k)

f_k = dfx.fem.Function(DG_k) # Finite element function for reading in the concentration
t_hat = dfx.fem.Function(DG_k) # Finite element function for storing the times to threshold
t_hat.x.array[:] = -1 # Initialize all t_hats as -1

# Load DGk file
times = np.arange(1, 38001) # The timesteps considered
threshold = 0.25 # The threshold for assigning times as t_hat

# Loop over all timesteps
for time in times:
    print(f"Time = {time}", flush=True)
    a4d.read_function(filename=input_filename, u=f_k, engine="BP4", time=time) # Read concentration field
    idx_set = np.where(f_k.x.array > threshold)[0] # Index to dofs of f_k that are above the threshold
    below = np.where(t_hat.x.array == -1)[0] # Index to dofs of t_hat that have not yet been assigned a time
    above_idx = np.intersect1d(idx_set, below) # The intersection of the above two, which are the dofs where t_hat should be assigned as current timestep
    t_hat.x.array[above_idx] = time
    
# Write output
with dfx.io.VTXWriter(comm, f"t_hat_model_{model_version}.bp", [t_hat], "BP4") as vtx: vtx.write(0) # VTX file (for visualization)
a4d.write_function_on_input_mesh(filename=f"t_hat_model_{model_version}", u=t_hat) # Checkpoint file (can be reloaded later)