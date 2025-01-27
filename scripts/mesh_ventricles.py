import dolfinx as dfx

from utilities.mesh import read_msh_parallel
from mpi4py import MPI

# for i in range(9):
i=10
filename = './geometries/ventricles_' + str(i) + '.msh'
mesh = read_msh_parallel(filename=filename, comm=MPI.COMM_WORLD)

with dfx.io.XDMFFile(MPI.COMM_WORLD, filename.removesuffix('.msh') + '.xdmf', 'w') as xdmf:
    xdmf.write_mesh(mesh=mesh)