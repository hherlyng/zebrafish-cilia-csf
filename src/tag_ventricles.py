import dolfinx as dfx
from mpi4py import MPI
from imports.mesh import mark_boundaries_flow, create_ventricle_volumes_meshtags

with dfx.io.XDMFFile(MPI.COMM_WORLD, "geometries/fore_middle_hind_shrunk.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()

# ft = mark_boundaries_flow(mesh, True, modify=3)
ct, _ = create_ventricle_volumes_meshtags(mesh)

with dfx.io.XDMFFile(MPI.COMM_WORLD, "new_mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, mesh.geometry)