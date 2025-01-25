import dolfinx as dfx
from mpi4py import MPI

with dfx.io.XDMFFile(MPI.COMM_WORLD, "./geometries/ventricles_0.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
print(f"Number of mesh cells: {mesh.topology.index_map(3).size_global}")
topo = mesh.topology
tdim = mesh.topology.dim
topo.create_connectivity(tdim, tdim-1)
topo.create_connectivity(tdim, tdim-2)
topo.create_connectivity(tdim-1, tdim)
topo.create_connectivity(tdim-1, tdim-2)
topo.create_connectivity(tdim-2, tdim-1)
topo.create_connectivity(tdim-2, tdim)
topo.create_connectivity(tdim-2, tdim-1)


bdry_facets = dfx.mesh.exterior_facet_indices(topo)
bdry_vertices = dfx.mesh.compute_incident_entities(topo, bdry_facets, tdim-1, tdim-2)

mesh = dfx.mesh.refine(mesh=mesh, edges=bdry_vertices)

with dfx.io.XDMFFile(MPI.COMM_WORLD, "./BL_refinement.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
print(f"Number of mesh cells in refined mesh: {mesh.topology.index_map(3).size_global}")