import dolfinx as dfx
import adios4dolfinx as a4d
from mpi4py.MPI import COMM_WORLD as comm
from basix.ufl import element

ghost_mode = dfx.mesh.GhostMode.shared_facet
file_A = '../output/checkpoints/transport/model_A/D1/concentration_inj=anterior_dorsal/'
file_C = '../output/checkpoints/transport/model_C/D1/concentration_inj=anterior_dorsal/'
mesh = a4d.read_mesh(comm, file_A, "BP4", ghost_mode)
el_dg = element("DG", mesh.basix_cell(), 1)
el_cg = element("Lagrange", mesh.basix_cell(), 1)
DG1 = dfx.fem.functionspace(mesh, el_dg)
P1  = dfx.fem.functionspace(mesh, el_cg)
c_A = dfx.fem.Function(DG1)
c_C = dfx.fem.Function(DG1)
c_diff = dfx.fem.Function(DG1)
c_out = dfx.fem.Function(P1)

read_time = 0
a4d.read_function(c_A, file_A, time=read_time)
a4d.read_function(c_C, file_C, time=read_time)
c_diff.x.array[:] = c_A.x.array.copy() - c_C.x.array.copy()
c_out.interpolate(c_diff)
xdmf_c_diff = dfx.io.XDMFFile(comm, 'c_diff.xdmf', 'w')
xdmf_c_diff.write_mesh(mesh)
xdmf_c_diff.write_function(c_out, read_time)
for _ in range(199):
    read_time += 1
    print(f"Time = {read_time}")
    a4d.read_function(c_A, file_A, time=read_time)
    a4d.read_function(c_C, file_C, time=read_time)
    c_diff.x.array[:] = c_A.x.array.copy() - c_C.x.array.copy()
    print(f"Maximum concentration difference: {c_diff.x.array.max()}")
    c_out.interpolate(c_diff)
    xdmf_c_diff.write_function(c_out, read_time)
xdmf_c_diff.close()