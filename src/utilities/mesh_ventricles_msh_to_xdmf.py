from mpi4py    import MPI
from pathlib   import Path

import ufl
import basix
import meshio
import numpy   as np
import dolfinx as dfx

def read_msh_parallel(filename: str | Path, comm: MPI.Comm, rank: int=0, num_refinements: int=0) -> dfx.mesh.Mesh:
    """ Read mesh file of type .msh and return a dolfinx.mesh.Mesh mesh.
        
        Possibly in parallel and possibly with refinements.

    Parameters
    ----------
    filename : str | Path
        Filename of .msh mesh as string or path.
    comm : MPI.Comm
        MPI communicator.
    rank : int, optional
        Rank owning mesh, by default 0
    num_refinements : int, optional
        Number of mesh refinements to perform, by default 0

    Returns
    -------
    mesh : dolfinx.mesh.Mesh

    """

    if comm.rank==rank:
        msh = meshio.read(filename)
        # Extract cells and points from the .msh mesh
        cells  = msh.get_cells_type("tetra")
        points = msh.points
        points /= 1000  # Divide geometry by thousand to get millimeters
    else:
        cells  = np.empty((0, 4), dtype=np.int64)
        points = np.empty((0, 3), dtype=np.float64)

    element = basix.ufl.element("Lagrange", basix.CellType.tetrahedron, 1, shape=(3,))
    domain = ufl.Mesh(element)
    partitioner = dfx.cpp.mesh.create_cell_partitioner(dfx.mesh.GhostMode.shared_facet)
    mesh = dfx.mesh.create_mesh(comm, cells, points, domain, partitioner=partitioner)
    if comm.rank==rank:
        print("Total # cells in mesh: {0}.\n".format(mesh.topology.index_map(3).size_global))

    if num_refinements!=0:
        print(f"Refining the mesh {num_refinements} times.")
        for _ in range(num_refinements):
            mesh.topology.create_entities(1)
            mesh = dfx.mesh.refine(mesh)
    
        if comm.rank==rank:
            print("Total # cells in refined mesh: {0}.\n".format(mesh.topology.index_map(3).size_global))
    
    return mesh

if __name__=='__main__':
    i=5
    filename = '../../geometries/ventricles/ventricles_' + str(i) + '.msh'
    mesh = read_msh_parallel(filename=filename, comm=MPI.COMM_WORLD)

    with dfx.io.XDMFFile(MPI.COMM_WORLD, filename.removesuffix('.msh') + '.xdmf', 'w') as xdmf:
        xdmf.write_mesh(mesh=mesh)