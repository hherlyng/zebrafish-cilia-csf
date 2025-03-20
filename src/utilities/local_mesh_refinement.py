import numpy   as np
import dolfinx as dfx

from mesh   import create_ventricle_volumes_meshtags, create_refinement_meshtags
from mpi4py import MPI

''' Refine mesh around the photoconversion site (region of interest 1 in the paper.)'''


if __name__=='__main__':
    input_mesh_filename = '../../geometries/standard/original_ventricles.xdmf'
    output_mesh_filename = input_mesh_filename.removesuffix('.xdmf') + '_refined.xdmf'

    # Read mesh
    with dfx.io.XDMFFile(MPI.COMM_WORLD, input_mesh_filename, 'r') as xdmf:
        mesh = xdmf.read_mesh()

    ct, tags = create_refinement_meshtags(mesh)
    edges = []
    tdim = mesh.topology.dim
    mesh.topology.create_entities(tdim-2)
    mesh.topology.create_connectivity(tdim, tdim-2)
    c_to_e = mesh.topology.connectivity(tdim, tdim-2)
    for tag in tags:
        for cell in ct.find(tag):
            [edges.append(edge) for edge in c_to_e.links(cell) if edge not in edges]
    ct, ROI_tags = create_ventricle_volumes_meshtags(mesh)

    for cell in ct.find(ROI_tags[0]):
        [edges.append(edge) for edge in c_to_e.links(cell) if edge not in edges]
    edges = np.array(edges, dtype=np.int32)

    print('Refining mesh ...', flush=True)
    refined_mesh, _, _ = dfx.mesh.refine(mesh, edges)

    # Write mesh
    with dfx.io.XDMFFile(MPI.COMM_WORLD, output_mesh_filename, 'w') as xdmf:
        xdmf.write_mesh(refined_mesh)