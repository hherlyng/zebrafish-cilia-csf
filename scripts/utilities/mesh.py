import ufl
import meshio
import numpy.typing

import numpy     as np
import dolfinx   as dfx

from mpi4py   import MPI
from pathlib  import Path

# Mesh tags
NOSLIP    = 1
INLET     = 2
OUTLET    = 3
VOLUME    = 4
LOWER     = 5
UPPER     = 6
ANTERIOR  = 7
ANTERIOR2 = 8
SLIP      = 9


# Mesh tags for flow
ANTERIOR_PRESSURE    = 2
POSTERIOR_PRESSURE   = 3
VOLUME               = 4
MIDDLE_VENTRAL_CILIA = 5
MIDDLE_DORSAL_CILIA  = 6
ANTERIOR_CILIA       = 7
ANTERIOR_CILIA2      = 8
SLIP                 = 9

# Mesh tags for transport
ANTERIOR_DORSAL         = 11
POSTERIOR_DORSAL        = 12
MIDDLE_DORSAL_POSTERIOR = 13
MIDDLE_DORSAL_ANTERIOR  = 14
POSTERIOR_VENTRAL       = 15
ANTERIOR_VENTRAL        = 16
MIDDLE_VENTRAL          = 17

def old_create_ventricle_volumes_meshtags(mesh: dfx.mesh.Mesh) -> tuple((dfx.mesh.MeshTags, list[int])):
    """ Create cell meshtags for the different regions of interest (ROI)
        used in the analysis and comparison with experimental data.

    Parameters
    ----------
    mesh : dfx.mesh.Mesh
        The brain ventricles mesh.

    Returns
    -------
    tuple(dfx.mesh.MeshTags, list[int])
        The celltags and a list of the integer tag values for each ROI.
    """
    
    # Define the planes that separate the anterior, middle and posterior ventricles
    z_min = mesh.comm.allreduce(mesh.geometry.x[:, 2].min(), op=MPI.MIN)
    z_max = mesh.comm.allreduce(mesh.geometry.x[:, 2].max(), op=MPI.MAX)
    z_mid = (z_min + z_max) / 2
    a1 = [0.165, 0.0, z_mid]
    b1 = [0.125, 0.0, z_max*0.78]

    a2 = [0.33, 0.0, z_mid*0.8]
    b2 = [0.345, 0.0, z_max*0.75]

    line1 = lambda t: a1[2] + (b1[2] - a1[2])/(b1[0] - a1[0])*(t - a1[0])
    line2 = lambda t: a2[2] + (b2[2] - a2[2])/(b2[0] - a2[0])*(t - a2[0])

    # Define locator functions for each region of interest (ROI)
    def ROI_1(x):
        """ The dorsal posterior region of the middle ventricle. """
        x_range = np.logical_and(x[0] > 0.295, x[0] < 0.325)
        z_range = np.logical_and(x[2] > 0.200, x[2] < 0.225)
        return np.logical_and(x_range, z_range)

    def ROI_2(x):
        """ The dorsal anterior region of the middle ventricle. """
        x_range = np.logical_and(x[0] > 0.135, x[0] < 0.165)
        z_range = np.logical_and(x[2] > 0.200, x[2] < 0.225)
        return np.logical_and(x_range, z_range)

    def ROI_3(x):
        """ The ventral posterior region of the middle ventricle. """
        x_range = np.logical_and(x[0] > 0.265, x[0] < 0.295)
        z_range = np.logical_and(x[2] > 0.110, x[2] < 0.135)
        return np.logical_and(x_range, z_range)

    def ROI_4(x):
        """ The entire middle ventricle. """
        return np.logical_and(x[2] >= line1(x[0]), x[2] >= line2(x[0]))

    def ROI_5(x):
        """ The anterior ventricle. """
        return x[2] < line1(x[0])

    def ROI_6(x):
        """ The posterior ventricle. """
        return x[2] < line2(x[0])


    tdim = mesh.topology.dim
    ## TODO: update to using midpoint coordinate of cells located
    ROI_cells = {1 : dfx.mesh.locate_entities(mesh, tdim, ROI_1),
                 2 : dfx.mesh.locate_entities(mesh, tdim, ROI_2),
                 3 : dfx.mesh.locate_entities(mesh, tdim, ROI_3),
                 4 : dfx.mesh.locate_entities(mesh, tdim, ROI_4),
                 5 : dfx.mesh.locate_entities(mesh, tdim, ROI_5),
                 6 : dfx.mesh.locate_entities(mesh, tdim, ROI_6)
    }
    num_volumes   = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts # Total number of volumes
    DEFAULT  = 9 # default cell tag value
    ROI_tags = [1, 2, 3, 4, 5, 6]
    volume_marker = np.full(num_volumes, DEFAULT, dtype=np.int32)
    for i in reversed(ROI_tags): volume_marker[ROI_cells[i]] = ROI_tags[i-1] # Mark the cells in each ROI with the corresponding ROI tag

    return (dfx.mesh.meshtags(mesh, tdim, np.arange(num_volumes, dtype=np.int32), volume_marker), ROI_tags)

def create_ventricle_volumes_meshtags(mesh: dfx.mesh.Mesh) -> tuple((dfx.mesh.MeshTags, list[int])):
    """ Create cell meshtags for the different regions of interest (ROI)
        used in the analysis and comparison with experimental data.

    Parameters
    ----------
    mesh : dfx.mesh.Mesh
        The brain ventricles mesh.

    Returns
    -------
    tuple(dfx.mesh.MeshTags, list[int])
        The celltags and a list of the integer tag values for each ROI.
    """
    # Define regions of interest by separating planes

    line_ROI4_start = lambda t: -1.6*t + 0.45
    line_ROI4_end = lambda t: 0.1*t + 0.31

    line_ROI5_start_x = lambda t: -2*t + 0.25
    line_ROI5_end_x = lambda t: -1.6*t + 0.385
    line_ROI5_end_z = lambda t: 0.9*t + 0.1
    line_ROI5_start_z = lambda t: 1.2*t - 0.035
    
    line_ROI6_start_x = lambda t: -0.1*t + 0.375
    line_ROI6_end_x = lambda t: 1.0*t + 0.42
    line_ROI6_start_z = lambda t: -0.3*t + 0.221
    line_ROI6_end_z = lambda t: -0.6*t + 0.43 

    # Define locator functions for each region of interest (ROI)
    def ROI_1(x):
        """ The dorsal posterior region of the middle ventricle. """
        x_range = np.logical_and(x[0]>0.295, x[0]<0.325)
        y_range = np.logical_and(x[1]>0.145, x[1]<0.155)
        z_range = np.logical_and(x[2]>0.200, x[2]<0.225)
        xz_range = np.logical_and(x_range, z_range)
        return np.logical_and(y_range, xz_range)

    def ROI_2(x):
        """ The dorsal anterior region of the middle ventricle. """
        x_range = np.logical_and(x[0] > 0.135, x[0] < 0.165)
        z_range = np.logical_and(x[2] > 0.200, x[2] < 0.225)
        return np.logical_and(x_range, z_range)

    def ROI_3(x):
        """ The ventral posterior region of the middle ventricle. """
        x_range = np.logical_and(x[0] > 0.265, x[0] < 0.295)
        z_range = np.logical_and(x[2] > 0.110, x[2] < 0.135)
        return np.logical_and(x_range, z_range)

    def ROI_4(x):
        """ The entire middle ventricle. """
        x_range = np.logical_and(x[0]>line_ROI4_start(x[2]), x[0]<line_ROI4_end(x[2]))
        z_range = x[2] > 0.5*x[0] - 0.03
        return np.logical_and(x_range, z_range)

    def ROI_5(x):
        """ The anterior ventricle. """
        x_range = np.logical_and(x[0]>line_ROI5_start_x(x[2]), x[0]<line_ROI5_end_x(x[2]))
        z_range = np.logical_and(x[2]>line_ROI5_start_z(x[0]), x[2]<line_ROI5_end_z(x[0]))
        return np.logical_and(x_range, z_range)

    def ROI_6(x):
        """ The posterior ventricle. """
        x_range = np.logical_and(x[0]>line_ROI6_start_x(x[2]), x[0]<line_ROI6_end_x(x[2]))
        z_range1 = np.logical_and(x[2]>line_ROI6_start_z(x[0]), x[2]<line_ROI6_end_z(x[0]))
        z_range2 = x[2] < 0.3*x[0] + 0.06
        z_range = np.logical_and(z_range1, z_range2)
        return np.logical_and(x_range, z_range)


    tdim = mesh.topology.dim
    ## TODO: update to using midpoint coordinate of cells located
    ROI_cells = {1 : dfx.mesh.locate_entities(mesh, tdim, ROI_1),
                 2 : dfx.mesh.locate_entities(mesh, tdim, ROI_2),
                 3 : dfx.mesh.locate_entities(mesh, tdim, ROI_3),
                 4 : dfx.mesh.locate_entities(mesh, tdim, ROI_4),
                 5 : dfx.mesh.locate_entities(mesh, tdim, ROI_5),
                 6 : dfx.mesh.locate_entities(mesh, tdim, ROI_6)
    }
    num_volumes   = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts # Total number of volumes
    DEFAULT  = 9 # default cell tag value
    ROI_tags = [1, 2, 3, 4, 5, 6]
    volume_marker = np.full(num_volumes, DEFAULT, dtype=np.int32)
    for i in reversed(ROI_tags): volume_marker[ROI_cells[i]] = ROI_tags[i-1] # Mark the cells in each ROI with the corresponding ROI tag

    return (dfx.mesh.meshtags(mesh, tdim, np.arange(num_volumes, dtype=np.int32), volume_marker), ROI_tags)

def read_msh_parallel(filename: str|Path, comm: MPI.Comm, rank: int=0, num_refinements: int=0) -> dfx.mesh.Mesh:
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

    if comm.rank == rank:
        msh = meshio.read(filename)
        # Extract cells and points from the .msh mesh
        cells  = msh.get_cells_type("tetra")
        points = msh.points
        points /= 1000  # Divide geometry by thousand to get millimeters
    
    else:
        cells  = np.empty((0, 4), dtype=np.int64)
        points = np.empty((0, 3), dtype=np.float64)
    element = ufl.VectorElement("Lagrange", ufl.tetrahedron, 1)
    domain = ufl.Mesh(element)
    mesh = dfx.mesh.create_mesh(comm, cells, points, domain, partitioner=dfx.cpp.mesh.create_cell_partitioner(dfx.mesh.GhostMode.shared_facet))
    if comm.rank == rank: print("Total # cells in mesh: {0}.\n".format(mesh.topology.index_map(3).size_global))

    if num_refinements != 0:
        print(f"Refining the mesh {num_refinements} times.")
        for _ in range(num_refinements):
            mesh.topology.create_entities(1)
            mesh = dfx.mesh.refine(mesh)
    
        if comm.rank==rank: print("Total # cells in refined mesh: {0}.\n".format(mesh.topology.index_map(3).size_global))
    
    return mesh

## Mesh utility functions
def mark_boundaries_flow_and_transport(mesh: dfx.mesh.Mesh, inflow_outflow: bool) -> dfx.mesh.MeshTags:
    facet_dim = mesh.topology.dim - 1 # Facets dimension

    # Generate mesh topology
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim+1)

    num_facets   = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts # Total number of facets
    facet_marker = np.full(num_facets, VOLUME, dtype=np.int32) # Default facet marker value
    
    # Facets of the boundary of the mesh where noslip condition is imposed
    boundary_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    
    # Default facet marker values = SLIP
    facet_marker[boundary_facets] = SLIP

    # Facets of the anterior ventricle boundary
    anterior_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, anterior_cilia_volume)
    facet_marker[anterior_boundary_facets] = ANTERIOR_CILIA

    # Facets of the upper part of the middle ventricle boundary
    upper_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, middle_dorsal_cilia_volume)
    facet_marker[upper_boundary_facets] = MIDDLE_DORSAL_CILIA

    # Facets of the lower part of the middle ventricle boundary
    lower_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, middle_ventral_cilia_volume)
    facet_marker[lower_boundary_facets] = MIDDLE_VENTRAL_CILIA

    # Facets of the dorsal anterior ventricle boundary
    anterior_dorsal_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, anterior_dorsal_injection_site)
    facet_marker[anterior_dorsal_boundary_facets] = ANTERIOR_DORSAL

    # Facets of the dorsal posterior middle ventricle boundary
    middle_dorsal_posterior_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, middle_dorsal_posterior_injection_site)
    facet_marker[middle_dorsal_posterior_boundary_facets] = MIDDLE_DORSAL_POSTERIOR

    # Facets of the dorsal posterior ventricle boundary
    posterior_dorsal_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, posterior_dorsal_injection_site)
    facet_marker[posterior_dorsal_boundary_facets] = POSTERIOR_DORSAL

    if inflow_outflow:
        # Facets of the inlet boundary of the mesh
        inlet_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, anterior_pressure_boundary)
        facet_marker[inlet_boundary_facets] = ANTERIOR_PRESSURE # Tags for the inlet boundary

        # Facets of the outlet boundary of the mesh
        outlet_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, posterior_pressure_boundary)
        facet_marker[outlet_boundary_facets] = POSTERIOR_PRESSURE # Tags for the inlet boundary

    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype=np.int32), facet_marker)

    return facet_tags

def mark_boundaries_flow(mesh: dfx.mesh.Mesh, inflow_outflow: bool, modify: int=0) -> dfx.mesh.MeshTags:
    """ Function that marks the boundaries of the mesh with integer tags and returns them as dolfinx meshtags.
        The markings are used for flow simulations.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh object
        A mesh object of the dolfinx Mesh class type

    inflow_outflow : bool
        If True, mark pressure BC regions for inflow/outflow of the boundary.

    Returns
    -------
    facet_tags
        Dolfinx meshtags for boundary facets
    """

    # Decide whether to use modified mesh markers
    markers = [anterior_cilia_volume, middle_dorsal_cilia_volume, middle_ventral_cilia_volume]
    if modify!=0:
        if modify in [1, 2, 3, 4]:
            # Shrunk ventricles
            markers = [shrunk_anterior_cilia_volume, shrunk_middle_dorsal_cilia_volume, shrunk_middle_ventral_cilia_volume]
        elif modify==5:
            # Modify middle-dorsal cilia
            markers[1] = mod_middle_dorsal_cilia_volume
        elif modify==6:
            # Modify middle-ventral cilia
            markers[2] = mod_middle_ventral_cilia_volume


    facet_dim = mesh.topology.dim - 1 # Facets dimension

    # Generate mesh entities
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim+1)

    num_facets   = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts # Total number of facets
    facet_marker = np.full(num_facets, VOLUME, dtype=np.int32) # Default facet marker value
    
    # Facets of the boundary of the mesh where noslip condition is imposed
    boundary_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    
    # Default facet marker values = SLIP
    facet_marker[boundary_facets] = SLIP

    # Facets of the anterior ventricle boundary
    anterior_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, markers[0])
    facet_marker[anterior_boundary_facets] = ANTERIOR_CILIA

    # Facets of the upper part of the middle ventricle boundary
    upper_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, markers[1])
    facet_marker[upper_boundary_facets] = MIDDLE_DORSAL_CILIA

    # Facets of the lower part of the middle ventricle boundary
    lower_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, markers[2])
    facet_marker[lower_boundary_facets] = MIDDLE_VENTRAL_CILIA

    if inflow_outflow:
        # Facets of the inlet boundary of the mesh
        inlet_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, anterior_pressure_boundary)
        facet_marker[inlet_boundary_facets] = ANTERIOR_PRESSURE # Tags for the inlet boundary

        # Facets of the outlet boundary of the mesh
        outlet_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, posterior_pressure_boundary)
        facet_marker[outlet_boundary_facets] = POSTERIOR_PRESSURE # Tags for the inlet boundary

    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype=np.int32), facet_marker)

    return facet_tags

def recal_mark_boundaries_flow(mesh: dfx.mesh.Mesh, inflow_outflow: bool, modify: int=0) -> dfx.mesh.MeshTags:
    """ Function that marks the boundaries of the mesh with integer tags and returns them as dolfinx meshtags.
        The markings are used for flow simulations.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh object
        A mesh object of the dolfinx Mesh class type

    inflow_outflow : bool
        If True, mark pressure BC regions for inflow/outflow of the boundary.

    Returns
    -------
    facet_tags
        Dolfinx meshtags for boundary facets
    """

    # Decide whether to use modified mesh markers
    markers = [recal_anterior_cilia_volume, recal_anterior_cilia_volume2, recal_middle_dorsal_cilia_volume, recal_middle_ventral_cilia_volume]


    facet_dim = mesh.topology.dim-1 # Facets dimension

    # Generate mesh entities
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim+1)

    num_facets   = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts # Total number of facets
    facet_marker = np.full(num_facets, VOLUME, dtype=np.int32) # Default facet marker value
    
    # Facets of the boundary of the mesh where noslip condition is imposed
    boundary_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    
    # Default facet marker values = SLIP
    facet_marker[boundary_facets] = SLIP

    # Facets of the anterior ventricle boundary
    anterior_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, markers[0])
    facet_marker[anterior_boundary_facets] = ANTERIOR_CILIA
    anterior_boundary_facets2 = dfx.mesh.locate_entities_boundary(mesh, facet_dim, markers[1])
    facet_marker[anterior_boundary_facets2] = ANTERIOR_CILIA2

    # Facets of the upper part of the middle ventricle boundary
    upper_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, markers[2])
    facet_marker[upper_boundary_facets] = MIDDLE_DORSAL_CILIA

    # Facets of the lower part of the middle ventricle boundary
    lower_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, markers[3])
    facet_marker[lower_boundary_facets] = MIDDLE_VENTRAL_CILIA

    if inflow_outflow:
        # Facets of the inlet boundary of the mesh
        inlet_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, anterior_pressure_boundary)
        facet_marker[inlet_boundary_facets] = ANTERIOR_PRESSURE # Tags for the inlet boundary

        # Facets of the outlet boundary of the mesh
        outlet_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, posterior_pressure_boundary)
        facet_marker[outlet_boundary_facets] = POSTERIOR_PRESSURE # Tags for the inlet boundary

    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype=np.int32), facet_marker)

    return facet_tags


def mark_boundaries_transport(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function that marks the boundaries of the mesh with integer tags and returns them as dolfinx meshtags.
        The markings are used for transport simulations.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh object
        A mesh object of the dolfinx Mesh class type

    Returns
    -------
    facet_tags
        Dolfinx meshtags for boundary facets
    """

    facet_dim = mesh.topology.dim - 1 # Facets dimension

    # Generate mesh topology
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim+1)

    num_facets   = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts # Total number of facets
    facet_marker = np.full(num_facets, VOLUME, dtype=np.int32) # Default facet marker value
    
    # Facets of the boundary of the mesh where noslip condition is imposed
    boundary_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    
    # Default facet marker values = SLIP
    facet_marker[boundary_facets] = SLIP

    # Facets of the dorsal anterior ventricle boundary
    anterior_dorsal_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, anterior_dorsal_injection_site)
    facet_marker[anterior_dorsal_boundary_facets] = ANTERIOR_DORSAL

    # Facets of the ventral anterior ventricle boundary
    anterior_ventral_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, anterior_ventral_injection_site)
    facet_marker[anterior_ventral_boundary_facets] = ANTERIOR_VENTRAL

    # Facets of the ventral middle ventricle boundary
    middle_ventral_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, middle_ventral_injection_site)
    facet_marker[middle_ventral_boundary_facets] = MIDDLE_VENTRAL

    # Facets of the dorsal anterior middle ventricle boundary
    middle_dorsal_anterior_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, middle_dorsal_anterior_injection_site)
    facet_marker[middle_dorsal_anterior_boundary_facets] = MIDDLE_DORSAL_ANTERIOR

    # Facets of the dorsal posterior middle ventricle boundary
    middle_dorsal_posterior_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, middle_dorsal_posterior_injection_site)
    facet_marker[middle_dorsal_posterior_boundary_facets] = MIDDLE_DORSAL_POSTERIOR

    # Facets of the dorsal posterior ventricle boundary
    posterior_dorsal_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, posterior_dorsal_injection_site)
    facet_marker[posterior_dorsal_boundary_facets] = POSTERIOR_DORSAL

    # Facets of the ventral posterior ventricle boundary
    posterior_ventral_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, posterior_ventral_injection_site)
    facet_marker[posterior_ventral_boundary_facets] = POSTERIOR_VENTRAL


    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype=np.int32), facet_marker)

    return facet_tags

def mark_volumes(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function that marks volume segments of the mesh with integer tags and returns them as dolfinx meshtags.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh object
        A mesh object of the dolfinx Mesh class type

    Returns
    -------
    volume_tags
        Dolfinx meshtags for volume segments
    """

    mesh_dim = mesh.topology.dim # Mesh dimension

    # Generate mesh topology
    mesh.topology.create_entities(mesh_dim)
    mesh.topology.create_entities(mesh_dim-1)
    mesh.topology.create_entities(mesh_dim-2)
    mesh.topology.create_connectivity(mesh_dim-1, mesh_dim)     # Connectivity facets   <-> cells
    mesh.topology.create_connectivity(mesh_dim-2, mesh_dim)     # Connectivity vertices <-> cells
    mesh.topology.create_connectivity(mesh_dim-1, mesh_dim-2) # Connectivity facets   <-> vertices

    num_volumes   = mesh.topology.index_map(mesh_dim).size_local + mesh.topology.index_map(mesh_dim).num_ghosts # Total number of volumes
    volume_marker = np.full(num_volumes, VOLUME, dtype=np.int32) # Default volume marker value

    boundary_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    bndry_vertices  = dfx.mesh.compute_incident_entities(mesh.topology, boundary_facets, mesh_dim-1, mesh_dim-2)
    bndry_cells     = dfx.mesh.compute_incident_entities(mesh.topology, bndry_vertices, mesh_dim-2, mesh_dim)

    ## Marking of volumes; only those that have a facet that is the boundary of the mesh
    # Volume segment in anterior ventricle
    anterior_cells = dfx.mesh.locate_entities(mesh, mesh_dim, anterior_cilia_volume)
    anterior_cell_indices = np.intersect1d(bndry_cells, anterior_cells)
    volume_marker[anterior_cell_indices] = ANTERIOR

    # Upper volume segment in middle ventricle
    lower_volumes = dfx.mesh.locate_entities(mesh, mesh_dim, middle_ventral_cilia_volume)
    lower_cell_indices = np.intersect1d(bndry_cells, lower_volumes)
    volume_marker[lower_cell_indices] = LOWER

    # Lower volume segment in middle ventricle
    upper_volumes = dfx.mesh.locate_entities(mesh, mesh_dim, middle_dorsal_cilia_volume)
    upper_cell_indices = np.intersect1d(bndry_cells, upper_volumes)
    volume_marker[upper_cell_indices] = UPPER

    # Create meshtags with volume tags
    volume_tags = dfx.mesh.meshtags(mesh, mesh_dim, np.arange(num_volumes, dtype=np.int32), volume_marker)

    return volume_tags

def posterior_pressure_boundary(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    """ Marker function for the boundary where pressure BCs are applied in the posterior ventricle.

    Parameters
    ----------
    x : array_like
        Spatial coordinates of the mesh.

    Returns
    -------
    array_like
        Boolean, True if coordinate is on boundary to be marked, else False.
    """
    return (x[0] > 0.59)

def anterior_pressure_boundary(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    """ Marker function for the boundary where pressure BCs are applied in the anterior ventricle.

    Parameters
    ----------
    x : array_like
        Spatial coordinates of the mesh.

    Returns
    -------
    array_like
        Boolean, True if coordinate is on boundary to be marked, else False.
    """
    return np.logical_and(x[0] < 0.10, x[2] < 0.07)

def recal_middle_ventral_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:

    x_range1  = np.logical_and(x[0] > 0.155, x[0] < 0.22)
    y_range1  = np.logical_and(x[1] > 0.11, x[1] < 0.18)
    z_range1  = (x[2] < 0.169)

    xy_bool1  = np.logical_and(x_range1, y_range1)
    xyz_bool1 = np.logical_and(xy_bool1, z_range1)

    x_range2  = np.logical_and(x[0] > 0.22, x[0] < 0.265)
    y_range2  = np.logical_and(x[1] > 0.09, x[1] < 0.2)
    z_range2  = (x[2] < 0.16)

    xy_bool2  = np.logical_and(x_range2, y_range2)
    xyz_bool2 = np.logical_and(xy_bool2, z_range2)

    x_range3  = np.logical_and(x[0] > 0.265, x[0] < 0.31)
    y_range3  = np.logical_and(x[1] > 0.12, x[1] < 0.165)
    z_range3  = (x[2] < 0.1375)

    xy_bool3  = np.logical_and(x_range3, y_range3)
    xyz_bool3 = np.logical_and(xy_bool3, z_range3)

    or1 = np.logical_or(xyz_bool1, xyz_bool2)

    return np.logical_or(or1, xyz_bool3)

def recal_middle_dorsal_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range3 = np.logical_and(x[0]>0.14, x[0]<0.165)
    z_range3 = (x[2]>0.21)
    xz_bool3 = np.logical_and(x_range3, z_range3)

    x_range1  = np.logical_and(x[0]>0.175, x[0]<0.230)
    z_range1  = (x[2]>0.215)
    xz_bool1 = np.logical_and(x_range1, z_range1)

    x_range2 = np.logical_and(x[0]>0.230, x[0]<0.305)
    z_range2 = (x[2]>0.23)
    xz_bool2 = np.logical_and(x_range2, z_range2)

    x_range4 = np.logical_and(x[0]>0.305, x[0]<0.335)
    z_range4 = (x[2]>0.20)
    xz_bool4 = np.logical_and(x_range4, z_range4)
    
    # xz_range1 = np.logical_or(xz_bool1, xz_bool2)
    # xz_range2 = np.logical_or(xz_bool3, xz_bool4)
    # xz_range = np.logical_or(xz_range1, xz_range2)
    xz_range = np.logical_or(xz_bool1, np.logical_or(xz_bool2, xz_bool4))
    y_range  = np.logical_and(x[1] > 0.13, x[1] < 0.16825)

    xyz_bool = np.logical_and(xz_range, y_range)

    return xyz_bool

def recal_anterior_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    # x_range = np.logical_and(x[0]<0.0965, x[0]>0.05)
    # xz_bool = np.logical_and(x_range, x[2]>0.1475)
    # y_range = np.logical_and(x[1]>-0.025, x[1]<0.025)

    x_range = np.logical_and(x[0]<0.0865, x[0]>0.05)
    xz_bool = np.logical_and(x_range, x[2]>0.1475)
    y_range = np.logical_and(x[1] > 0.12, x[1] < 0.1755)

    return np.logical_and(xz_bool, y_range)

def recal_anterior_cilia_volume2(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range = np.logical_and(x[0]<0.0965, x[0]>=0.075)
    xz_bool = np.logical_and(x_range, x[2]>0.143)
    y_range = np.logical_and(x[1] > 0.12, x[1] < 0.1755)

    return np.logical_and(xz_bool, y_range)



def middle_ventral_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:

    x_range1  = np.logical_and(x[0] > 0.135, x[0] < 0.22)
    y_range1  = np.logical_and(x[1] > 0.11, x[1] < 0.18)
    z_range1  = (x[2] < 0.1625)

    xy_bool1  = np.logical_and(x_range1, y_range1)
    xyz_bool1 = np.logical_and(xy_bool1, z_range1)

    x_range2  = np.logical_and(x[0] > 0.22, x[0] < 0.3)
    y_range2  = np.logical_and(x[1] > 0.09, x[1] < 0.2)
    z_range2  = (x[2] < 0.15)

    xy_bool2  = np.logical_and(x_range2, y_range2)
    xyz_bool2 = np.logical_and(xy_bool2, z_range2)

    x_range3  = np.logical_and(x[0] > 0.3, x[0] < 0.33)
    y_range3  = np.logical_and(x[1] > 0.12, x[1] < 0.165)
    z_range3  = (x[2] < 0.175)

    xy_bool3  = np.logical_and(x_range3, y_range3)
    xyz_bool3 = np.logical_and(xy_bool3, z_range3)

    or1 = np.logical_or(xyz_bool1, xyz_bool2)

    return np.logical_or(or1, xyz_bool3)

def mod_middle_ventral_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:

    x_range1  = np.logical_and(x[0] > 0.135, x[0] < 0.22)
    y_range1  = np.logical_and(x[1] > 0.11, x[1] < 0.18)
    z_range1  = (x[2] < 0.1625)

    xy_bool1  = np.logical_and(x_range1, y_range1)
    xyz_bool1 = np.logical_and(xy_bool1, z_range1)

    x_range2  = np.logical_and(x[0] > 0.22, x[0] < 0.29)
    y_range2  = np.logical_and(x[1] > 0.09, x[1] < 0.2)
    z_range2  = (x[2] < 0.15)

    xy_bool2  = np.logical_and(x_range2, y_range2)
    xyz_bool2 = np.logical_and(xy_bool2, z_range2)

    or1 = np.logical_or(xyz_bool1, xyz_bool2)

    return or1

def shrunk_middle_ventral_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:

    x_range1  = np.logical_and(x[0] > 0.137, x[0] < 0.22)
    y_range1  = np.logical_and(x[1] > 0.11, x[1] < 0.18)
    z_range1  = (x[2] < 0.1625)

    xy_bool1  = np.logical_and(x_range1, y_range1)
    xyz_bool1 = np.logical_and(xy_bool1, z_range1)

    x_range2  = np.logical_and(x[0] > 0.22, x[0] < 0.3)
    y_range2  = np.logical_and(x[1] > 0.09, x[1] < 0.2)
    z_range2  = (x[2] < 0.15)

    xy_bool2  = np.logical_and(x_range2, y_range2)
    xyz_bool2 = np.logical_and(xy_bool2, z_range2)

    x_range3  = np.logical_and(x[0] > 0.3, x[0] < 0.33)
    y_range3  = np.logical_and(x[1] > 0.12, x[1] < 0.165)
    z_range3  = (x[2] < 0.175)

    xy_bool3  = np.logical_and(x_range3, y_range3)
    xyz_bool3 = np.logical_and(xy_bool3, z_range3)

    or1 = np.logical_or(xyz_bool1, xyz_bool2)

    return np.logical_or(or1, xyz_bool3)

def middle_dorsal_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range  = np.logical_and(x[0] > 0.12075, x[0] < 0.38)
    y_range  = np.logical_and(x[1] > 0.13, x[1] < 0.16825)
    z_range  = (x[2] > 0.1975)

    xy_bool  = np.logical_and(x_range, y_range)
    xyz_bool = np.logical_and(xy_bool, z_range)

    return xyz_bool

def mod_middle_dorsal_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range  = np.logical_and(x[0] > 0.155, x[0] < 0.38)
    y_range  = np.logical_and(x[1] > 0.13, x[1] < 0.16825)
    z_range  = (x[2] > 0.1975)

    xy_bool  = np.logical_and(x_range, y_range)
    xyz_bool = np.logical_and(xy_bool, z_range)

    return xyz_bool

def shrunk_middle_dorsal_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range  = np.logical_and(x[0] > 0.131, x[0] < 0.38)
    y_range  = np.logical_and(x[1] > 0.13, x[1] < 0.16825)
    z_range  = (x[2] > 0.1975)

    xy_bool  = np.logical_and(x_range, y_range)
    xyz_bool = np.logical_and(xy_bool, z_range)

    return xyz_bool

def anterior_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range = np.logical_and(x[0] < 0.0965, x[0] > 0.05)
    xz_bool = np.logical_and(x_range, x[2] > 0.16)
    y_range = np.logical_and(x[1] > 0.12, x[1] < 0.1755)

    return np.logical_and(xz_bool, y_range)

def mod_anterior_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range = np.logical_and(x[0] < 0.0965, x[0] > 0.05)
    xz_bool = np.logical_and(x_range, x[2] > 0.16)
    y_range = np.logical_and(x[1] > 0.12, x[1] < 0.1755)

    return np.logical_and(xz_bool, y_range)

def shrunk_anterior_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range = np.logical_and(x[0] < 0.0955, x[0] > 0.05)
    xz_bool = np.logical_and(x_range, x[2] > 0.16)
    y_range = np.logical_and(x[1] > 0.12, x[1] < 0.1755)

    return np.logical_and(xz_bool, y_range)

def middle_ventral_injection_site(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range  = np.logical_and(x[0] > 0.15, x[0] < 0.2)
    y_range  = np.logical_and(x[1] > 0.135, x[1] < 0.155)
    z_range  = (x[2] < 0.17)

    xy_bool  = np.logical_and(x_range, y_range)
    xyz_bool = np.logical_and(xy_bool, z_range)

    return xyz_bool

def middle_dorsal_anterior_injection_site(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range  = np.logical_and(x[0] > 0.12, x[0] < 0.175)
    y_range  = np.logical_and(x[1] > 0.13, x[1] < 0.165)
    z_range  = (x[2] > 0.22)

    xy_bool  = np.logical_and(x_range, y_range)
    xyz_bool = np.logical_and(xy_bool, z_range)

    return xyz_bool

def middle_dorsal_posterior_injection_site(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range  = np.logical_and(x[0] > 0.3, x[0] < 0.375)
    y_range  = np.logical_and(x[1] > 0.13, x[1] < 0.17)
    z_range  = (x[2] > 0.22)

    xy_bool  = np.logical_and(x_range, y_range)
    xyz_bool = np.logical_and(xy_bool, z_range)

    return xyz_bool

def posterior_dorsal_injection_site(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range  = np.logical_and(x[0] > 0.42, x[0] < 0.48)
    y_range  = np.logical_and(x[1] > 0.13, x[1] < 0.16)
    z_range  = (x[2] > 0.12)

    xy_bool  = np.logical_and(x_range, y_range)
    xyz_bool = np.logical_and(xy_bool, z_range)

    return xyz_bool

def posterior_ventral_injection_site(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range  = np.logical_and(x[0] > 0.425, x[0] < 0.475)
    y_range  = np.logical_and(x[1] > 0.13, x[1] < 0.16)
    z_range  = (x[2] < 0.1)

    xy_bool  = np.logical_and(x_range, y_range)
    xyz_bool = np.logical_and(xy_bool, z_range)

    return xyz_bool

def anterior_dorsal_injection_site(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    return np.logical_and(x[0] < 0.1, x[2] > 0.1825)

def anterior_ventral_injection_site(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range = np.logical_and(x[0] > 0.05, x[0] < 0.1)
    y_range = np.logical_and(x[1] > 0.132, x[1] < 0.155)
    xy_bool = np.logical_and(x_range, y_range)
    return np.logical_and(xy_bool, x[2] < 0.07)


###########################
#       TEST MESHES       #
###########################
LEFT    = 1
RIGHT   = 2
FRONT   = 3
BACK    = 4
BOTTOM  = 5
TOP     = 6

def create_square_mesh_with_tags(N_cells: int) -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags)):
        mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, N_cells, N_cells,
                                           cell_type = dfx.mesh.CellType.triangle,
                                           ghost_mode = dfx.mesh.GhostMode.shared_facet)

        def left(x):
            return np.isclose(x[0], 0.0)
        
        def right(x):
            return np.isclose(x[0], 1.0)

        def bottom(x):
            return np.isclose(x[1], 0.0)

        def top(x):
            return np.isclose(x[1], 1.0)

        # Facet tags
        bc_facet_indices, bc_facet_markers = [], []
        fdim = mesh.topology.dim - 1

        inlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, left)
        bc_facet_indices.append(inlet_BC_facets)
        bc_facet_markers.append(np.full_like(inlet_BC_facets, LEFT))

        outlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, right)
        bc_facet_indices.append(outlet_BC_facets)
        bc_facet_markers.append(np.full_like(outlet_BC_facets, RIGHT))

        bottom_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, bottom)
        bc_facet_indices.append(bottom_BC_facets)
        bc_facet_markers.append(np.full_like(bottom_BC_facets, BOTTOM))

        top_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, top)
        bc_facet_indices.append(top_BC_facets)
        bc_facet_markers.append(np.full_like(top_BC_facets, TOP))

        bc_facet_indices = np.hstack(bc_facet_indices).astype(np.int32)
        bc_facet_markers = np.hstack(bc_facet_markers).astype(np.int32)

        sorted_facets = np.argsort(bc_facet_indices)

        facet_tags = dfx.mesh.meshtags(mesh, fdim, bc_facet_indices[sorted_facets], bc_facet_markers[sorted_facets])

        return mesh, facet_tags

def create_cube_mesh_with_tags(N_cells: int) -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags)):
        mesh = dfx.mesh.create_unit_cube(MPI.COMM_WORLD, N_cells, N_cells, N_cells,
                                           cell_type = dfx.mesh.CellType.tetrahedron,
                                           ghost_mode = dfx.mesh.GhostMode.shared_facet)

        def left(x):
            return np.isclose(x[0], 0.0)
        
        def right(x):
            return np.isclose(x[0], 1.0)

        def front(x):
            return np.isclose(x[1], 0.0)

        def back(x):
            return np.isclose(x[1], 1.0)

        def bottom(x):
            return np.isclose(x[2], 0.0)

        def top(x):
            return np.isclose(x[2], 1.0)

        # Facet tags
        bc_facet_indices, bc_facet_markers = [], []
        fdim = mesh.topology.dim - 1

        inlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, left)
        bc_facet_indices.append(inlet_BC_facets)
        bc_facet_markers.append(np.full_like(inlet_BC_facets, LEFT))

        outlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, right)
        bc_facet_indices.append(outlet_BC_facets)
        bc_facet_markers.append(np.full_like(outlet_BC_facets, RIGHT))

        front_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, front)
        bc_facet_indices.append(front_BC_facets)
        bc_facet_markers.append(np.full_like(front_BC_facets, FRONT))

        back_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, back)
        bc_facet_indices.append(back_BC_facets)
        bc_facet_markers.append(np.full_like(back_BC_facets, BACK))

        bottom_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, bottom)
        bc_facet_indices.append(bottom_BC_facets)
        bc_facet_markers.append(np.full_like(bottom_BC_facets, BOTTOM))

        top_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, top)
        bc_facet_indices.append(top_BC_facets)
        bc_facet_markers.append(np.full_like(top_BC_facets, TOP))

        bc_facet_indices = np.hstack(bc_facet_indices).astype(np.int32)
        bc_facet_markers = np.hstack(bc_facet_markers).astype(np.int32)

        sorted_facets = np.argsort(bc_facet_indices)

        facet_tags = dfx.mesh.meshtags(mesh, fdim, bc_facet_indices[sorted_facets], bc_facet_markers[sorted_facets])

        return mesh, facet_tags