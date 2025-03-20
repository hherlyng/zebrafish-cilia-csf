import numpy.typing

import numpy     as np
import dolfinx   as dfx

from mpi4py   import MPI

# Mesh tags for flow
ANTERIOR_PRESSURE    = 2 # The pressure BC facets on the anterior ventricle boundary
POSTERIOR_PRESSURE   = 3 # The pressure BC facets on the posterior ventricle boundary
MIDDLE_VENTRAL_CILIA = 5 # The cilia BC facets on the ventral wall of the middle ventricle
MIDDLE_DORSAL_CILIA  = 6 # The cilia BC facets on the dorsal wall of the middle ventricle
ANTERIOR_CILIA1      = 7 # The cilia BC facets on the dorsal, anterior walls of the anterior ventricle
ANTERIOR_CILIA2      = 8 # The cilia BC facets on the dorsal, posterior walls of the anterior ventricle
SLIP                 = 9 # The free-slip facets of the boundary
DEFAULT = 10 # Default tag

def create_refinement_meshtags(mesh: dfx.mesh.Mesh):

    def ROI_1_refinement_bot(x):
        x_range = np.logical_and(x[0]>0.290, x[0]<0.320)
        y_range = np.logical_and(x[1]>0.130, x[1]<0.150)
        z_range = np.logical_and(x[2]>0.175, x[2]<0.195)
        xz_range = np.logical_and(x_range, z_range)
        return np.logical_and(y_range, xz_range)
    
    def ROI_1_refinement_top(x):
        x_range = np.logical_and(x[0]>0.290, x[0]<0.320)
        y_range = np.logical_and(x[1]>0.130, x[1]<0.150)
        z_range = np.logical_and(x[2]>0.220, x[2]<0.250)
        xz_range = np.logical_and(x_range, z_range)
        return np.logical_and(y_range, xz_range)

    def ROI_1_refinement_left(x):
        x_range = np.logical_and(x[0]>0.270, x[0]<0.290)
        y_range = np.logical_and(x[1]>0.130, x[1]<0.150)
        z_range = np.logical_and(x[2]>0.195, x[2]<0.220)
        xz_range = np.logical_and(x_range, z_range)
        return np.logical_and(y_range, xz_range)
    
    def ROI_1_refinement_right(x):
        x_range = np.logical_and(x[0]>0.320, x[0]<0.340)
        y_range = np.logical_and(x[1]>0.130, x[1]<0.150)
        z_range = np.logical_and(x[2]>0.195, x[2]<0.220)
        xz_range = np.logical_and(x_range, z_range)
        return np.logical_and(y_range, xz_range)
    
    tdim = mesh.topology.dim

    cells = {1 : dfx.mesh.locate_entities(mesh, tdim, ROI_1_refinement_bot),
                 2 : dfx.mesh.locate_entities(mesh, tdim, ROI_1_refinement_top),
                 3 : dfx.mesh.locate_entities(mesh, tdim, ROI_1_refinement_left),
                 4 : dfx.mesh.locate_entities(mesh, tdim, ROI_1_refinement_right),
    }
    num_volumes = mesh.topology.index_map(tdim).size_local \
                + mesh.topology.index_map(tdim).num_ghosts # Total number of volumes = local + ghosts
    tags = [1, 2, 3, 4]
    volume_marker = np.full(num_volumes, DEFAULT, dtype=np.int32)
    for i in reversed(tags): volume_marker[cells[i]] = tags[i-1] # Mark the cells in each ROI with the corresponding ROI tag

    return (dfx.mesh.meshtags(mesh, tdim, np.arange(num_volumes, dtype=np.int32), volume_marker), tags)

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
        x_range = np.logical_and(x[0]>0.290, x[0]<0.320) #295, 325
        y_range = np.logical_and(x[1]>0.135, x[1]<0.145) #145, 155
        z_range = np.logical_and(x[2]>0.195, x[2]<0.220) #200, 225
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
        y_range = np.logical_and(x[1]>0.075, x[1]<0.225)
        z_range = np.logical_and(x[2] > 0.110, x[2] < 0.135)
        xz_range = np.logical_and(x_range, z_range)
        return np.logical_and(y_range, xz_range)

    def ROI_4(x):
        """ The entire middle ventricle. """
        x_range = np.logical_and(x[0]>line_ROI4_start(x[2]), x[0]<line_ROI4_end(x[2]))
        y_range = np.logical_and(x[1]>0.075, x[1]<0.225)
        z_range = x[2] > 0.5*x[0] - 0.03
        xz_range = np.logical_and(x_range, z_range)
        return np.logical_and(y_range, xz_range)

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

    ROI_cells = {1 : dfx.mesh.locate_entities(mesh, tdim, ROI_1),
                 2 : dfx.mesh.locate_entities(mesh, tdim, ROI_2),
                 3 : dfx.mesh.locate_entities(mesh, tdim, ROI_3),
                 4 : dfx.mesh.locate_entities(mesh, tdim, ROI_4),
                 5 : dfx.mesh.locate_entities(mesh, tdim, ROI_5),
                 6 : dfx.mesh.locate_entities(mesh, tdim, ROI_6)
    }
    num_volumes = mesh.topology.index_map(tdim).size_local \
                + mesh.topology.index_map(tdim).num_ghosts # Total number of volumes = local + ghosts
    ROI_tags = [1, 2, 3, 4, 5, 6]
    volume_marker = np.full(num_volumes, DEFAULT, dtype=np.int32)
    for i in reversed(ROI_tags): volume_marker[ROI_cells[i]] = ROI_tags[i-1] # Mark the cells in each ROI with the corresponding ROI tag

    return (dfx.mesh.meshtags(mesh, tdim, np.arange(num_volumes, dtype=np.int32), volume_marker), ROI_tags)

def mark_facets(mesh: dfx.mesh.Mesh, inflow_outflow: bool) -> dfx.mesh.MeshTags:
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

    facet_dim = mesh.topology.dim-1 # Facets dimension

    # Generate mesh entities and connectivity between facets and cells
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim+1)

    num_facets   = mesh.topology.index_map(facet_dim).size_local \
                 + mesh.topology.index_map(facet_dim).num_ghosts # Number of facets = local + ghosts
    facet_marker = np.full(num_facets, DEFAULT, dtype=np.int32) # Default facet marker value
    
    # Facets of the boundary of the mesh where noslip condition is imposed
    boundary_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    
    # Default facet marker values = SLIP
    facet_marker[boundary_facets] = SLIP

    # Facets of the anterior ventricle boundary
    anterior_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, anterior_cilia_volume1)
    facet_marker[anterior_boundary_facets] = ANTERIOR_CILIA1
    anterior_boundary_facets2 = dfx.mesh.locate_entities_boundary(mesh, facet_dim, anterior_cilia_volume2)
    facet_marker[anterior_boundary_facets2] = ANTERIOR_CILIA2

    # Facets of the upper part of the middle ventricle boundary
    upper_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, middle_dorsal_cilia_volume)
    facet_marker[upper_boundary_facets] = MIDDLE_DORSAL_CILIA

    # Facets of the lower part of the middle ventricle boundary
    lower_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, middle_ventral_cilia_volume)
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
    return (x[0]>0.59)

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
    return np.logical_and(x[0]<0.10, x[2]<0.07)

def middle_ventral_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:

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

def middle_dorsal_cilia_volume(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:

    x_range1  = np.logical_and(x[0]>0.175, x[0]<0.230)
    z_range1  = (x[2]>0.215)
    xz_bool1 = np.logical_and(x_range1, z_range1)

    x_range2 = np.logical_and(x[0]>0.230, x[0]<0.305)
    z_range2 = (x[2]>0.23)
    xz_bool2 = np.logical_and(x_range2, z_range2)

    x_range4 = np.logical_and(x[0]>0.305, x[0]<0.335)
    z_range4 = (x[2]>0.20)
    xz_bool4 = np.logical_and(x_range4, z_range4)

    xz_range = np.logical_or(xz_bool1, np.logical_or(xz_bool2, xz_bool4))
    y_range  = np.logical_and(x[1] > 0.13, x[1] < 0.16825)

    xyz_bool = np.logical_and(xz_range, y_range)

    return xyz_bool

def anterior_cilia_volume1(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:

    x_range = np.logical_and(x[0]<0.0865, x[0]>0.05)
    xz_bool = np.logical_and(x_range, x[2]>0.1475)
    y_range = np.logical_and(x[1] > 0.12, x[1] < 0.1755)

    return np.logical_and(xz_bool, y_range)

def anterior_cilia_volume2(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
    x_range = np.logical_and(x[0]<0.0965, x[0]>=0.075)
    xz_bool = np.logical_and(x_range, x[2]>0.143)
    y_range = np.logical_and(x[1] > 0.12, x[1] < 0.1755)

    return np.logical_and(xz_bool, y_range)


###########################
#   VERIFICATION MESHES   #
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

        def left(x):   return np.isclose(x[0], 0.0)
        def right(x):  return np.isclose(x[0], 1.0)
        def bottom(x): return np.isclose(x[1], 0.0)
        def top(x):    return np.isclose(x[1], 1.0)

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

        def left(x):   return np.isclose(x[0], 0.0)
        def right(x):  return np.isclose(x[0], 1.0)
        def front(x):  return np.isclose(x[1], 0.0)
        def back(x):   return np.isclose(x[1], 1.0)
        def bottom(x): return np.isclose(x[2], 0.0)
        def top(x):    return np.isclose(x[2], 1.0)

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