import gmsh

import numpy   as np
import dolfinx as dfx

from mpi4py import MPI

# Mesh tags
LEFT    = 1
RIGHT   = 2
FRONT   = 3
BACK    = 4
BOTTOM  = 5
TOP     = 6

# Rotated meshes
def create_rectangle_mesh_gmsh(L: int=1, H: int=1, res: float=0.1, theta: float=np.pi/4) \
                            -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags)):
    """
    Create a rectangle of length L and height H, rotated theta degrees
    around origin.


    Parameters
    ----------
    L
        Length of the box
    H
        Height of the box
    res
        Mesh resolution (uniform)
    theta
        Rotation angle
    """
    gmsh.initialize()

    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.add("Rectangle")

        # Create square
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0,
                                                L, H)
        gmsh.model.occ.synchronize()

        # Find entity markers before rotation
        surfaces = gmsh.model.occ.getEntities(dim=1)

        left_wall   = []
        right_wall  = []
        bottom_wall = []
        top_wall    = []

        for surface in surfaces:
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            if np.allclose(com, [0, H/2, 0]):
                left_wall.append(surface[1])
            elif np.allclose(com, [L, H/2, 0]):
                right_wall.append(surface[1])
            elif np.isclose(com[1], 0):
                bottom_wall.append(surface[1])
            elif np.isclose(com[1], H):
                top_wall.append(surface[1])
        # Rotate channel theta degrees in the xy-plane
        gmsh.model.occ.rotate([(2, rectangle)], 0, 0, 0,
                              0, 0, 1, theta)
        gmsh.model.occ.synchronize()

        # Add physical markers
        gmsh.model.addPhysicalGroup(2, [rectangle], 1)
        gmsh.model.setPhysicalName(2, 1, "Fluid volume")
        gmsh.model.addPhysicalGroup(1, left_wall, LEFT)
        gmsh.model.setPhysicalName(1, LEFT, "Left wall")
        gmsh.model.addPhysicalGroup(1, right_wall, RIGHT)
        gmsh.model.setPhysicalName(1, RIGHT, "Right wall")
        gmsh.model.addPhysicalGroup(1, bottom_wall, BOTTOM)
        gmsh.model.setPhysicalName(1, BOTTOM, "Bottom wall")
        gmsh.model.addPhysicalGroup(1, top_wall, TOP)
        gmsh.model.setPhysicalName(1, TOP, "Top wall")

        # Set number of threads used for mesh
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", MPI.COMM_WORLD.size)

        # Set uniform mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

        # Generate mesh
        gmsh.model.mesh.generate(dim = 2)
    # Convert gmsh model to DOLFINx Mesh and meshtags
    mesh, _, ft = dfx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()

    return mesh, ft

def create_box_mesh_gmsh(L: int=1, W: int=1, H: int=1, res: float=0.025, theta: float=np.pi/4) \
                            -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags)):
    """
    Create a box of length L, width W and height H, rotated theta degrees
    around origin.


    Parameters
    ----------
    L
        Length of the box
    W
        Width of the box
    H
        Height of the box
    res
        Mesh resolution (uniform)
    theta
        Rotation angle
    """
    gmsh.initialize()
    
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.add("Box")

        # Create box
        box = gmsh.model.occ.addBox(0, 0, 0,
                                    L, W, H)
        gmsh.model.occ.synchronize()

        # Find entity markers before rotation
        surfaces = gmsh.model.occ.getEntities(dim = 2)

        left_wall   = []
        right_wall  = []
        front_wall  = []
        back_wall   = []
        bottom_wall = []
        top_wall    = []

        for surface in surfaces:
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            if np.allclose(com, [0, W/2, H/2]):
                left_wall.append(surface[1])
            elif np.allclose(com, [L, W/2, H/2]):
                right_wall.append(surface[1])
            elif np.allclose(com, [L/2, 0, H/2]):
                front_wall.append(surface[1])
            elif np.allclose(com, [L/2, W, H/2]):
                back_wall.append(surface[1])
            elif np.allclose(com, [L/2, W/2, 0]):
                bottom_wall.append(surface[1])
            elif np.allclose(com, [L/2, W/2, H]):
                top_wall.append(surface[1])

        # Rotate channel theta degrees in the xy-plane
        gmsh.model.occ.rotate([(3, box)], 0, 0, 0,
                              0, 1, 0, -theta)
        gmsh.model.occ.synchronize()

        # Add physical markers
        gmsh.model.addPhysicalGroup(3, [box], 1)
        gmsh.model.setPhysicalName(3, 1, "Fluid volume")
        gmsh.model.addPhysicalGroup(2, left_wall, LEFT)
        gmsh.model.setPhysicalName(2, LEFT, "Left wall")
        gmsh.model.addPhysicalGroup(2, right_wall, RIGHT)
        gmsh.model.setPhysicalName(2, RIGHT, "Right wall")
        gmsh.model.addPhysicalGroup(2, front_wall, FRONT)
        gmsh.model.setPhysicalName(2, FRONT, "Front wall")
        gmsh.model.addPhysicalGroup(2, back_wall, BACK)
        gmsh.model.setPhysicalName(2, BACK, "Back wall")
        gmsh.model.addPhysicalGroup(2, bottom_wall, BOTTOM)
        gmsh.model.setPhysicalName(2, BOTTOM, "Bottom wall")
        gmsh.model.addPhysicalGroup(2, top_wall, TOP)
        gmsh.model.setPhysicalName(2, TOP, "Top wall")

        # Set number of threads used for mesh
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", MPI.COMM_WORLD.size)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", MPI.COMM_WORLD.size)

        # Set uniform mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

        # Generate mesh
        gmsh.model.mesh.generate(dim = 3)
    # Convert gmsh model to DOLFINx Mesh and meshtags
    mesh, _, ft = dfx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()

    return mesh, ft