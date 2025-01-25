import gmsh

import numpy   as np

from   mpi4py  import MPI

gmsh.initialize()

# Mesh and domain parameters
L = 1 # Length of domain [m]
H = 1 # Height of domain [m]
D = 1 # Depth of domain [m]
r = 0.2 # Radius of sphere [m]

c_x = 0.5 # x coordinate of sphere center [m]
c_y = 0.5 # y coordinate of sphere center [m]
c_z = 0.5 # z coordinate of sphere center [m]

gmsh_dim = 3
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    box_domain = gmsh.model.occ.addBox(0, 0, 0, L, H, D, tag = 1)
    ball       = gmsh.model.occ.addSphere(c_x, c_y, c_z, r)

# Remove the cylinder from the box domain
if mesh_comm.rank == model_rank:
    fluid_domain = gmsh.model.occ.cut([(gmsh_dim, box_domain)], [(gmsh_dim, ball)])
    gmsh.model.occ.synchronize()

# Add physical volume marker for the fluid mesh
fluid_marker = 10
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim = gmsh_dim)
    assert(len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")
    
# Tag the boundaries of the mesh by computing the centers of mass of the boundaries
LEFT    = 1
RIGHT   = 2
FRONT   = 3
BACK    = 4
BOTTOM  = 5
TOP     = 6
SPHERE  = 8
left, right, front, back, bottom, top, sphere = [], [], [], [], [], [], []

#INLET, OUTLET, WALL, CYLINDER = 2, 3, 4, 5 # Marker values
#inflow, outflow, walls, cyl = [], [], [], []

if mesh_comm.rank == model_rank:
    # Get boundaries and loop over all
    boundaries = gmsh.model.getBoundary(volumes, oriented = False)
    for boundary in boundaries:
        com = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(com, [0, H/2, D/2]):
            # Boundary is inlet/left side
            left.append(boundary[1])
        elif np.allclose(com, [L, H/2, D/2]):
            # Boundary is outlet/right side
            right.append(boundary[1])
        elif np.allclose(com, [L/2, 0, D/2]):
            # Boundary is lower wall
            front.append(boundary[1])
        elif np.allclose(com, [L/2, H, D/2]):
            # Boundar is upper wall
            back.append(boundary[1])
        elif np.allclose(com, [L/2, H/2, 0]):
            bottom.append(boundary[1])
        elif np.allclose(com, [L/2, H/2, D]):
            top.append(boundary[1])
        else:
            # Boundary is on the sphere
            sphere.append(boundary[1])
    
    # Add boundary tags to the model
    gmsh.model.addPhysicalGroup(gmsh_dim - 1, left, LEFT)
    gmsh.model.setPhysicalName(gmsh_dim - 1, LEFT, "Left")
    gmsh.model.addPhysicalGroup(gmsh_dim - 1, right, RIGHT)
    gmsh.model.setPhysicalName(gmsh_dim - 1, RIGHT, "Right")
    gmsh.model.addPhysicalGroup(gmsh_dim - 1, front, FRONT)
    gmsh.model.setPhysicalName(gmsh_dim - 1, FRONT, "Front")
    gmsh.model.addPhysicalGroup(gmsh_dim - 1, back, BACK)
    gmsh.model.setPhysicalName(gmsh_dim - 1, BACK, "Back")
    gmsh.model.addPhysicalGroup(gmsh_dim - 1, bottom, BOTTOM)
    gmsh.model.setPhysicalName(gmsh_dim - 1, BOTTOM, "Bottom")
    gmsh.model.addPhysicalGroup(gmsh_dim - 1, top, TOP)
    gmsh.model.setPhysicalName(gmsh_dim - 1, TOP, "Top")
    gmsh.model.addPhysicalGroup(gmsh_dim - 1, sphere, SPHERE)
    gmsh.model.setPhysicalName(gmsh_dim - 1, SPHERE, "Sphere")

# Refine mesh cells close to the cylinder
refine = False
if refine:
    res_min = r / 3
    if mesh_comm.rank == model_rank:
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", sphere)

        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)

        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

# Generate the mesh
length_factor = 0.2
if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", length_factor)
    gmsh.model.mesh.generate(gmsh_dim)
    gmsh.model.mesh.optimize("Netgen")

gmsh.write("sphere_in_cube_LF=" + str(length_factor) + ".msh")