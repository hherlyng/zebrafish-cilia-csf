import numpy as np
import gmsh

# --------------------------------------------------------------------
def cylinder_gmsh(model: gmsh.model, length: float, radius: float):
    #         B
    #      C  0  A
    #         D
    fac = model.occ
    
    origin = fac.addPoint(-length/2, 0, 0)
    A = fac.addPoint(-length/2, radius, 0)
    B = fac.addPoint(-length/2, 0, radius)
    C = fac.addPoint(-length/2, -radius, 0)
    D = fac.addPoint(-length/2, 0, -radius)    

    OA = fac.addLine(origin, A)
    OB = fac.addLine(origin, B)
    OC = fac.addLine(origin, C)
    OD = fac.addLine(origin, D)

    AOB = fac.addCircleArc(A, origin, B)
    BOC = fac.addCircleArc(B, origin, C)
    COD = fac.addCircleArc(C, origin, D)
    DOA = fac.addCircleArc(D, origin, A)

    ur = fac.addPlaneSurface([fac.addCurveLoop([AOB, -OB, OA])])
    ul = fac.addPlaneSurface([fac.addCurveLoop([BOC, -OC, OB])])
    ll = fac.addPlaneSurface([fac.addCurveLoop([COD, -OD, OC])])
    lr = fac.addPlaneSurface([fac.addCurveLoop([DOA, -OA, OD])])

    [fac.extrude([(2, s)], length, 0, 0) for s in (ur, ul, ll, lr)]
    fac.synchronize()

    fac.removeAllDuplicates()
    fac.synchronize()
    
    vol = model.getEntities(3)
    bdry = model.getBoundary(vol)

    tol = 1E-10
    normals = {}
    left, right, wall = [], [], []
    for dim, tag in bdry:
        tag = abs(tag)
        x = fac.getCenterOfMass(dim, tag)
        # p = model.getParametrization(dim, tag, x)
        # normal = model.getNormal(tag, p)
        
        if abs(x[0]+length/2) < tol:
            left.append(tag)
        elif abs(x[0]-length/2) < tol:
            right.append(tag)
        else:
            wall.append(tag)

    fac.synchronize()            
    model.addPhysicalGroup(2, left, 1)
    model.addPhysicalGroup(2, right, 2)
    [model.addPhysicalGroup(2, [w], tag) for (tag, w) in enumerate(wall, 3)]
    model.addPhysicalGroup(3, [v[1] for v in vol], 1)

    fac.synchronize()
    
    # wall_normal = df.Expression(('0', 'x[1]/r', 'x[2]/r'), r=radius, degree=1)
    
    normals = []
    # {1: df.Constant((-1, 0, 0)),
    #            2: df.Constant((1, 0, 0)),
    #            3: wall_normal,
    #            4: wall_normal,
    #            5: wall_normal,
    #            6: wall_normal}

    fac.synchronize()
                                       
    return model, normals


def cylinder_gmsh_meshes(length, radius, ncells, nrefs):
    '''Generate refinements.
       
       Facet tags (quadrants in yz-plane):
        - 1 = left side
        - 2 = right side
        - 3 = 1st quadrant
        - 4 = 2nd quadrant
        - 5 = 3rd quadrant
        - 6 = 4th quadrant
    '''
    gmsh.initialize()
    model = gmsh.model
    model, normals = cylinder_gmsh(model, length, radius)

    scale = 1./ncells
    for k in range(nrefs):
        gmsh.option.setNumber('Mesh.MeshSizeFactor', scale/2**(k))
        model.mesh.generate(dim = 3)

        gmsh.write(f"cylinder_{k}.msh")
        
        # Ready for next round
        gmsh.model.mesh.clear()
    # At this point we are done with gmsh
    gmsh.finalize()

# ---

def torus_gmsh(model, length, radius):
    #         B
    #      C  0  A
    #         D
    fac = model.occ
    
    origin = fac.addPoint(-length/2, 0, 0)
    A = fac.addPoint(-length/2, radius, 0)
    B = fac.addPoint(-length/2, 0, radius)
    C = fac.addPoint(-length/2, -radius, 0)
    D = fac.addPoint(-length/2, 0, -radius)    

    OA = fac.addLine(origin, A)
    OB = fac.addLine(origin, B)
    OC = fac.addLine(origin, C)
    OD = fac.addLine(origin, D)

    AOB = fac.addCircleArc(A, origin, B)
    BOC = fac.addCircleArc(B, origin, C)
    COD = fac.addCircleArc(C, origin, D)
    DOA = fac.addCircleArc(D, origin, A)

    ur = fac.addPlaneSurface([fac.addCurveLoop([AOB, -OB, OA])])
    ul = fac.addPlaneSurface([fac.addCurveLoop([BOC, -OC, OB])])
    ll = fac.addPlaneSurface([fac.addCurveLoop([COD, -OD, OC])])
    lr = fac.addPlaneSurface([fac.addCurveLoop([DOA, -OA, OD])])

    theta = np.pi/4
    R = length/theta
    y_shift = np.sqrt(R**2 - (length/2)**2)
    
    [fac.revolve([(2, s)], 0, -y_shift, 0, 0, 0, 1, theta) for s in (ur, ul, ll, lr)]
    fac.synchronize()

    fac.removeAllDuplicates()
    fac.synchronize()

    vol = model.getEntities(3)
    bdry = model.getBoundary(vol)

    tol = 1E-10
    normals = {}
    left, right, wall = [], [], []
    for dim, tag in bdry:
        tag = abs(tag)
        x = fac.getCenterOfMass(dim, tag)
        p = model.getParametrization(dim, tag, x)
        curvature = model.getCurvature(dim, tag, p)
        normal = model.getNormal(tag, p)

        if abs(curvature) < tol:
            if abs(x[0]+length/2) < tol:
                left.append(tag)
                normals[1] = df.Constant(normal)
            else:
                right.append(tag)
                normals[2] = df.Constant(normal)
        else:
            wall.append(tag)
    fac.synchronize()

    model.addPhysicalGroup(2, left, 1)
    model.addPhysicalGroup(2, right, 2)
    [model.addPhysicalGroup(2, [w], tag) for (tag, w) in enumerate(wall, 3)]

    model.addPhysicalGroup(3, [v[1] for v in vol], 1)

    fac.synchronize()

    # Add normals for curved bit
    class WallNormal(df.UserExpression):
        def __init__(self, model, dim, tag, **kwargs):
            self.model = model
            self.dim = dim
            self.tag = tag
            df.UserExpression.__init__(self, **kwargs)            

        def value_shape(self): return (3, )

        def eval(self, values, x):
            p = self.model.getParametrization(self.dim, self.tag, x)
            normal = model.getNormal(self.tag, p)
            values[:] = normal

    for (phys_tag, tag) in enumerate(wall, 3):
        normals[phys_tag] = WallNormal(model, 2, tag, degree=1)

    return model, normals


def torus_gmsh_meshes(length, radius, ncells, nrefs):
    '''Generate refinments'''
    gmsh.initialize()
    model = gmsh.model
    model, normals = torus_gmsh(model, length, radius)

    scale = 1./ncells
    for k in range(nrefs):
        gmsh.option.setNumber('Mesh.MeshSizeFactor', scale/2**(k))

        nodes, topologies = msh_gmsh_model(model, 3)
        mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

        yield entity_functions[2], normals
        
        # Ready for next round
        gmsh.model.mesh.clear()
    # At this point we are done with gmsh
    gmsh.finalize()

# --------------------------------------------------------------------

if __name__ == '__main__':
    
    gen = cylinder_gmsh_meshes(length=1, radius=0.25, ncells=1, nrefs=5)