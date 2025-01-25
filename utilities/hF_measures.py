import ufl

import dolfinx as dfx

from ufl import dot, avg, sym, grad, jump, sqrt, inner, dx, ds, dS
from petsc4py import PETSc

# NOTE: these are the jump operators from Krauss, Zikatonov paper.
# Jump is just a difference and it preserves the rank 
Jump = lambda arg: arg('+') - arg('-')
# Average uses dot with normal and AGAIN MINUS; it reduces the rank
Avg = lambda arg, n: 0.5*(dot(arg('+'), n('+')) - dot(arg('-'), n('-')))
# Action of (1 - n x n)
Tangent = lambda v, n: v - n*dot(v, n)    

Normal = lambda u, n: n*dot(u, n)


def Stabilization(u, v, mu, stab_h, stab_gamma):
    '''Displacement/Flux Stabilization from Krauss et al paper'''
    mesh = u.ufl_domain().ufl_cargo()

    get_nom, get_den = get_penalty(stab_h, stab_gamma)
    penalty = get_nom(u)

    hA = avg(get_den(mesh))

    n = ufl.FacetNormal(mesh)

    D = lambda v: sym(grad(v))

    return (-inner(Avg(2*mu*D(u), n), Jump(Tangent(v, n)))*dS
            -inner(Avg(2*mu*D(v), n), Jump(Tangent(u, n)))*dS
            + 2*mu*(penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS), penalty/hA


def get_penalty(stab_h, stab_gamma):
    '''Nom/Denominator of the penalty formula'''
    nitsche_conformity_penalty = lambda u, stab_gamma=stab_gamma: stab_gamma

    hA = {'diameter': CellDiameter2,
          'edge': EdgeLength,
          'centroid': CentroidDistance,
          }[stab_h]

    return nitsche_conformity_penalty, hA


def CellDiameter2(mesh):
    '''Length of edge as P0 edge function'''
    Q = dfx.fem.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    q = ufl.TestFunction(Q)

    fh = dfx.fem.Function(Q)
    hA = ufl.FacetArea(mesh)
    cd = ufl.CellDiameter(mesh)
    
    dfx.fem.assemble_scalar(dfx.fem.form((1/avg(hA))*inner(avg(q), avg(cd))*dS + (1/hA)*inner(q, cd)*ds,
             tensor=fh.vector()))

    return fh


def EdgeLength(mesh):
    '''Length of edge as P0 edge function'''
    Q = dfx.fem.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    q = ufl.TestFunction(Q)

    fh = dfx.fem.Function(Q)
    const1 = dfx.fem.Constant(mesh, PETSc.ScalarType(1))
    # This really is a FacetArea
    dfx.fem.assemble_scalar(dfx.fem.form(inner(const1, avg(q))*dS + inner(const1, q)*ds,
             tensor=fh.vector()))

    return fh


def CentroidDistance(mesh: dfx.mesh.Mesh):
    '''Discontinuous Lagrange Trace function that holds the cell-to-cell distance'''
    # Cell-cell distance for the interior facet is defined as a distance 
    # of midpoints of the cells that share the facet. For exterior facet
    # we take the distance of cell midpoint and the facet midpoint

    Q = dfx.fem.FunctionSpace(mesh, ('DG', 0))
    V = dfx.fem.FunctionSpace(mesh, ('CG', 1))
    # Create submesh to define DG trace space
    submesh, _, _, _ = dfx.mesh.create_submesh(mesh, mesh.topology.dim - 1, dfx.mesh.exterior_facet_indices(mesh.topology))
    L = dfx.fem.FunctionSpace(submesh, ('DG', 0))

    cK, fK = ufl.CellVolume(mesh), ufl.FacetArea(mesh)
    q, l = ufl.TestFunction(Q), ufl.TestFunction(L)
    # The idea here to first assemble component by component the cell 
    # and (exterior) facet midpoint
    cell_centers, facet_centers = [], []
    for xi in ufl.SpatialCoordinate(mesh):
        qi = dfx.fem.Function(Q)
        # Pretty much use the definition that a midpoint is int_{cell} x_i/vol(cell)
        # It's thanks to eval in q that we take values :)
        from IPython import embed;embed()
        qi.vector[:] = dfx.fem.assemble_scalar(dfx.fem.form((1/cK)*inner(xi, q)*dx))
        cell_centers.append(qi)
        # Same here but now our mean is over an edge
        li = dfx.fem.Function(L)
        li.vector[:] = dfx.fem.assemble_scalar(dfx.fem.form((1/fK)*inner(xi, l)*ds))
        facet_centers.append(li)
    # We build components to vectors
    cell_centers, facet_centers = map(ufl.as_vector, (cell_centers, facet_centers))

    distances = dfx.fem.Function(L)
    # FIXME: This might not be necessary but it's better to be certain
    dS_, ds_ = dS(metadata={'quadrature_degree': 0}), ds(metadata={'quadrature_degree': 0})
    # Finally we assemble magniture of the vector that is determined by the
    # two centers
    distances.vector[:] = [dfx.fem.assemble_scalar(dfx.fem.form(((1/fK('+'))*inner(sqrt(dot(jump(cell_centers), jump(cell_centers))), l('+'))*dS_+
              (1/fK)*inner(sqrt(dot(cell_centers-facet_centers, cell_centers-facet_centers)), l)*ds_))) for i in range(len(cell_centers))]

    return distances
