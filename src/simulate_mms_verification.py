import ufl

import numpy   as np
import dolfinx as dfx

from ufl       import div, dot, inner, sym, grad, avg, curl
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element, mixed_element
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, set_bc, apply_lifting

print = PETSc.Sys.Print

# Operators
# NOTE: these are the jump operators from Krauss, Zikatonov paper.
# Jump is just a difference and it preserves the rank 
Jump = lambda arg: arg('+') - arg('-')

# Average uses dot with normal and AGAIN MINUS; it reduces the rank
Avg = lambda arg, n: .5*(dot(arg('+'), n('+')) - dot(arg('-'), n('-')))

# Action of (1 - n x n)
Tangent = lambda v, n: v - n*dot(v, n)

# Symmetric gradient
eps = lambda u: sym(grad(u))

def assemble_system(a_cpp: dfx.fem.form, L_cpp: dfx.fem.form, bcs: list[dfx.fem.dirichletbc]):
    A = assemble_matrix(a_cpp, bcs)
    A.assemble()

    b = assemble_vector(L_cpp)
    apply_lifting(b, [a_cpp], bcs=[bcs])
    # b.zeroEntries()
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs=bcs)
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    return A, b

# Stabilization terms for the variational form
def Stabilization(mesh: dfx.mesh.Mesh,
                u: ufl.TrialFunction,
                v: ufl.TestFunction,
                mu: dfx.fem.Constant,
                penalty: dfx.fem.Constant,
                consistent: bool=True):

    '''Displacement/Flux Stabilization from Krauss et al paper'''
    n, hA = ufl.FacetNormal(mesh), avg(ufl.CellDiameter(mesh))

    D  = lambda v: sym(grad(v)) # the symmetric gradient
    dS = ufl.Measure('dS', domain=mesh)

    if consistent:
        return (-inner(Avg(2*mu*D(u), n), Jump(Tangent(v, n)))*dS
                -inner(Avg(2*mu*D(v), n), Jump(Tangent(u, n)))*dS
                + 2*mu*(penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS)

    # For preconditioning
    return 2*mu*(penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS

def get_system_mms(msh: dfx.mesh.Mesh, penalty_val: float, mu_val: float, direct: bool):

    k  = 1
    cell = msh.basix_cell()
    Velm = element('BDM', cell, k)
    Qelm = element('DG', cell, k-1)
    Welm = mixed_element([Velm, Qelm])
    W = dfx.fem.functionspace(msh, Welm)
    V, _ = W.sub(0).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Exact expressions and normals
    dx = ufl.Measure("dx", msh)
    ds = ufl.Measure("ds", domain=msh)
    n = ufl.FacetNormal(msh)

    length_scale = mesh.comm.allreduce(mesh.geometry.x[:, 0].max(), op=MPI.MAX) - mesh.comm.allreduce(mesh.geometry.x[:, 0].min(), op=MPI.MIN)
    if msh.geometry.dim==2:
        x, y = ufl.SpatialCoordinate(msh)
        vel = ufl.as_vector((ufl.sin(ufl.pi*(x-y)/length_scale), ufl.sin(ufl.pi*(x-y)/length_scale)))
        pres = ufl.cos(ufl.pi*(x+y)/length_scale)
    elif msh.geometry.dim==3:
        x, y, z = ufl.SpatialCoordinate(msh)
        phi = ufl.as_vector((ufl.sin(ufl.pi*(x-y)/length_scale), ufl.sin(ufl.pi*(y+z)/length_scale), ufl.sin(ufl.pi*(x-z)/length_scale)))
        pres = ufl.cos(ufl.pi*(x+y+z)/length_scale)
        vel = curl(phi)
        
    # Variational form
    tangent_traction = lambda n: Tangent(dot(sigma, n), n)
    penalty = dfx.fem.Constant(msh, dfx.default_scalar_type(penalty_val))
    mu = dfx.fem.Constant(msh, dfx.default_scalar_type(mu_val))

    sigma = 2*mu*eps(vel) - pres*ufl.Identity(msh.geometry.dim)
    f = -div(sigma)

    a  = 2*mu*inner(eps(u), eps(v)) * dx - p * div(v) * dx - q * div(u) * dx
    a += Stabilization(msh, u, v, mu, penalty=penalty)

    L  = inner(f, v) * dx

    L += inner(Tangent(v, n), tangent_traction(n)) * ds

    u_bc = dfx.fem.Function(V)
    u_bc_expr = dfx.fem.Expression(vel, V.element.interpolation_points())
    u_bc.interpolate(u_bc_expr)

    msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
    bdry_facets = dfx.mesh.exterior_facet_indices(msh.topology)
    dofs = dfx.fem.locate_dofs_topological((W.sub(0), V), msh.topology.dim-1, bdry_facets)
    bc   = dfx.fem.dirichletbc(u_bc, dofs, W.sub(0))

    bcs = [bc]

    # Assemble system
    a_cpp, L_cpp = dfx.fem.form(a), dfx.fem.form(L)
    A, b = assemble_system(a_cpp, L_cpp, bcs)

    # Create nullspace vector
    Q, Q_dofs = W.sub(1).collapse()
    ns_vec = A.createVecLeft()
    ns_vec.setValuesLocal(Q_dofs, np.ones(Q_dofs.__len__()))
    ns_vec.assemble()
    ns_vec.normalize()
    nullspace = PETSc.NullSpace().create(vectors=[ns_vec], comm=msh.comm)
    assert(nullspace.test(A))
    A.setNullSpace(nullspace) if direct else A.setNearNullSpace(nullspace)
    nullspace.remove(b)

    u_ex = u_bc
    p_ex = dfx.fem.Function(Q)
    p_ex_expr = dfx.fem.Expression(pres, Q.element.interpolation_points())
    p_ex.interpolate(p_ex_expr)

    # Preconditioner
    a_prec = (2*mu*inner(eps(u), eps(v))*dx
             + (1/mu)*inner(p, q)*dx)

    # Account for HDiv
    a_prec += Stabilization(msh, u, v, mu, penalty=penalty, consistent=False)
    a_prec_cpp = dfx.fem.form(a_prec)
    B, _ = assemble_system(a_prec_cpp, L_cpp, bcs)

    return A, b, W, B, u_ex, p_ex

# Mesh tags for flow
ANTERIOR_PRESSURE    = 2
POSTERIOR_PRESSURE   = 3
VOLUME               = 4
MIDDLE_VENTRAL_CILIA = 5
MIDDLE_DORSAL_CILIA  = 6
ANTERIOR_CILIA       = 7
SLIP                 = 8

def get_system(msh: dfx.mesh.Mesh, penalty_val: float, mu_val: float, direct: bool):

    from utilities.mesh import mark_boundaries_flow
    bdry_facets = mark_boundaries_flow(mesh=msh, inflow_outflow=False)
    k  = 1
    cell = msh.basix_cell()
    Velm = element('BDM', cell, k)
    Qelm = element('DG', cell, k-1)
    Welm = mixed_element([Velm, Qelm])
    W = dfx.fem.functionspace(msh, Welm)
    V, _ = W.sub(0).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Exact expressions and normals
    dx = ufl.Measure("dx", msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=bdry_facets)
    n = ufl.FacetNormal(msh)
    
    # Define the stress vector used in the slip boundary conditions
    tau = 6.5e-4 # Tangential stress BC parameter
    tau_vec   = tau*ufl.as_vector((1, 0, 1)) # Stress vector to be projected tangentially onto the mesh
    tangent_traction = lambda n: Tangent(tau_vec, n) # The tangential component of the stress vector

    f = dfx.fem.Function(V)
    
    # Variational form
    penalty = dfx.fem.Constant(msh, dfx.default_scalar_type(penalty_val))
    mu = dfx.fem.Constant(msh, dfx.default_scalar_type(mu_val))

    a  = 2*mu*inner(eps(u), eps(v)) * dx - p * div(v) * dx - q * div(u) * dx
    a += Stabilization(msh, u, v, mu, penalty=penalty)

    L  = inner(f, v) * dx
    L += -inner(Tangent(v, n), tangent_traction(n)) * (ds(ANTERIOR_CILIA) + ds(MIDDLE_DORSAL_CILIA))
    L +=  inner(Tangent(v, n), tangent_traction(n)) * ds(MIDDLE_VENTRAL_CILIA)
    
    # Impose impermeability boundary condition strongly
    # Create facet-cell connectivity and get the facets of the slip boundary
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim) 
    imperm_bdry = np.concatenate((bdry_facets.find(ANTERIOR_CILIA),
                                  bdry_facets.find(MIDDLE_DORSAL_CILIA),
                                  bdry_facets.find(MIDDLE_VENTRAL_CILIA),
                                  bdry_facets.find(SLIP)))
    u_bc = dfx.fem.Function(V)
    dofs = dfx.fem.locate_dofs_topological((W.sub(0), V), msh.topology.dim-1, imperm_bdry)
    bcs  = [dfx.fem.dirichletbc(u_bc, dofs, W.sub(0))]

    # Assemble system
    a_cpp, L_cpp = dfx.fem.form(a), dfx.fem.form(L)
    A, b = assemble_system(a_cpp, L_cpp, bcs)

    # Create nullspace vector 
    ns_vec = A.createVecLeft()
    _, Q_dofs = W.sub(1).collapse()
    ns_vec.setValuesLocal(Q_dofs, np.ones(Q_dofs.__len__()))
    ns_vec.assemble()
    ns_vec.normalize()
    nullspace = PETSc.NullSpace().create(vectors=[ns_vec], comm=msh.comm)
    assert(nullspace.test(A))
    A.setNullSpace(nullspace)
    if not direct: A.setNearNullSpace(nullspace)
    nullspace.remove(b)

    # Preconditioner
    a_prec = (2*mu*inner(eps(u), eps(v))*dx
             + (1/mu)*inner(p, q)*dx)

    # Account for HDiv
    a_prec += Stabilization(msh, u, v, mu, penalty=penalty, consistent=False)
    a_prec_cpp = dfx.fem.form(a_prec)
    B, _ = assemble_system(a_prec_cpp, L_cpp, bcs)

    return A, b, W, B

def solve(A: PETSc.Mat, B: PETSc.Mat, b: PETSc.Vec, W: dfx.fem.FunctionSpace, direct: bool):
    wh = dfx.fem.Function(W)

    if direct:
        # Setup solver
        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        opts = PETSc.Options()
        opts.setValue('ksp_view', None)
        opts.setValue('ksp_monitor_true_residual', None)                
        opts.setValue('ksp_converged_reason', None)
        ksp.setFromOptions()

        ksp.solve(b, wh.vector)

    else:
        ksp = PETSc.KSP().create(mesh.comm)
        ksp.setOperators(A, B)

        opts = PETSc.Options()
        opts.setValue('ksp_type', 'minres')
        opts.setValue('ksp_rtol', 1E-12)                
        opts.setValue('ksp_view', None)
        opts.setValue('ksp_monitor_true_residual', None)                
        opts.setValue('ksp_converged_reason', None)
        opts.setValue('fieldsplit_0_ksp_type', 'preonly')
        opts.setValue('fieldsplit_0_pc_type', 'lu')
        opts.setValue('fieldsplit_1_ksp_type', 'preonly')
        opts.setValue('fieldsplit_1_pc_type', 'lu')           

        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.FIELDSPLIT)
        V, _ = W.sub(0).collapse()
        Q, _ = W.sub(1).collapse()
        V_map = V.dofmap.index_map
        Q_map = Q.dofmap.index_map
        offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + Q_map.local_range[0]
        offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
        is_u = PETSc.IS().createStride(V_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF)
        is_p = PETSc.IS().createStride(Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)


        pc.setFieldSplitIS(('0', is_u), ('1', is_p))
        pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE) 

        ksp.setUp()
        pc.setFromOptions()
        ksp.setFromOptions()

        ksp.solve(b, wh.vector)
    
    print(f"Converged reason: {ksp.getConvergedReason()}")
    wh.x.scatter_forward()
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    uh = dfx.fem.Function(V)
    ph = dfx.fem.Function(Q)

    uh.x.array[:] = wh.sub(0).collapse().x.array.copy()
    ph.x.array[:] = wh.sub(1).collapse().x.array.copy()

    uh_out = dfx.fem.Function(dfx.fem.functionspace(mesh, ufl.VectorElement("DG", mesh.ufl_cell(), 1)))
    uh_out.interpolate(uh)

    if not direct:
        # Get number of iterations and residual norm
        niters = ksp.getIterationNumber()
        rnorm  = ksp.getResidualNorm()
    else:
        # Placeholders
        niters = None
        rnorm  = None

    return uh, ph, uh_out, niters, rnorm


if __name__ == '__main__':

    import tabulate
    import time
    tic = time.perf_counter()
    mu_value = 1.0
    penalty_value = 10.0
    direct = True
    mesh_type = 'cyl'

    history = []

    if mesh_type=='cyl':

        # init
        if direct:
            headers = ('hmin', 'hmax', '#cells', 'dimW', '|eu|_0', 'EOC', '|eu|_1', 'EOC', '|eu|_div', 'EOC', '|eu|_Linf', 'EOC', '|div u|_0', '|ep|_0', 'EOC')
        else:
            headers = ('hmin', 'hmax', '#cells', 'dimW', '|eu|_0', 'EOC', '|eu|_1', 'EOC', '|eu|_div', 'EOC', '|eu|_Linf', 'EOC', '|div u|_0', '|ep|_0', 'EOC', 'niters', '|r|')
        eu_L2_prev  = 0
        eu_H1_prev  = 0
        eu_div_prev = 0
        ep_L2_prev  = 0

        for i in [0, 1, 2, 3, 4, 5]:

            with dfx.io.XDMFFile(MPI.COMM_WORLD, f'./geometries/cylinder_{i}.xdmf', "r") as xdmf:
                mesh = xdmf.read_mesh()
            mesh.geometry.x[:] *= 1000 # Scale mesh
            
            A, b, W, B, u0, p0 = get_system_mms(mesh, penalty_value, mu_value, direct)
            uh, ph, uh_out, niters, rnorm = solve(A, B, b, W, direct)

            # Calculate mean pressure and subtract it from the calculated pressure
            dx = ufl.Measure('dx', domain=mesh) 
            vol = mesh.comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1 * dx)), op=MPI.SUM)
            mean_p_h = mesh.comm.allreduce(1/vol * dfx.fem.assemble_scalar(dfx.fem.form(ph * dx)), op=MPI.SUM)

            ph.x.array[:] -= mean_p_h
            p0.x.array[:] -= mesh.comm.allreduce(1/vol * dfx.fem.assemble_scalar(dfx.fem.form(p0 * dx)), op=MPI.SUM)

            # Calculate maximum velocity magnitude
            u1 = uh_out.sub(0).collapse().x.array
            u2 = uh_out.sub(1).collapse().x.array
            u3 = uh_out.sub(2).collapse().x.array
            uh_mag = np.sqrt(u1**2 + u2**2 + u3**2)
            uh_mag_max = uh_mag.max()
            u_Linf = mesh.comm.allreduce(uh_mag_max, op=MPI.MAX)
            #u_Linf_scaled = mesh.comm.allreduce(uh_mag_max/gamma_c, op=MPI.MAX)
            
            u0_out = dfx.fem.Function(dfx.fem.functionspace(mesh, ufl.VectorElement("DG", mesh.ufl_cell(), 1)))
            u0_out.interpolate(u0)
            u1_ex = u0_out.sub(0).collapse().x.array
            u2_ex = u0_out.sub(1).collapse().x.array
            u3_ex = u0_out.sub(2).collapse().x.array
            u0_mag = np.sqrt(u1_ex**2 + u2_ex**2 + u3_ex**2)
            u0_mag_max = u0_mag.max()
            u0_Linf = mesh.comm.allreduce(u0_mag_max, op=MPI.MAX)

            from utilities.helpers import calc_error_H1, calc_error_L2, calc_error_Hdiv
            eu_H1  = calc_error_H1(u_approx=uh, u_exact=u0, dX=dx)
            eu_L2  = calc_error_L2(u_approx=uh, u_exact=u0, dX=dx)
            ep_L2  = calc_error_L2(u_approx=ph, u_exact=p0, dX=dx)
            eu_div = calc_error_Hdiv(u_approx=uh, u_exact=u0, dX=dx)
            eu_Linf = u_Linf - u0_Linf

            div_uh = np.sqrt(mesh.comm.allreduce(abs(dfx.fem.assemble_scalar(dfx.fem.form(div(uh)**2*dx))), op=MPI.SUM))
            tdim = mesh.topology.dim
            num_cells = mesh.topology.index_map(tdim).size_local
            cells = np.arange(num_cells, dtype=np.int32)
            h = dfx.cpp.mesh.h(mesh._cpp_object, tdim, cells)
            hmin = h.min()
            hmax = h.max()
            hmin = mesh.comm.allreduce(hmin, op=MPI.MIN)
            hmax = mesh.comm.allreduce(hmax, op=MPI.MAX)

            if i==0:
                if direct:
                    history.append((hmin, hmax, mesh.topology.index_map(3).size_global, W.dofmap.index_map.size_global, f'{eu_L2:.1E}', '--', f'{eu_H1:.1E}', '--', f'{eu_div:.1E}', '--', f'{eu_Linf:.1E}', '--', f'{div_uh:.1E}', f'{ep_L2:.1E}', '--'))
                else:
                    history.append((hmin, hmax, mesh.topology.index_map(3).size_global, W.dofmap.index_map.size_global, f'{eu_L2:.1E}', '--', f'{eu_H1:.1E}', '--', f'{eu_div:.1E}', '--', f'{eu_Linf:.1E}', '--', f'{div_uh:.1E}', f'{ep_L2:.1E}', '--', niters, rnorm))
            else:
                eu_L2_eoc   = np.log(eu_L2_prev/eu_L2) / np.log(2)
                eu_H1_eoc   = np.log(eu_H1_prev/eu_H1) / np.log(2)
                eu_div_eoc  = np.log(eu_div_prev/eu_div) / np.log(2)
                eu_Linf_eoc = np.log(eu_Linf_prev/eu_Linf) / np.log(2)
                ep_L2_eoc   = np.log(ep_L2_prev/ep_L2) / np.log(2)
                if direct:
                    history.append((hmin, hmax, mesh.topology.index_map(3).size_global, W.dofmap.index_map.size_global, f'{eu_L2:.1E}', f'{eu_L2_eoc:.2f}', f'{eu_H1:.1E}', f'{eu_H1_eoc:.2f}', f'{eu_div:.1E}', f'{eu_div_eoc:.2f}', f'{eu_Linf:.1E}', f'{eu_Linf_eoc:.2f}', f'{div_uh:.1E}', f'{ep_L2:.1E}', f'{ep_L2_eoc:.2f}'))
                else:
                    history.append((hmin, hmax, mesh.topology.index_map(3).size_global, W.dofmap.index_map.size_global, f'{eu_L2:.1E}', f'{eu_L2_eoc:.2f}', f'{eu_H1:.1E}', f'{eu_H1_eoc:.2f}', f'{eu_div:.1E}', f'{eu_div_eoc:.2f}', f'{eu_Linf:.1E}', f'{eu_Linf_eoc:.2f}', f'{div_uh:.1E}', f'{ep_L2:.1E}', f'{ep_L2_eoc:.2f}', niters, rnorm))
            
            eu_L2_prev   = eu_L2
            eu_H1_prev   = eu_H1
            eu_div_prev  = eu_div
            eu_Linf_prev = eu_Linf
            ep_L2_prev   = ep_L2
            print(tabulate.tabulate(history, headers=headers))
            

    elif mesh_type=='zfish':

        # init
        headers = ('hmin', 'hmax', '#cells', 'dimW', '|u|_0', '|u|_Linf', '|u|_Linf_s', '|div u|_0', 'niters', '|r|')

        for i in [0, 1]:#, 2]:
            
            with dfx.io.XDMFFile(MPI.COMM_WORLD, f'./geometries/ventricles_{i}.xdmf', "r") as xdmf:
                mesh = xdmf.read_mesh()

            mu_value = 6.97e-4
            A, b, W, B = get_system(mesh, penalty_value, mu_value, direct)
            uh, ph, uh_out, niters, rnorm = solve(A, B, b, W, direct)

            # Calculate mean pressure and subtract it from the calculated pressure
            from utilities.mesh import mark_boundaries_flow
            ft = mark_boundaries_flow(mesh, inflow_outflow=False)
            dx = ufl.Measure('dx', domain=mesh)
            ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
            vol = mesh.comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1 * dx)), op=MPI.SUM)
            gamma_c = mesh.comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*ds((ANTERIOR_CILIA, MIDDLE_DORSAL_CILIA, MIDDLE_VENTRAL_CILIA)))), op=MPI.SUM)
            mean_p_h = mesh.comm.allreduce(1/vol * dfx.fem.assemble_scalar(dfx.fem.form(ph * dx)), op=MPI.SUM)

            ph.x.array[:] -= mean_p_h
            
            uh_L2  = np.sqrt(mesh.comm.allreduce(1/vol * dfx.fem.assemble_scalar(dfx.fem.form(dot(uh_out, uh_out) * dx)), op=MPI.SUM))
            div_uh_L2 = np.sqrt(mesh.comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(div(uh_out)*div(uh_out)*dx)), op=MPI.SUM))
            
            # Calculate maximum velocity magnitude
            u1 = uh_out.sub(0).collapse().x.array
            u2 = uh_out.sub(1).collapse().x.array
            u3 = uh_out.sub(2).collapse().x.array
            uh_mag = np.sqrt(u1**2 + u2**2 + u3**2)
            uh_mag_max = uh_mag.max()
            u_Linf = mesh.comm.allreduce(uh_mag_max, op=MPI.MAX)
            u_Linf_scaled = mesh.comm.allreduce(uh_mag_max/gamma_c, op=MPI.MAX)

            tdim = mesh.topology.dim
            num_cells = mesh.topology.index_map(tdim).size_local
            cells = np.arange(num_cells, dtype=np.int32)
            h = dfx.cpp.mesh.h(mesh._cpp_object, tdim, cells)
            hmin = mesh.comm.allreduce(h.min(), op=MPI.MIN)
            hmax = mesh.comm.allreduce(h.max(), op=MPI.MAX)

            # Scale to micrometers
            uh_L2   *= 1e3
            u_Linf *= 1e3
            u_Linf_scaled *= 1e-3
            div_uh_L2 *= 1e9

            history.append((hmin, hmax, mesh.topology.index_map(3).size_global, W.dofmap.index_map.size_global, uh_L2, u_Linf, u_Linf_scaled, div_uh_L2, niters, rnorm))
            print(tabulate.tabulate(history, headers=headers))

    elif mesh_type=='square':

        # init
        if direct:
            headers = ('hmin', 'hmax', '#cells', 'dimW', '|eu|_0', 'EOC', '|eu|_1', 'EOC', '|eu|_div', 'EOC', '|eu|_Linf', 'EOC', '|div u|_0', '|ep|_0', 'EOC')
        else:
            headers = ('hmin', 'hmax', '#cells', 'dimW', '|eu|_0', 'EOC', '|eu|_1', 'EOC', '|eu|_div', 'EOC', '|eu|_Linf', 'EOC', '|div u|_0', '|ep|_0', 'EOC', 'niters', '|r|')
        eu_L2_prev  = 0
        eu_H1_prev  = 0
        eu_div_prev = 0
        ep_L2_prev  = 0

        for i in [3, 4, 5, 6]:
            
            N_cells = 2**i
            mesh = dfx.mesh.create_unit_square(comm=MPI.COMM_WORLD, nx=N_cells, ny=N_cells, ghost_mode=dfx.mesh.GhostMode.shared_facet)
            
            A, b, W, B, u0, p0 = get_system_mms(mesh, penalty_value, mu_value, direct)
            uh, ph, uh_out, niters, rnorm = solve(A, B, b, W, direct)

            # Calculate mean pressure and subtract it from the calculated pressure
            dx = ufl.Measure('dx', domain=mesh) 
            vol = mesh.comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1 * dx)), op=MPI.SUM)
            mean_p_h = mesh.comm.allreduce(1/vol * dfx.fem.assemble_scalar(dfx.fem.form(ph * dx)), op=MPI.SUM)

            ph.x.array[:] -= mean_p_h
            p0.x.array[:] -= mesh.comm.allreduce(1/vol * dfx.fem.assemble_scalar(dfx.fem.form(p0 * dx)), op=MPI.SUM)

            # Calculate maximum velocity magnitude
            u1 = uh_out.sub(0).collapse().x.array
            u2 = uh_out.sub(1).collapse().x.array
            uh_mag = np.sqrt(u1**2 + u2**2)
            uh_mag_max = uh_mag.max()
            u_Linf = mesh.comm.allreduce(uh_mag_max, op=MPI.MAX)
            #u_Linf_scaled = mesh.comm.allreduce(uh_mag_max/gamma_c, op=MPI.MAX)
            
            u0_out = dfx.fem.Function(dfx.fem.functionspace(mesh, ufl.VectorElement("DG", mesh.ufl_cell(), 1)))
            u0_out.interpolate(u0)
            u1_ex = u0_out.sub(0).collapse().x.array
            u2_ex = u0_out.sub(1).collapse().x.array
            u0_mag = np.sqrt(u1_ex**2 + u2_ex**2)
            u0_mag_max = u0_mag.max()
            u0_Linf = mesh.comm.allreduce(u0_mag_max, op=MPI.MAX)

            from utilities.helpers import calc_error_H1, calc_error_L2, calc_error_Hdiv
            eu_H1  = calc_error_H1(u_approx=uh_out, u_exact=u0_out, dX=dx, vector_elements=True)
            eu_L2  = calc_error_L2(u_approx=uh_out, u_exact=u0_out, dX=dx, vector_elements=True)
            ep_L2  = calc_error_L2(u_approx=ph, u_exact=p0, dX=dx)
            eu_div = calc_error_Hdiv(u_approx=uh, u_exact=u0, dX=dx)
            eu_Linf = u_Linf - u0_Linf

            div_uh = np.sqrt(mesh.comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(div(uh)**2*dx)), op=MPI.SUM))
            tdim = mesh.topology.dim
            gdim = mesh.geometry.dim
            num_cells = mesh.topology.index_map(tdim).size_local
            cells = np.arange(num_cells, dtype=np.int32)
            h = dfx.cpp.mesh.h(mesh._cpp_object, tdim, cells)
            hmin = mesh.comm.allreduce(h.min(), op=MPI.MIN)
            hmax = mesh.comm.allreduce(h.max(), op=MPI.MAX)

            if i==3:
                if direct:
                    from decimal import Decimal
                    history.append((hmin, hmax, mesh.topology.index_map(gdim).size_global, W.dofmap.index_map.size_global, f'{eu_L2:.1E}', '--', f'{eu_H1:.1E}', '--', f'{eu_div:.1E}', '--', f'{eu_Linf:.1E}', '--', f'{div_uh:.1E}', f'{ep_L2:.1E}', '--'))
                else:
                    history.append((hmin, hmax, mesh.topology.index_map(gdim).size_global, W.dofmap.index_map.size_global, f'{eu_L2:.1E}', '--', f'{eu_H1:.1E}', '--', f'{eu_div:.1E}', '--', f'{eu_Linf:.1E}', '--', f'{div_uh:.1E}', f'{ep_L2:.1E}', '--', niters, rnorm))
            else:
                eu_L2_eoc   = np.log(eu_L2_prev/eu_L2) / np.log(2)
                eu_H1_eoc   = np.log(eu_H1_prev/eu_H1) / np.log(2)
                eu_div_eoc  = np.log(eu_div_prev/eu_div) / np.log(2)
                eu_Linf_eoc = np.log(eu_Linf_prev/eu_Linf) / np.log(2)
                ep_L2_eoc   = np.log(ep_L2_prev/ep_L2) / np.log(2)
                if direct:
                    history.append((hmin, hmax, mesh.topology.index_map(gdim).size_global, W.dofmap.index_map.size_global, f'{eu_L2:.1E}', f'{eu_L2_eoc:.2f}', f'{eu_H1:.1E}', f'{eu_H1_eoc:.2f}', f'{eu_div:.1E}', f'{eu_div_eoc:.2f}', f'{eu_Linf:.1E}', f'{eu_Linf_eoc:.2f}', f'{div_uh:.1E}', f'{ep_L2:.1E}', f'{ep_L2_eoc:.2f}'))
                else:
                    history.append((hmin, hmax, mesh.topology.index_map(gdim).size_global, W.dofmap.index_map.size_global, f'{eu_L2:.1E}', f'{eu_L2_eoc:.2f}', f'{eu_H1:.1E}', f'{eu_H1_eoc:.2f}', f'{eu_div:.1E}', f'{eu_div_eoc:.2f}', f'{eu_Linf:.1E}', f'{eu_Linf_eoc:.2f}', f'{div_uh:.1E}', f'{ep_L2:.1E}', f'{ep_L2_eoc:.2f}', niters, rnorm))
            
            eu_L2_prev   = eu_L2
            eu_H1_prev   = eu_H1
            eu_div_prev  = eu_div
            eu_Linf_prev = eu_Linf
            ep_L2_prev   = ep_L2
            print(tabulate.tabulate(history, headers=headers))

    elif mesh_type=='cube':

        # init
        if direct:
            headers = ('hmin', 'hmax', '#cells', 'dimW', '|eu|_0', 'EOC', '|eu|_1', 'EOC', '|eu|_div', 'EOC', '|eu|_Linf', 'EOC', '|div u|_0', '|ep|_0', 'EOC')
        else:
            headers = ('hmin', 'hmax', '#cells', 'dimW', '|eu|_0', 'EOC', '|eu|_1', 'EOC', '|eu|_div', 'EOC', '|eu|_Linf', 'EOC', '|div u|_0', '|ep|_0', 'EOC', 'niters', '|r|')
        eu_L2_prev  = 0
        eu_H1_prev  = 0
        eu_div_prev = 0
        ep_L2_prev  = 0

        for i in [2, 3, 4, 5]:
            
            N_cells = 2**i
            mesh = dfx.mesh.create_unit_cube(comm=MPI.COMM_WORLD, nx=N_cells, ny=N_cells, nz=N_cells)
            
            A, b, W, B, u0, p0 = get_system_mms(mesh, penalty_value, mu_value, direct)
            uh, ph, uh_out, niters, rnorm = solve(A, B, b, W, direct)

            # Calculate mean pressure and subtract it from the calculated pressure
            dx = ufl.Measure('dx', domain=mesh) 
            vol = mesh.comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1 * dx)), op=MPI.SUM)
            mean_p_h = mesh.comm.allreduce(1/vol * dfx.fem.assemble_scalar(dfx.fem.form(ph * dx)), op=MPI.SUM)

            ph.x.array[:] -= mean_p_h
            p0.x.array[:] -= mesh.comm.allreduce(1/vol * dfx.fem.assemble_scalar(dfx.fem.form(p0 * dx)), op=MPI.SUM)

            # Calculate maximum velocity magnitude
            u1 = uh_out.sub(0).collapse().x.array
            u2 = uh_out.sub(1).collapse().x.array
            u3 = uh_out.sub(2).collapse().x.array
            uh_mag = np.sqrt(u1**2 + u2**2 + u3**2)
            uh_mag_max = uh_mag.max()
            u_Linf = mesh.comm.allreduce(uh_mag_max, op=MPI.MAX)
            #u_Linf_scaled = mesh.comm.allreduce(uh_mag_max/gamma_c, op=MPI.MAX)
            
            u0_out = dfx.fem.Function(dfx.fem.functionspace(mesh, ufl.VectorElement("DG", mesh.ufl_cell(), 1)))
            u0_out.interpolate(u0)
            u1_ex = u0_out.sub(0).collapse().x.array
            u2_ex = u0_out.sub(1).collapse().x.array
            u3_ex = u0_out.sub(2).collapse().x.array
            u0_mag = np.sqrt(u1_ex**2 + u2_ex**2 + u3_ex**2)
            u0_mag_max = u0_mag.max()
            u0_Linf = mesh.comm.allreduce(u0_mag_max, op=MPI.MAX)

            from utilities.helpers import calc_error_H1, calc_error_L2, calc_error_Hdiv
            eu_H1  = calc_error_H1(u_approx=uh, u_exact=u0, dX=dx)
            eu_L2  = calc_error_L2(u_approx=uh, u_exact=u0, dX=dx)
            ep_L2  = calc_error_L2(u_approx=ph, u_exact=p0, dX=dx)
            eu_div = calc_error_Hdiv(u_approx=uh, u_exact=u0, dX=dx)
            eu_Linf = u_Linf - u0_Linf

            div_uh = np.sqrt(mesh.comm.allreduce(abs(dfx.fem.assemble_scalar(dfx.fem.form(div(uh)**2*dx))), op=MPI.SUM))
            tdim = mesh.topology.dim
            gdim = mesh.geometry.dim
            num_cells = mesh.topology.index_map(tdim).size_local
            cells = np.arange(num_cells, dtype=np.int32)
            h = dfx.cpp.mesh.h(mesh._cpp_object, tdim, cells)
            hmin = mesh.comm.allreduce(h.min(), op=MPI.MIN)
            hmax = mesh.comm.allreduce(h.max(), op=MPI.MAX)

            if i==2:
                if direct:
                    from decimal import Decimal
                    history.append((hmin, hmax, mesh.topology.index_map(gdim).size_global, W.dofmap.index_map.size_global, f'{eu_L2:.1E}', '--', f'{eu_H1:.1E}', '--', f'{eu_div:.1E}', '--', f'{eu_Linf:.1E}', '--', f'{div_uh:.1E}', f'{ep_L2:.1E}', '--'))
                else:
                    history.append((hmin, hmax, mesh.topology.index_map(gdim).size_global, W.dofmap.index_map.size_global, f'{eu_L2:.1E}', '--', f'{eu_H1:.1E}', '--', f'{eu_div:.1E}', '--', f'{eu_Linf:.1E}', '--', f'{div_uh:.1E}', f'{ep_L2:.1E}', '--', niters, rnorm))
            else:
                eu_L2_eoc   = np.log(eu_L2_prev/eu_L2) / np.log(2)
                eu_H1_eoc   = np.log(eu_H1_prev/eu_H1) / np.log(2)
                eu_div_eoc  = np.log(eu_div_prev/eu_div) / np.log(2)
                eu_Linf_eoc = np.log(eu_Linf_prev/eu_Linf) / np.log(2)
                ep_L2_eoc   = np.log(ep_L2_prev/ep_L2) / np.log(2)
                if direct:
                    history.append((hmin, hmax, mesh.topology.index_map(gdim).size_global, W.dofmap.index_map.size_global, f'{eu_L2:.1E}', f'{eu_L2_eoc:.2f}', f'{eu_H1:.1E}', f'{eu_H1_eoc:.2f}', f'{eu_div:.1E}', f'{eu_div_eoc:.2f}', f'{eu_Linf:.1E}', f'{eu_Linf_eoc:.2f}', f'{div_uh:.1E}', f'{ep_L2:.1E}', f'{ep_L2_eoc:.2f}'))
                else:
                    history.append((hmin, hmax, mesh.topology.index_map(gdim).size_global, W.dofmap.index_map.size_global, f'{eu_L2:.1E}', f'{eu_L2_eoc:.2f}', f'{eu_H1:.1E}', f'{eu_H1_eoc:.2f}', f'{eu_div:.1E}', f'{eu_div_eoc:.2f}', f'{eu_Linf:.1E}', f'{eu_Linf_eoc:.2f}', f'{div_uh:.1E}', f'{ep_L2:.1E}', f'{ep_L2_eoc:.2f}', niters, rnorm))
            
            eu_L2_prev   = eu_L2
            eu_H1_prev   = eu_H1
            eu_div_prev  = eu_div
            eu_Linf_prev = eu_Linf
            ep_L2_prev   = ep_L2
            print(tabulate.tabulate(history, headers=headers))
    
    else: raise ValueError('Unknown mesh_type.')
    print(f"Script time = {time.perf_counter() - tic:.2f} sec")
    vtx_u = dfx.io.VTXWriter(comm=mesh.comm, filename="MMS_BDM_velocity.bp", output=[uh_out], engine="BP4")
    vtx_u.write(0)
    vtx_u.close()