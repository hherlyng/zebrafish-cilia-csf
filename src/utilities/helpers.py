import ufl

import numpy     as np
import dolfinx   as dfx

from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element

def nonmatching_mesh_interpolation(mesh0: dfx.mesh.Mesh,
                                   mesh1: dfx.mesh.Mesh, 
                                   u: dfx.fem.Function, 
                                   vector: bool=False, 
                                   write_output: bool=False,
                                   xtype=np.float64):
    """ Interpolate the function u onto a rectangular slice that lies in the yz-plane.

    Parameters
    ----------
    mesh0 : dfx.mesh.Mesh
        Brain ventricles mesh.
    mesh1 : dfx.mesh.Mesh
        Rectangle mesh.
    u : dfx.fem.Function
        Finite element function to be interpolated.
    vector : bool, optional
        Use vector finite elements if True, by default False.
    write_output : bool, optional
        Write the function u and the rectangle interpolation of u to files if True, by default False.
    xtype : numpy dtype, optional
        Numpy datatype, by default np.float64.

    Returns
    -------
    dfx.fem.Function
        interpolation of function u onto the rectangle slice.
    """
    el0 = element("DG", mesh0.basix_cell(), 1, shape=(3,), dtype=xtype) if vector else element("DG", mesh0.basix_cell(), 1, dtype=xtype)
    V0 = dfx.fem.functionspace(mesh0, el0)
    el1 = element("DG", mesh1.basix_cell(), 1, shape=(3,), dtype=xtype) if vector else element("DG", mesh1.basix_cell(), 1, dtype=xtype)
    V1 = dfx.fem.functionspace(mesh1, el1)
    padding = 1e-14

    # Check that both interfaces of create nonmatching meshes interpolation data returns the same
    mesh1_cell_map = mesh1.topology.index_map(mesh1.topology.dim)
    num_cells_on_proc = mesh1_cell_map.size_local + mesh1_cell_map.num_ghosts
    cells = np.arange(num_cells_on_proc, dtype=np.int32)
    interpolation_data = dfx.fem.create_interpolation_data(V1, V0, cells, padding=padding)

    # Interpolate 3D->2D
    u1 = dfx.fem.Function(V1, dtype=xtype)
    u1.interpolate_nonmatching(u, cells, interpolation_data=interpolation_data)
    u1.x.scatter_forward()
    
    if write_output:
        # Write the functions to file to visualize
        string = "_vector" if vector else "_scalar"
        with dfx.io.VTXWriter(mesh1.comm, f"2D_function{string}.bp", [u1], "BP4") as vtx:
            vtx.write(0)
        
        with dfx.io.VTXWriter(mesh0.comm, f"3D_function{string}.bp", [u], "BP4") as vtx:
            vtx.write(0)
    
    # # Check that the interpolation is correct
    # nz_dofs = np.where(u1.x.array!=0.0)[0] # only check dofs where plane intersection is interior to the mesh
    # assert np.allclose(
    #     u.x.array[nz_dofs],
    #     u1.x.array[nz_dofs],
    #     rtol=np.sqrt(np.finfo(xtype).eps),
    #     atol=np.sqrt(np.finfo(xtype).eps),
    # )

    return u1

def calc_error_L2(u_approx: dfx.fem.Function, u_exact: dfx.fem.Function, dX: ufl.Measure, vector_elements: bool=False, degree_raise: int=3) -> float:
        """ Calculate the L2 error for a solution approximated with finite elements.

        Parameters
        ----------
        u_approx : dolfinx.fem.Function
            The solution function approximated with finite elements.

        u_exact : dolfinx.fem.Function
            The exact solution function.

        dX : ufl.Measure
            Volume integral measure.

        vector_elements : bool, optional
            Use vector finite elements if True, by default False.

        degree_raise : int, optional
            The amount of polynomial degrees that the approximated solution
            is refined, by default 3.

        Returns
        -------
        error_global : float
            The L2 error norm.
        """
        # Create higher-order function space for solution refinement
        degree = u_approx.function_space.ufl_element().degree()
        family = u_approx.function_space.ufl_element().family()
        mesh   = u_approx.function_space.mesh
        if vector_elements:
            # Create higher-order function space based on vector elements
            W = dfx.fem.FunctionSpace(mesh, ufl.VectorElement(family = family, 
                                      degree = (degree + degree_raise), cell = mesh.ufl_cell()))
        else:
            # Create higher-order funciton space based on finite elements
            W = dfx.fem.FunctionSpace(mesh, (family, degree + degree_raise))
            
        # Interpolate the approximate solution into the refined space
        u_W = dfx.fem.Function(W)
        u_W.interpolate(u_approx)

        # Interpolate exact solution, special handling if exact solution
        # is a ufl expression or a python lambda function
        u_exact_W = dfx.fem.Function(W)

        if isinstance(u_exact, ufl.core.expr.Expr):
            u_expr = dfx.fem.Expression(u_exact, W.element.interpolation_points())
            u_exact_W.interpolate(u_expr)
        else:
            u_exact_W.interpolate(u_exact)
        
        # Compute the error in the higher-order function space
        e_W = dfx.fem.Function(W)
        e_W.x.array[:] = u_W.x.array - u_exact_W.x.array

        # Integrate the error
        error        = dfx.fem.form(ufl.inner(e_W, e_W) * dX)
        error_local  = dfx.fem.assemble_scalar(error)
        error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
        error_global = np.sqrt(error_global)

        return error_global

def calc_error_H1(u_approx: dfx.fem.Function, u_exact: dfx.fem.Function, dX: ufl.Measure, vector_elements: bool = False, degree_raise: int = 3) -> float:
        """ Calculate the H1 error for a solution approximated with finite elements.

        Parameters
        ----------
        u_approx : dolfinx.fem Function
            The solution function approximated with finite elements.

        u_exact : dolfinx.fem Function
            The exact solution function.

        dX : ufl.Measure
            Volume integral measure.

        degree_raise : int, optional
            The amount of polynomial degrees that the approximated solution
            is refined, by default 3

        Returns
        -------
        error_global : float
            The H1 error norm.
        """
        # Create higher-order function space for solution refinement
        degree = u_approx.function_space.ufl_element().degree()
        family = u_approx.function_space.ufl_element().family()
        mesh   = u_approx.function_space.mesh
        
        if vector_elements:
            # Create higher-order function space based on vector elements
            W = dfx.fem.FunctionSpace(mesh, ufl.VectorElement(family=family, 
                                      degree=(degree+degree_raise), cell=mesh.ufl_cell()))
        else:
            # Create higher-order funciton space based on finite elements
            W = dfx.fem.FunctionSpace(mesh, (family, degree+degree_raise))
            
        # Interpolate the approximate solution into the refined space
        u_W = dfx.fem.Function(W)
        u_W.interpolate(u_approx)

        # Interpolate exact solution, special handling if exact solution
        # is a ufl expression or a python lambda function
        u_exact_W = dfx.fem.Function(W)

        if isinstance(u_exact, ufl.core.expr.Expr):
            u_expr = dfx.fem.Expression(u_exact, W.element.interpolation_points())
            u_exact_W.interpolate(u_expr)
        else:
            u_exact_W.interpolate(u_exact)
        
        # Compute the error in the higher-order function space
        e_W = dfx.fem.Function(W)
        e_W.x.array[:] = u_W.x.array - u_exact_W.x.array
        
        # Integrate the error
        error        = dfx.fem.form(ufl.inner(ufl.grad(e_W), ufl.grad(e_W)) * dX)
        error_local  = dfx.fem.assemble_scalar(error)
        error_global = mesh.comm.allreduce(error_local, op = MPI.SUM)
        error_global = np.sqrt(error_global)

        return error_global

def calc_error_Hdiv(u_approx: dfx.fem.Function, u_exact: dfx.fem.Function, dX: ufl.Measure, vector_elements: bool = False, degree_raise: int = 3) -> float:
        """ Calculate the Hdiv error for a solution approximated with finite elements.

        Parameters
        ----------
        u_approx : dolfinx.fem Function
            The solution function approximated with finite elements.

        u_exact : dolfinx.fem Function
            The exact solution function.

        dX : ufl.Measure
            Volume integral measure.

        degree_raise : int, optional
            The amount of polynomial degrees that the approximated solution
            is refined, by default 3

        Returns
        -------
        error_global : float
            The Hdiv error norm.
        """
        # Create higher-order function space for solution refinement
        degree = u_approx.function_space.ufl_element().degree()
        family = u_approx.function_space.ufl_element().family()
        mesh   = u_approx.function_space.mesh
        
        if vector_elements:
            # Create higher-order function space based on vector elements
            W = dfx.fem.FunctionSpace(mesh, ufl.VectorElement(family=family, 
                                      degree=(degree + degree_raise), cell=mesh.ufl_cell()))
        else:
            # Create higher-order funciton space based on finite elements
            W = dfx.fem.FunctionSpace(mesh, (family, degree+degree_raise))
            
        # Interpolate the approximate solution into the refined space
        u_W = dfx.fem.Function(W)
        u_W.interpolate(u_approx)

        # Interpolate exact solution, special handling if exact solution
        # is a ufl expression or a python lambda function
        u_exact_W = dfx.fem.Function(W)

        if isinstance(u_exact, ufl.core.expr.Expr):
            u_expr = dfx.fem.Expression(u_exact, W.element.interpolation_points())
            u_exact_W.interpolate(u_expr)
        else:
            u_exact_W.interpolate(u_exact)
        
        # Compute the error in the higher-order function space
        e_W = dfx.fem.Function(W)
        e_W.x.array[:] = u_W.x.array - u_exact_W.x.array
        
        # Integrate the error
        error    = dfx.fem.form(ufl.inner(ufl.div(e_W), ufl.div(e_W)) * dX 
                              + ufl.inner(e_W, e_W) * dX)
        error_local  = dfx.fem.assemble_scalar(error)
        error_global = mesh.comm.allreduce(error_local, op = MPI.SUM)
        error_global = np.sqrt(error_global)

        return error_global

def calc_mean_velocity(mesh, u):
    dx_omega = ufl.Measure("dx", domain = mesh)
    volume   = dfx.fem.assemble_scalar(dfx.fem.form(1 * dx_omega))
    velocity = dfx.fem.assemble_scalar(dfx.fem.form(ufl.sqrt(ufl.dot(u, u)) * dx_omega))

    avg_velocity = velocity / volume

    return avg_velocity

def calc_mean_pressure(mesh, p):
    dx_omega = ufl.Measure("dx", domain = mesh)
    volume   = dfx.fem.assemble_scalar(dfx.fem.form(1 * dx_omega))
    pressure = dfx.fem.assemble_scalar(dfx.fem.form(p * dx_omega))
    
    avg_pressure = pressure / volume

    return avg_pressure

def write_to_file(filename, param, u_bar, p_bar):
    file = open(filename, "a")
    file.write(str(param) + '\t' + str(u_bar) + '\t' + str(p_bar) + '\n')
    file.close()

def tangential_projection(u: ufl.Coefficient, n: ufl.FacetNormal) -> ufl.Coefficient:
    """
    See for instance:
    https://link.springer.com/content/pdf/10.1023/A:1022235512626.pdf
    """
    return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u

def tangential_projection_unit(u: ufl.Coefficient, n: ufl.FacetNormal) -> ufl.Coefficient:
    t = (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u
    t /= ufl.sqrt(ufl.dot(t, t))

    return t

def convert_to_unit_tangential_vector(vec: ufl.Coefficient, n: ufl.FacetNormal):
    vec_tang = tangential_projection(vec, n)
    vec_tang_length = ufl.sqrt(ufl.dot(vec_tang, vec_tang))
    unit_vec_tang = vec_tang / vec_tang_length
    
    return unit_vec_tang

def facet_vector_approximation(V: dfx.fem.FunctionSpace,
                               mt: dfx.mesh.MeshTags,
                               mt_id: int, tangent = False,
                               jit_options: dict = {},
                               form_compiler_options: dict = {}):
    """
    Approximate the facet normal by projecting it into the function space for a set of facets.

    Args:
        V: The function space to project into
        mt: The `dolfinx.mesh.MeshTags` containing facet markers
        mt_id: The id for the facets in `mt` we want to represent the normal at
        tangent: To approximate the tangent to the facet set this flag to `True`
        jit_options: Parameters used in CFFI JIT compilation of C code generated by FFCx.
            See https://github.com/FEniCS/dolfinx/blob/main/python/dolfinx/jit.py#L22-L37
            for all available parameters. Takes priority over all other parameter values.
        form_compiler_options: Parameters used in FFCx compilation of this form. Run `ffcx - -help` at
            the commandline to see all available options. Takes priority over all
            other parameter values, except for `scalar_type` which is determined by
            DOLFINx.
"""
    timer = dfx.common.Timer("~MPC: Facet normal projection")
    comm  = V.mesh.comm
    n     = ufl.FacetNormal(V.mesh)
    nh    = dfx.fem.Function(V)
    u, v  = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds    = ufl.ds(domain = V.mesh)#, subdomain_data = mt, subdomain_id = mt_id)
    
    if tangent:
        if V.mesh.geometry.dim == 1:
            raise ValueError("Tangent not defined for 1D problem")
        elif V.mesh.geometry.dim == 2:
            a = ufl.inner(u, v) * ds
            L = ufl.inner(ufl.as_vector([-n[1], n[0]]), v) * ds
        else:
            c = dfx.fem.Constant(V.mesh, (1.0, 1.0, 1.0))
            a = ufl.inner(u, v) * ds
            L = ufl.inner(tangential_projection(c, n), v) * ds
    else:
        a = ufl.inner(u, v) * ds
        L = ufl.inner(n, v) * ds

    # Find all dofs that are not boundary dofs
    imap = V.dofmap.index_map
    all_blocks = np.arange(imap.size_local, dtype=np.int32)
    facets = dfx.mesh.exterior_facet_indices(V.mesh.topology) #mt.find(mt_id))
    top_blocks = dfx.fem.locate_dofs_topological(V, V.mesh.topology.dim - 1, facets)
    deac_blocks = all_blocks[np.isin(all_blocks, top_blocks, invert=True)]

    # Note there should be a better way to do this
    # Create sparsity pattern only for constraint + bc
    bilinear_form = dfx.fem.form(a, jit_options=jit_options,
                              form_compiler_options=form_compiler_options)
    pattern = dfx.fem.create_sparsity_pattern(bilinear_form)
    pattern.insert_diagonal(deac_blocks)
    pattern.assemble()
    u_0 = dfx.fem.Function(V)
    u_0.vector.set(0)

    bc_deac = dfx.fem.dirichletbc(u_0, deac_blocks)
    A = dfx.cpp.la.petsc.create_matrix(comm, pattern)
    A.zeroEntries()

    # Assemble the matrix with all entries
    form_coeffs = dfx.cpp.fem.pack_coefficients(bilinear_form)
    form_consts = dfx.cpp.fem.pack_constants(bilinear_form)
    dfx.cpp.fem.petsc.assemble_matrix(A, bilinear_form, form_consts, form_coeffs, [bc_deac])
    if bilinear_form.function_spaces[0] is bilinear_form.function_spaces[1]:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dfx.cpp.fem.petsc.insert_diagonal(A, bilinear_form.function_spaces[0], [bc_deac], 1.0)
    A.assemble()
    linear_form = dfx.fem.form(L, jit_options=jit_options,
                            form_compiler_options=form_compiler_options)
    b = dfx.fem.petsc.assemble_vector(linear_form)

    dfx.fem.petsc.apply_lifting(b, [bilinear_form], [[bc_deac]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dfx.fem.petsc.set_bc(b, [bc_deac])

    # Solve Linear problem
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType("cg")
    solver.rtol = 1e-8
    solver.setOperators(A)
    solver.solve(b, nh.vector)
    nh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    timer.stop()
    u_norm = ufl.sqrt(ufl.inner(nh,nh))
    nh_normalized = dfx.fem.Expression(ufl.as_vector((nh[0]/u_norm, nh[1]/u_norm, nh[2]/u_norm)), V.element.interpolation_points())
    n_out = dfx.fem.Function(V)

    return n_out

def boundary_vector_approximation(V: dfx.fem.FunctionSpace, vec, mt: dfx.mesh.MeshTags, mt_id: int,
                                  jit_options: dict = {}, form_compiler_options: dict = {}):
    """
    Approximate a vector on the boundary of a mesh.

    Args:
        V: The function space to project into.
        vec: The vector to approximate.
        mt: The `dolfinx.mesh.MeshTags` containing facet markers.
        mt_id: The id for the facets in `mt` we want to represent the normal at
        tangent: To approximate the tangent to the facet set this flag to `True`
        jit_options: Parameters used in CFFI JIT compilation of C code generated by FFCx.
            See https://github.com/FEniCS/dolfinx/blob/main/python/dolfinx/jit.py#L22-L37
            for all available parameters. Takes priority over all other parameter values.
        form_compiler_options: Parameters used in FFCx compilation of this form. Run `ffcx - -help` at
            the commandline to see all available options. Takes priority over all
            other parameter values, except for `scalar_type` which is determined by
            DOLFINx.
"""

    timer = dfx.common.Timer("~MPC: Facet normal projection")
    comm  = V.mesh.comm
    vec_h = dfx.fem.Function(V)
    u, v  = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds    = ufl.ds(domain = V.mesh, subdomain_data=mt, subdomain_id=mt_id)
    a     = ufl.inner(u, v) * ds
    L     = ufl.inner(vec, v) * ds

    # Find all dofs that are not boundary dofs
    imap = V.dofmap.index_map
    all_blocks = np.arange(imap.size_local, dtype=np.int32)
    top_blocks = dfx.fem.locate_dofs_topological(V, V.mesh.topology.dim - 1, mt.find(mt_id))
    deac_blocks = all_blocks[np.isin(all_blocks, top_blocks, invert=True)]

    # Note there should be a better way to do this
    # Create sparsity pattern only for constraint + bc
    bilinear_form = dfx.fem.form(a, jit_options=jit_options,
                              form_compiler_options=form_compiler_options)
    pattern = dfx.fem.create_sparsity_pattern(bilinear_form)
    pattern.insert_diagonal(deac_blocks)
    pattern.assemble()
    u_0 = dfx.fem.Function(V)
    u_0.vector.set(0)

    bc_deac = dfx.fem.dirichletbc(u_0, deac_blocks)
    A = dfx.cpp.la.petsc.create_matrix(comm, pattern)
    A.zeroEntries()

    # Assemble the matrix with all entries
    form_coeffs = dfx.cpp.fem.pack_coefficients(bilinear_form)
    form_consts = dfx.cpp.fem.pack_constants(bilinear_form)
    dfx.cpp.fem.petsc.assemble_matrix(A, bilinear_form, form_consts, form_coeffs, [bc_deac])
    if bilinear_form.function_spaces[0] is bilinear_form.function_spaces[1]:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dfx.cpp.fem.petsc.insert_diagonal(A, bilinear_form.function_spaces[0], [bc_deac], 1.0)
    A.assemble()
    linear_form = dfx.fem.form(L, jit_options=jit_options,
                            form_compiler_options=form_compiler_options)
    b = dfx.fem.petsc.assemble_vector(linear_form)

    dfx.fem.petsc.apply_lifting(b, [bilinear_form], [[bc_deac]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dfx.fem.petsc.set_bc(b, [bc_deac])

    # Solve Linear problem
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType("cg")
    solver.rtol = 1e-8
    solver.setOperators(A)
    solver.solve(b, vec_h.vector)
    vec_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    timer.stop()
    return vec_h