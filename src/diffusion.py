import ufl

import numpy   as np
import dolfinx as dfx

from ufl               import dot, grad, avg, jump
from mpi4py            import MPI
from pathlib           import Path
from petsc4py          import PETSc
from basix.ufl         import element
from utilities.mesh    import create_ventricle_volumes_meshtags
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, apply_lifting, set_bc

print = PETSc.Sys.Print

# Set compiler options for runtime optimization
cache_dir = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                   "cache_dir"               : cache_dir,
                   "cffi_libraries"          : ["m"]
}

def diffusion_problem(mesh: dfx.mesh.Mesh, k: int, D_value: float):
    ''' Solve the diffusion equation for two timesteps with
        the implicit Euler timestepping scheme, using a diffusion
        coefficient that is 10000 times larger than D_value.
    '''

    comm = mesh.comm
    # Problem parameter values
    t  = 0.0  # Simulation start time [s]
    f = 2.22 # Cardiac frequency [Hz]
    period = 1 / f # Cardiac period [s]
    T  = 1900*period  # Simulation end time [s]
    dt = period / 20 # Timestep size [s]

    c_tilde  = 1.0
    a_growth = 65.0

    def photoconversion_curve(t):
        return c_tilde*np.log(1+t/a_growth)/np.log(1+T/a_growth)

    # Convert to compiled variables
    deltaT = dfx.fem.Constant(mesh, dfx.default_scalar_type(dt))
    D = dfx.fem.Constant(mesh, 1e5*D_value) # Diffusion coefficient [mm^2/s]

    el = element("Discontinuous Lagrange", mesh.basix_cell(), k) # Continuous Lagrange finite elements, polynomial order k
    W = dfx.fem.functionspace(mesh, el)

    c_h = dfx.fem.Function(W) # Function for storing the solution
    c_  = dfx.fem.Function(W) # Function for storing the solution at the previous timestep
    c__ = dfx.fem.Function(W)
    c_rhs = dfx.fem.Function(W)

    # Trial and test functions
    c, w = ufl.TrialFunction(W), ufl.TestFunction(W)

    # Integral measures
    dx = ufl.Measure('dx', domain=mesh)
    dS = ufl.Measure('dS', domain=mesh)

    # Mesh properties
    hf = ufl.CellDiameter(domain=mesh)
    n  = ufl.FacetNormal(domain=mesh)

    # SIPG penalty parameter 
    alpha = dfx.fem.Constant(mesh, dfx.default_scalar_type(25.0))

    # Set Dirichlet BC on sub-volume
    ROI_ct, ROI_tags = create_ventricle_volumes_meshtags(mesh=mesh) # Create meshtags for the different ventricle regions
    ROI1_indices = ROI_ct.find(ROI_tags[0])
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
    ROI1_dofs = dfx.fem.locate_dofs_topological(W, mesh.topology.dim, ROI1_indices)
    bc_func = dfx.fem.Function(W)
    bc_func.x.array[:] = photoconversion_curve(t)
    bc = dfx.fem.dirichletbc(bc_func, ROI1_dofs)
    bcs = [bc]
    
    # The variational problem
    # Bilinear form
    a0 = c * w / deltaT * dx

    a1 = dot(grad(w), D * grad(c)) * dx # Flux term integrated by parts

    # Diffusive terms with interior penalization
    a2  = D('+') * alpha('+') / avg(hf) * dot(jump(w, n), jump(c, n)) * dS
    a2 -= D('+') * dot(avg(grad(w)), jump(c, n)) * dS
    a2 -= D('+') * dot(jump(w, n), avg(grad(c))) * dS

    a = a0+a1+a2

    L  = c_rhs * w / deltaT * dx # Time derivative

    # Compile forms
    a_cpp = dfx.fem.form(a, jit_options=jit_parameters)
    L_cpp = dfx.fem.form(L, jit_options=jit_parameters)

    # Assemble matrix and create RHS vector
    A = assemble_matrix(a_cpp, bcs=bcs)
    A.assemble()
    b = create_vector(L_cpp)

    # Create and configure linear solver
    solver_options = {'ksp_type' : 'preonly',
                      'pc_type'  : 'lu', 
                      'factor_solver_type' : 'mumps'
    }
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(solver_options['ksp_type'])
    solver.getPC().setType(solver_options['pc_type'])
    solver.getPC().setFactorSolverType(solver_options['factor_solver_type'])
    solver.getPC().getFactorMatrix().setMumpsIcntl(icntl=58, ival=1) # activate symbolic factorization

    def assemble_RHS(b, a_cpp, L_cpp, bcs):
        with b.localForm() as b_loc: b_loc.set(0) # Zero values from previous timestep
        assemble_vector(b, L_cpp) # Assemble the vector
        apply_lifting(b, [a_cpp], bcs=[bcs]) # Apply boundary condition lifting
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) # Perform ghost update: retrieve
        set_bc(b, bcs=bcs) # Set boundary conditions
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD) # Perform ghost update: send

    # The first solve
    t += dt # Increment time
    bc_func.x.array[:] = photoconversion_curve(t) # Update BC
    assemble_RHS(b, a_cpp, L_cpp, bcs) # Assemble right-hand side vector

    # Linear solve
    solver.solve(b, c_h.x.petsc_vec)
    c_h.x.scatter_forward() # Ghost update
    c__.x.array[:] = c_h.x.array.copy() # previous-previous timestep solution
    c_rhs.x.array[:] = c_h.x.array.copy() # The right-hand side concentration

    max_c = comm.allreduce(c_h.x.array.max(), op=MPI.MAX)
    min_c = comm.allreduce(c_h.x.array.min(), op=MPI.MIN)

    # Print stuff
    print("Maximum concentration: ", max_c)
    print("Minimum concentration: ", min_c)

    total_c_form = dfx.fem.form(c_h*dx)
    total_c = dfx.fem.assemble_scalar(total_c_form)
    total_c = comm.allreduce(total_c, op=MPI.SUM)
    print(f"Total concentration: {total_c:.2e}")

    # The second solve
    t += dt # Increment time
    bc_func.x.array[:] = photoconversion_curve(t) # Update BC
    assemble_RHS(b, a_cpp, L_cpp, bcs) # Assemble right-hand side vector

    # Linear solve
    solver.solve(b, c_h.x.petsc_vec)
    c_h.x.scatter_forward() # Ghost update
    c_.x.array[:] = c_h.x.array.copy() # previous timestep solution

    max_c = comm.allreduce(c_h.x.array.max(), op=MPI.MAX)
    min_c = comm.allreduce(c_h.x.array.min(), op=MPI.MIN)

    # Print stuff
    print("Maximum concentration: ", max_c)
    print("Minimum concentration: ", min_c)

    total_c = dfx.fem.assemble_scalar(total_c_form)
    total_c = comm.allreduce(total_c, op=MPI.SUM)
    print(f"Total concentration: {total_c:.2e}")

    return c__, c_

if __name__=='__main__':
    ''' Solve the diffusion problem on the original mesh. '''
    with dfx.io.XDMFFile(MPI.COMM_WORLD, '../geometries/standard/original_ventricles.xdmf', 'r') as xdmf:
        mesh = xdmf.read_mesh()
        
    c_h_, c_h = diffusion_problem(mesh, 1, 2e-6)
    with dfx.io.VTXWriter(mesh.comm, '../output/diffusion.bp', [c_h_, c_h], engine='BP4') as vtx:
        vtx.write(0)