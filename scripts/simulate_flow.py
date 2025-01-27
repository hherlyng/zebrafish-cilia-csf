import ufl
import time

import numpy         as np
import dolfinx       as dfx
import adios4dolfinx as a4d

from sys       import argv
from ufl       import div, dot, inner, sym, grad, avg
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element, mixed_element
from dolfinx.fem.petsc import assemble_matrix, _assemble_vector_vec, set_bc, apply_lifting, create_vector

from utilities.mesh                 import recal_mark_boundaries_flow, recal_anterior_cilia_volume, recal_middle_ventral_cilia_volume
from utilities.forcing_expressions  import OscillatoryPressure
from utilities.normals_and_tangents import facet_vector_approximation

print = PETSc.Sys.Print

# Mesh tags for flow
ANTERIOR_PRESSURE    = 2 # The pressure BC facets on the anterior ventricle boundary
POSTERIOR_PRESSURE   = 3 # The pressure BC facets on the posterior ventricle boundary
MIDDLE_VENTRAL_CILIA = 5 # The cilia BC facets on the ventral wall of the middle ventricle
MIDDLE_DORSAL_CILIA  = 6 # The cilia BC facets on the dorsal wall of the middle ventricle
ANTERIOR_CILIA       = 7 # The cilia BC facets on the dorsal, anterior walls of the anterior ventricle
ANTERIOR_CILIA2      = 8 # The cilia BC facets on the dorsal, posterior walls of the anterior ventricle
SLIP                 = 9 # The free-slip facets of the boundary

# Operators
# NOTE: these are the jump operators from Krauss, Zikatonov paper.
# Jump is just a difference and it preserves the rank 
Jump = lambda arg: arg('+') - arg('-')

# Average uses dot with normal and AGAIN MINUS; it reduces the rank
Avg = lambda arg, n: .5*(dot(arg('+'), n('+')) - dot(arg('-'), n('-')))

# Action of (1 - n x n) on a vector yields the tangential component
Tangent = lambda v, n: v - n*dot(v, n)

# Symmetric gradient
eps = lambda u: sym(grad(u))

# Velocity of reference frame
class ReferenceFrameVelocity:
    """ Class for generating a velocity expression used to prescribe the motion of the reference frame. """

    def __init__(self, amp, freq):
        self.t = 0
        self.amp = amp
        self.omega = 2*np.pi*freq

    def __call__(self, x):
        return self.amp * np.cos(self.omega*self.t) * np.stack((np.ones (x.shape[1]),
                                                                np.zeros(x.shape[1]),
                                                                np.zeros(x.shape[1])))

class FlowSolver:

    #----------CLASS CONSTANTS AND PARAMETERS----------#
    # Material parameters
    nu_val  = 0.697 # Kinematic viscosity [mm^2 / s]
    rho_val = 1e-3  # Fluid density [g / mm^3]

    # Model parameters
    tau  = 6.5e-4 # Tangential stress BC parameter [Pa]
    freq = 2.22   # Cardiac frequency [Hz]
    A_pressure   = 0.0015 # Pressure BC amplitude [Pa]
    A_moving_ref = 0.75*np.sqrt(3/2) # Moving reference frame velocity amplitude

    # Finite elements
    BDM_penalty_val = 10.0 # BDM interior penalty parameter

    def __init__(self, mesh: dfx.mesh.Mesh, ft: dfx.mesh.MeshTags, direct: bool, model_version: str, mesh_version: int,
                 T: float, dt: float, relative: bool, write_output: bool=False, write_checkpoint: bool=False):
        """ Constructor.

        Parameters
        ----------
        mesh : dfx.mesh.Mesh
            Computational mesh.
        
        ft : dfx.mesh.MeshTags
            Facet tags for the mesh.
        
        direct : bool
            Use a direct solver if True, else use an iterative solver.
        
        model_version : str
            The problem setup considered, either model A, B or C.
                - Model A: only cilia forces
                - Model B: only cardiac forces
                - Model C: cilia and cardiac forces, i.e. A+B
        
        mesh_version : str
            The mesh version used, as specified by the mesh filename prefix.
        
        T : float
            Simulation end time in seconds.
        
        dt : float
            Timestep size in seconds.
        
        relative : bool
            How to model pulsatile cardiac motion. Use moving reference frame formulation if True, else use pressure boundary conditions.
        
        write_output : bool
            Write velocity to VTX file if True.
        
        write_checkpoint : bool
            Write velocity to adios4dolfinx checkpoint file if True. 
        """
        
        self.mesh = mesh
        self.ft = ft
        self.use_direct_solver = direct
        self.model_version = model_version
        self.mesh_version = mesh_version
        self.T = T
        self.dt = dt
        self.relative = relative
        self.write_output = write_output
        self.write_checkpoint = write_checkpoint

    def stabilization(self, u: ufl.TrialFunction, v: ufl.TestFunction, consistent: bool=True):
        """ Displacement/Flux Stabilization term from Krauss et al paper. 

        Parameters
        ----------
        u : ufl.TrialFunction
            The finite element trial function.
        
        v : ufl.TestFunction
            The finite element test function.
        
        consistent : bool
            Add symmetric gradient terms to the form if True.

        Returns
        -------
        ufl.Coefficient
            Stabilization term for the bilinear form.
        """

        n, hA = ufl.FacetNormal(self.mesh), avg(ufl.CellDiameter(self.mesh)) # Facet normal vector and average cell diameter
        dS = ufl.Measure('dS', domain=self.mesh) # Interior facet integral measure

        if consistent: # Add symmetrization terms
            return (-inner(Avg(2*self.mu*eps(u), n), Jump(Tangent(v, n)))*dS
                    -inner(Avg(2*self.mu*eps(v), n), Jump(Tangent(u, n)))*dS
                    + 2*self.mu*(self.penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS)

        # For preconditioning
        return 2*self.mu*(self.penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS

    def assemble_RHS_vector(self):
        """ Assemble the right-hand side vector of the linear system. """

        with self.b.localForm() as loc: loc.set(0) # Zero-out previous entries
        _assemble_vector_vec(self.b, self.L_cpp) # Assemble the vector
        apply_lifting(self.b, [self.a_cpp], bcs=[self.bcs]) # Apply lifting of boundary conditions
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) # Ghost update, receive
        set_bc(self.b, bcs=self.bcs) # Set the boundary conditions in the vector
        self.b.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD) # Ghost update, send

    def post_process(self):
        """ Perform post processing of the numerical solution: Prints the maximum and minimum values of
        the divergence of the velocity field. Also prints the value of integrating the normal velocity component
        over the boundary of the mesh. """

        # Calculate the minimum and maximum values of the normal velocity component on the mesh boundary
        n_h = facet_vector_approximation(V=self.uh_out.function_space, mt=self.ft) # Finite element approximation of the facet normal vector on the mesh boundary
        u_dot_n = dfx.fem.Expression(dot(self.uh_out, n_h), self.uh_out.function_space.element.interpolation_points()) # The velocity component normal to the boundary
        u_dot_n_f = dfx.fem.Function(dfx.fem.functionspace(self.mesh, element("Lagrange", self.mesh.basix_cell(), 1)))
        u_dot_n_f.interpolate(u_dot_n)
        u_dot_n_min = u_dot_n_f.x.array[np.invert(np.isnan(u_dot_n_f.x.array))].min()
        u_dot_n_max = u_dot_n_f.x.array[np.invert(np.isnan(u_dot_n_f.x.array))].max()

        # Calculate total mass flux through the mesh boundary
        n  = ufl.FacetNormal(self.mesh) # The facet normal of the mesh
        ds = ufl.Measure('ds', domain=self.mesh) # Boundary integral measure
        u_dot_n_integrated = dfx.fem.assemble_scalar(dfx.fem.form(dot(self.uh_out, n) * ds))

        # Parallel communication
        u_dot_n_min = self.mesh.comm.allreduce(u_dot_n_min, op=MPI.MIN)
        u_dot_n_max = self.mesh.comm.allreduce(u_dot_n_max, op=MPI.MAX)
        u_dot_n_integrated = self.mesh.comm.allreduce(u_dot_n_integrated, op=MPI.SUM)
        
        # Print
        print(f"Mass conservation min: {u_dot_n_min:.2e}")
        print(f"Mass conservation max: {u_dot_n_max:.2e}")        
        print(f"Mass conservation integrated: {u_dot_n_integrated:.2e}")

    def setup(self):
        """ Set up the discrete variational problem of the Stokes equations,
            discretized with Brezzi-Douglas-Marini (velocity) and 
            Discontinuous Galerkin (pressure) elements. """

        # Define mesh and some general variables
        mesh = self.mesh
        dx = ufl.Measure("dx", mesh) # Cell integral measure
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.ft) # Boundary facet integral measure
        n  = ufl.FacetNormal(mesh) # Facet (outer) normal vector
        deltaT = dfx.fem.Constant(mesh, dfx.default_scalar_type(self.dt)) # Form-compiled timestep
        self.penalty = dfx.fem.Constant(mesh, dfx.default_scalar_type(self.BDM_penalty_val))

        # Fluid parameters
        rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(self.rho_val)) # Density [g / mm^3]
        self.mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(self.nu_val*self.rho_val)) # Dynamic viscosity [g / (mm * s)]

        # Finite elements
        k  = 1 # Velocity FEM order, pressure one order lower
        cell = mesh.basix_cell() # Mesh cell type
        Velm = element('BDM', cell, k)  # Velocity element: Brezzi-Douglas-Marini  of order k
        Qelm = element('DG', cell, k-1) # Pressure element: Discontinuous Galerkin of order k-1
        Welm = mixed_element([Velm, Qelm]) # Mixed velocity-pressure element
        self.W = W = dfx.fem.functionspace(mesh, Welm) # Mixed velocity-pressure function space
        self.V, _ = V, _ = W.sub(0).collapse() # Velocity subspace
        self.Q, _ = Q, _ = W.sub(1).collapse() # Pressure subspace

        # Trial and test functions
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)        
            
        self.u_ = dfx.fem.Function(V) # Velocity at previous timestep

        #---------------------------------------------------------------#
        # Define the stress vector used in the slip boundary conditions #
        #---------------------------------------------------------------#
        tau_vec   = self.tau*ufl.as_vector((1, 0, 1)) # Stress vector to be projected tangentially onto the mesh

        # Define coordinates used in the tau expressions
        xx, _, _ = ufl.SpatialCoordinate(mesh)
        x0_dorsal = 0.175
        xe_dorsal = 0.335

        x0_ventral = 0.155
        xe_ventral = 0.310

        # Define the tau expressions
        tau_vec_dorsal = 2.75*tau_vec * (xx - x0_dorsal) / (xe_dorsal - x0_dorsal) # Dorsal cilia lambda expression
        tau_vec_ventral = 0.5*tau_vec * (1 - (xx - x0_ventral) / (xe_ventral - x0_ventral)) # Ventral cilia lambda expression
        tau_vec_anterior1 = 0.4*tau_vec # Anterior cilia lambda expression (anterior part)
        tau_vec_anterior2 = tau_vec # Anterior cilia lambda expression (posterior part)

        # Use the tau expressions to define the tangent traction vectors
        tangent_traction_dorsal = lambda n: Tangent(tau_vec_dorsal, n)
        tangent_traction_ventral = lambda n: Tangent(tau_vec_ventral, n)
        tangent_traction_anterior1 = lambda n: Tangent(tau_vec_anterior1, n)
        tangent_traction_anterior2 = lambda n: Tangent(tau_vec_anterior2, n)

        # If using the ALE formulation to model pulsatile motion, create reference frame  
        # velocity expression and interpolate it into a finite element function
        if self.relative:
            self.u_ref_expr = ReferenceFrameVelocity(amp=self.A_moving_ref, freq=self.freq)
            self.u_ref = dfx.fem.Function(V)
            self.u_ref.interpolate(self.u_ref_expr)

        #------------BILINEAR FORM------------#
        a  = 0 # Initialize
        a += 2*self.mu*inner(eps(u), eps(v)) * dx # Viscous diffusion term
        a -= p * div(v) * dx # Pressure gradient term
        a -= q * div(u) * dx # Continuity equation term
        a += self.stabilization(u, v) # BDM stabilization terms

        if self.model_version in ['B', 'C']:
            # Add time-derivative term
            a += rho/deltaT*dot(u, v) * dx 

            # Pulsatile cardiac motion
            if self.relative: # Moving reference frame
                a += rho*inner(dot(self.u_ref, grad(u)), v) * dx # Convection with the moving reference frame velocity
            else: # Pressure BCs
                a -= self.mu*inner(dot(grad(u).T, n), v) * (ds(ANTERIOR_PRESSURE) + ds(POSTERIOR_PRESSURE)) # Transpose of velocity gradient term, ensures parallel flow at outlets

        #------------LINEAR FORM------------#
        L = 0 # Initialize
        
        if self.model_version=='A': # Only cilia forces
            # Add tangential traction
            L += -inner(Tangent(v, n), tangent_traction_anterior1(n)) * ds(ANTERIOR_CILIA)
            L +=  inner(Tangent(v, n), tangent_traction_anterior2(n)) * ds(ANTERIOR_CILIA2)
            L += -inner(Tangent(v, n), tangent_traction_dorsal(n)) * ds(MIDDLE_DORSAL_CILIA)
            L +=  inner(Tangent(v, n), tangent_traction_ventral(n)) * ds(MIDDLE_VENTRAL_CILIA)
        elif self.model_version=='B': # Only pulsatile cardiac motion
            # Add time-derivative term
            L += rho/deltaT*dot(self.u_, v) * dx 

            if not self.relative: # Impose pressure BCs weakly
                
                # Create pressure BC expression and interpolate it into a finite element function
                self.p_bc_expr = OscillatoryPressure(A=self.A_pressure, f=self.freq)
                self.p_bc = dfx.fem.Function(Q)
                self.p_bc.interpolate(self.p_bc_expr)

                # Add pressure term to the RHS
                L += inner(self.p_bc*n, v) * ds(ANTERIOR_PRESSURE) 
        elif self.model_version=='C': # Cilia forces + pulsatile cardiac motion
            # Add time-derivative term
            L += rho/deltaT*dot(self.u_, v) * dx

            # Add tangential traction
            L += -inner(Tangent(v, n), tangent_traction_anterior1(n)) * ds(ANTERIOR_CILIA)
            L +=  inner(Tangent(v, n), tangent_traction_anterior2(n)) * ds(ANTERIOR_CILIA2)
            L += -inner(Tangent(v, n), tangent_traction_dorsal(n)) * ds(MIDDLE_DORSAL_CILIA)
            L +=  inner(Tangent(v, n), tangent_traction_ventral(n)) * ds(MIDDLE_VENTRAL_CILIA)

            if not self.relative: # Impose pressure BCs weakly

                # Create pressure BC expression and interpolate it into a finite element function
                self.p_bc_expr = OscillatoryPressure(A=self.A_pressure, f=self.freq)
                self.p_bc = dfx.fem.Function(Q)
                self.p_bc.interpolate(self.p_bc_expr)

                # Add pressure term to the RHS
                L += inner(self.p_bc*n, v) * ds(ANTERIOR_PRESSURE)

        #---------------------------------------------------#
        # Impose impermeability boundary condition strongly #
        #---------------------------------------------------#
        # Create facet-cell connectivity
        mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim) 
        
        # Get the facets of the slip boundary
        imperm_bdry = np.concatenate((self.ft.find(ANTERIOR_CILIA),
                                      self.ft.find(ANTERIOR_CILIA2),
                                      self.ft.find(MIDDLE_DORSAL_CILIA),
                                      self.ft.find(MIDDLE_VENTRAL_CILIA),
                                      self.ft.find(SLIP)))
        
        # Get the slip boundary dofs
        u_bc_dofs = dfx.fem.locate_dofs_topological((W.sub(0), V), mesh.topology.dim-1, imperm_bdry)
        
        # Create BC velocity function with zero velocity (yields zero normal flow)
        self.u_bc_func = dfx.fem.Function(V)

        if self.relative: # Moving reference frame (ALE) formulation
            self.u_bc_func.interpolate(self.u_ref_expr) # Velocity normal to boundary must balance the velocity of the reference frame
        
        # Create boundary condition object
        u_bc = dfx.fem.dirichletbc(self.u_bc_func, u_bc_dofs, W.sub(0))
        self.bcs = [u_bc]

        # Compile bilinear and linear forms
        self.a_cpp, self.L_cpp = dfx.fem.form(a), dfx.fem.form(L)

        # Assemble system matrix and create RHS vector
        A = assemble_matrix(self.a_cpp, self.bcs)
        A.assemble()
        self.b = create_vector(self.L_cpp)

        #------------------------------------#
        #------------SOLVER SETUP------------#
        #------------------------------------#
        if not self.use_direct_solver: # Using iterative solver
            # Assemble preconditioner matrix
            a_prec = (2*self.mu*inner(eps(u), eps(v))*dx
                            + (1/self.mu)*inner(p, q)*dx)
            
            # Account for HDiv
            a_prec += self.stabilization(u, v, consistent=False)
            a_prec_cpp = dfx.fem.form(a_prec)
            B = assemble_matrix(a_prec_cpp, self.bcs)
            B.assemble()
        
        if self.model_version=='A' or self.relative: # Singular weak formulation
            # Create nullspace vector, nullspace=set of all constants
            ns_vec = A.createVecLeft()
            _, Q_dofs = W.sub(1).collapse()
            ns_vec.setValuesLocal(Q_dofs, np.ones(Q_dofs.__len__()))
            ns_vec.assemble()
            ns_vec.normalize()
            nullspace = PETSc.NullSpace().create(vectors=[ns_vec], comm=mesh.comm)
            
            # Check that the nullspace created is actually the nullspace of A
            assert(nullspace.test(A)) 

            # Set the nullspace of the system matrix
            A.setNullSpace(nullspace)
            if not self.use_direct_solver: A.setNearNullSpace(nullspace)
            
            # Assemble RHS vector b and orthogonalize it w.r.t. the nullspace
            self.assemble_RHS_vector()
            nullspace.remove(self.b)

        # Configure linear solver
        self.ksp = PETSc.KSP().create(mesh.comm)
        if self.use_direct_solver: # Using direct solver
            # Configure solver using mumps
            self.ksp.setOperators(A)
            self.ksp.setType("preonly")
            self.ksp.getPC().setType("lu")
            self.ksp.getPC().setFactorSolverType("mumps")
        else: # Using iterative solver
            # Configure solver using minres with preconditioner
            self.ksp.setOperators(A, B)
            opts = PETSc.Options()
            opts.setValue('ksp_type', 'minres')
            opts.setValue('ksp_rtol', 1E-12) # Important with high tolerance to achieve low div(u_h)               
            opts.setValue('ksp_view', None)
            opts.setValue('ksp_monitor_true_residual', None)                
            opts.setValue('ksp_converged_reason', None)
            opts.setValue('fieldsplit_0_ksp_type', 'preonly')
            opts.setValue('fieldsplit_0_pc_type', 'lu')
            opts.setValue('fieldsplit_1_ksp_type', 'preonly')
            opts.setValue('fieldsplit_1_pc_type', 'lu')           

            # Configure preconditioner with fieldsplit method
            pc = self.ksp.getPC()
            pc.setType(PETSc.PC.Type.FIELDSPLIT)

            # Create index sets
            _, V_dofs = W.sub(0).collapse()
            _, Q_dofs = W.sub(1).collapse()
            is_V = PETSc.IS().createGeneral(V_dofs)
            is_Q = PETSc.IS().createGeneral(Q_dofs)

            # Set field split index sets
            pc.setFieldSplitIS(('0', is_V), ('1', is_Q))
            pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE) 

            # Finalize setup
            self.ksp.setUp()
            pc.setFromOptions()
            self.ksp.setFromOptions()

        #------------------------------------#
        #------------OUTPUT SETUP------------#
        #------------------------------------#
        self.wh = dfx.fem.Function(W) # Function for storing the solution
        self.uh_diff = dfx.fem.Function(V) # Function for storing the velocity in the moving reference frame
        self.uh_out  = dfx.fem.Function(dfx.fem.functionspace(mesh, element('DG', cell, 1, shape=(3,)))) # Function for writing the velocity to output
        self.uh_mag_max = 0 # Used to store the maximum of the velocity magnitude
        self.write_time = 0 # Used to write velocity checkpoints
        tau_string = 'variable_tau'

        if self.write_output: # Write velocity to VTX file
            # Create output filenames based on the formulation used
            if self.relative:
                self.u_out_str = f"./output/flow/{tau_string}/velocity_relative+{self.mesh_version}_model{self.model_version}.bp"                
            else:
                self.u_out_str = f"./output/flow/{tau_string}/velocity_pressure+{self.mesh_version}_model{self.model_version}.bp"
            
            # Create velocity output file and write the initial velocity field to the output file
            self.vtx_u = dfx.io.VTXWriter(mesh.comm, self.u_out_str, [self.uh_out], "BP4") 
            self.vtx_u.write(0)

            print(f"Writing output to: {self.u_out_str}")

        if self.write_checkpoint: # Write velocity checkpoints to adios4dolfinx file
            if self.relative:
                self.checkpoint_fname = f'./output/flow/checkpoints/{tau_string}/relative+{self.mesh_version}/model_{self.model_version}/velocity_data_dt={self.dt:.4g}'
            else:
                self.checkpoint_fname = f'./output/flow/checkpoints/{tau_string}/pressure+{self.mesh_version}/model_{self.model_version}/velocity_data_dt={self.dt:.4g}'
            
            # Write mesh and initial velocity field to the checkpoint file
            a4d.write_mesh(mesh=mesh, filename=self.checkpoint_fname, engine="BP4")
            a4d.write_function(u=self.uh_out, filename=self.checkpoint_fname, engine="BP4", time=self.write_time)
            
            print(f"Writing checkpoints to {self.checkpoint_fname}")

    def solve(self):
        """ Solve the discretized Stokes equations. """

        if self.model_version=='A': # The velocity field is a steady state solution => no timeloop
            # Perform linear solve 
            tic = time.perf_counter()
            self.ksp.solve(self.b, self.wh.vector)
            print(f"Solved linear system in {time.perf_counter() - tic:.2f} seconds.")

            # Parallel communication, ghost update
            self.wh.x.scatter_forward()

            # Store solution
            self.u_.x.array[:] = self.wh.sub(0).collapse().x.array.copy()

            if self.relative: # ALE formulation
                # Calculate and output the velocity in the moving reference frame
                self.uh_diff.x.array[:] = self.wh.sub(0).collapse().x.array[:] - self.u_ref.x.array[:]
                self.uh_out.interpolate(self.uh_diff)
            else: # Weak pressure BC formulation
                # Output the velocity in the fixed reference frame
                self.uh_out.interpolate(self.u_)
            
            if self.write_output:
                # Write velocity field to VTX file
                self.vtx_u.write(0)

            if self.write_checkpoint:
                # Write velocity field checkpoint to adios4dolfinx file
                a4d.write_function(u=self.uh_out, filename=self.checkpoint_fname, engine="BP4")

            # Calculate maximum velocity magnitude
            u1 = self.uh_out.sub(0).collapse().x.array
            u2 = self.uh_out.sub(1).collapse().x.array
            u3 = self.uh_out.sub(2).collapse().x.array
            uh_mag = np.sqrt(u1**2 + u2**2 + u3**2)
            print(f"Mean velocity magnitude: {np.mean(uh_mag)}")
            self.uh_mag_max = max(self.uh_mag_max, uh_mag.max())
            self.uh_mag_max = self.mesh.comm.allreduce(self.uh_mag_max, op=MPI.MAX)

            # Calculate maximum velocity magnitudes ventrally and dorsally
            self.mesh.topology.create_connectivity(3, 3)
            ent = dfx.mesh.locate_entities(self.mesh, 3, recal_middle_ventral_cilia_volume)
            V_dofs, _ = dfx.fem.locate_dofs_topological((self.V, self.W), 3, ent)
            bdm_copy = self.u_.copy()
            dg_copy = self.uh_out.copy()
            bdm_copy.x.array[:] = 0
            bdm_copy.x.array[V_dofs] = self.u_.x.array[V_dofs]
            dg_copy.interpolate(bdm_copy)
            dg_reshaped = np.reshape(dg_copy.x.array, (int(len(dg_copy.x.array)/3), 3))
            u1 = dg_reshaped[:, 0] 
            u2 = dg_reshaped[:, 1] 
            u3 = dg_reshaped[:, 2]
            u_mag_ventral = np.sqrt(u1**2+u2**2+u3**2)
            u_mag_ventral_max = u_mag_ventral.max()
            u_mag_ventral_max = self.mesh.comm.allreduce(u_mag_ventral_max, op=MPI.MAX)

            ent = dfx.mesh.locate_entities(self.mesh, 3, recal_anterior_cilia_volume)
            V_dofs, _ = dfx.fem.locate_dofs_topological((self.V, self.W), 3, ent)
            bdm_copy = self.u_.copy()
            dg_copy = self.uh_out.copy()
            bdm_copy.x.array[:] = 0
            bdm_copy.x.array[V_dofs] = self.u_.x.array[V_dofs]
            dg_copy.interpolate(bdm_copy)
            dg_reshaped = np.reshape(dg_copy.x.array, (int(len(dg_copy.x.array)/3), 3))
            u1 = dg_reshaped[:, 0] 
            u2 = dg_reshaped[:, 1] 
            u3 = dg_reshaped[:, 2]
            u_mag_anterior = np.sqrt(u1**2+u2**2+u3**2)
            u_mag_anterior_max = u_mag_anterior.max()
            u_mag_anterior_max = self.mesh.comm.allreduce(u_mag_anterior_max, op=MPI.MAX)
       
        else: # Solve the equations for one cardiac period
            # Define some variables to be used
            tic_loop = time.perf_counter() # Timer for the solution timeloop
            t = 0.0 # Initial time
            u_mag_ventral_max  = 0.0
            u_mag_anterior_max = 0.0
            u_mag_means = []

            print("\n#-------------Solution timeloop-------------#")
            
            for _ in range(int(self.T / self.dt)):
                # Increment time and print current time
                t += dt 
                print(f"\n#------------------------------#\n Time t={t}")

                # Update cardiac motion boundary conditions
                if self.relative: # ALE formulation
                    self.u_ref_expr.t = t
                    self.u_ref.interpolate(self.u_ref_expr)
                    self.u_bc_func.interpolate(self.u_ref_expr)
                else: # Weak pressure BC formulation
                    self.p_bc_expr.t = t
                    self.p_bc.interpolate(self.p_bc_expr)
                    print(f"Max pressure of BC: {self.p_bc.x.array.max()}")

                # Assemble right-hand side vector
                tic = time.perf_counter()
                self.assemble_RHS_vector()
                print(f"Assembly of RHS in {time.perf_counter() - tic:.2f} seconds.")

                # Perform linear solve
                tic = time.perf_counter()
                self.ksp.solve(self.b, self.wh.vector)
                print(f"Solved linear system in {time.perf_counter() - tic:.2f} seconds.")
                
                # Parallel communication, ghost update
                self.wh.x.scatter_forward() 

                if not self.use_direct_solver:
                    # Print iterative solver information
                    niters = self.ksp.getIterationNumber()
                    rnorm  = self.ksp.getResidualNorm()
                    print(f"Number of iterations = {niters}.")
                    print(f"Residual norm = {rnorm:.2e}")

                # Store solution
                self.u_.x.array[:] = self.wh.sub(0).collapse().x.array.copy()
                
                if self.relative: # ALE formulation
                    # Calculate and output the velocity in the moving reference frame
                    self.uh_diff.x.array[:] = self.wh.sub(0).collapse().x.array[:] - self.u_ref.x.array[:]
                    self.uh_out.interpolate(self.uh_diff)
                else: # Weak pressure BC formulation
                    # Output the velocity in the fixed reference frame
                    self.uh_out.interpolate(self.u_)
                
                if self.write_output:
                    # Write velocity field to VTX file
                    self.vtx_u.write(t)

                if self.write_checkpoint:
                    # Write velocity field checkpoint to adios4dolfinx file
                    self.write_time += 1 # Increment checkpoint time index
                    a4d.write_function(u=self.uh_out, filename=self.checkpoint_fname, engine="BP4", time=self.write_time)

                # Calculate maximum velocity magnitude
                u1 = self.uh_out.sub(0).collapse().x.array
                u2 = self.uh_out.sub(1).collapse().x.array
                u3 = self.uh_out.sub(2).collapse().x.array
                uh_mag = np.sqrt(u1**2 + u2**2 + u3**2)
                u_mag_mean = np.mean(uh_mag)
                print(f"Mean velocity magnitude: {u_mag_mean}")
                u_mag_means.append(u_mag_mean)
                self.uh_mag_max = max(self.uh_mag_max, uh_mag.max())
                self.uh_mag_max = self.mesh.comm.allreduce(self.uh_mag_max, op=MPI.MAX)

                div_u_ufl = ufl.div(self.uh_out)
                Q = self.wh.sub(1).collapse().function_space
                quad_points = Q.element.interpolation_points()
                div_u_expr = dfx.fem.Expression(div_u_ufl, quad_points)
                div_u = dfx.fem.Function(Q)
                div_u.interpolate(div_u_expr)
                print(f"Divergence max: {np.max(div_u.x.array[:])}")

                # Calculate max velocities ventrally and dorsally
                self.mesh.topology.create_connectivity(3, 3)
                ent = dfx.mesh.locate_entities(self.mesh, 3, recal_middle_ventral_cilia_volume)
                V_dofs, _ = dfx.fem.locate_dofs_topological((self.V, self.W), 3, ent)
                bdm_copy = self.u_.copy()
                dg_copy = self.uh_out.copy()
                bdm_copy.x.array[:] = 0
                bdm_copy.x.array[V_dofs] = self.u_.x.array[V_dofs]
                dg_copy.interpolate(bdm_copy)
                dg_reshaped = np.reshape(dg_copy.x.array, (int(len(dg_copy.x.array)/3), 3))
                u1 = dg_reshaped[:, 0] 
                u2 = dg_reshaped[:, 1]  
                u3 = dg_reshaped[:, 2]
                u_mag_ventral = np.sqrt(u1**2+u2**2+u3**2)
                u_mag_ventral_max = max(u_mag_ventral_max, u_mag_ventral.max())
                u_mag_ventral_max = self.mesh.comm.allreduce(u_mag_ventral_max, op=MPI.MAX)

                ent = dfx.mesh.locate_entities(self.mesh, 3, recal_anterior_cilia_volume)
                V_dofs, _ = dfx.fem.locate_dofs_topological((self.V, self.W), 3, ent)
                bdm_copy = self.u_.copy()
                dg_copy = self.uh_out.copy()
                bdm_copy.x.array[:] = 0
                bdm_copy.x.array[V_dofs] = self.u_.x.array[V_dofs]
                dg_copy.interpolate(bdm_copy)
                dg_reshaped = np.reshape(dg_copy.x.array, (int(len(dg_copy.x.array)/3), 3))
                u1 = dg_reshaped[:, 0] 
                u2 = dg_reshaped[:, 1] 
                u3 = dg_reshaped[:, 2]
                u_mag_anterior = np.sqrt(u1**2+u2**2+u3**2)
                u_mag_anterior_max = max(u_mag_anterior_max, u_mag_anterior.max())
                u_mag_anterior_max = self.mesh.comm.allreduce(u_mag_anterior_max, op=MPI.MAX)
                
            print(f"Total solution loop time: {time.perf_counter() - tic_loop:.2f} seconds.")
        
        # Solution of linear system done -> print quantities and close output file
        print("Max ventral velocity: ", u_mag_ventral_max)
        print("Max anterior velocity: ", u_mag_anterior_max)
        print(f"Maximum velocity magnitude: {self.uh_mag_max}")

        if self.model_version in ['B', 'C']: # Print time-averaged velocity magnitude
            print(f"Velocity magnitude means: {u_mag_means}")
            print(f"Time-averaged mean velocity: {np.sum(u_mag_means)/20}")
        
        if self.write_output:
            # Close VTX file and write maximum velocity to .txt file
            self.vtx_u.close()
            with open(file=self.u_out_str.removesuffix(".bp")+f"_max_velocity.txt", mode="w+") as file: file.write(str(float(self.uh_mag_max)))

if __name__ == '__main__':

    # Input parameters
    direct = True # Use direct solver if True, else use iterative solver
    model = 'C' # Model version A (only cilia), B (only cardiac) or C (cilia+cardiac)
    f = 2.22 # Cardiac frequency [Hz]
    period = 1 / f # The cardiac period [s]
    T  = 1*period # Simulation end time [s]
    dt = period / 20 # Timestep size [s]
    relative = False # Modeling pulsatile motion: Use ALE formulation if True, else use weak pressure BC formulation
    write_output = False # Write velocity field to VTX file if True
    write_checkpoint = False # Write velocity field checkpoints to adios4dolfinx file if True

    # Set mesh version from input
    mesh_version_input = int(argv[1])
    if mesh_version_input==0:
        mesh_version = 'original'
    elif mesh_version_input==1:
        mesh_version = 'shrunk'
    elif mesh_version_input==2:
        mesh_version = 'hind_shrunk'
    elif mesh_version_input==3:
        mesh_version = 'middle_shrunk'
    elif mesh_version_input==4:
        mesh_version = 'fore_middle_hind_shrunk'
    elif mesh_version_input==5:
        mesh_version = 'mod_dorsal_cilia'
    elif mesh_version_input==6:
        mesh_version = 'mod_ventral_cilia'
    else:
        raise ValueError('Error in mesh version input. Choose an integer in the interval [0, 6].')

    mesh_filename = f'./geometries/{mesh_version}_ventricles.xdmf'
    with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_filename, "r") as xdmf: mesh = xdmf.read_mesh()
    
    # Determine whether to mark pressure boundaries
    if model=='A' or relative: # Pressure BCs not part of the formulation
        in_out = False
    else: # Weak pressure BC part of the formulation
        in_out = True
    
    # Generate boundary facet tags
    ft = recal_mark_boundaries_flow(mesh, inflow_outflow=in_out, modify=mesh_version_input)

    # Print information
    print("\n#-------------Simulation information-------------#")
    print(f"Model version = {model}")
    print(f"Mesh version = {mesh_version}")
    print(f"Linear solver type =", "direct" if direct else "iterative")
    print(f"Timestep size = {dt:.4g} s")
    print(f"Simulation end time = {T:.4g} s")
    if not model=='A': print("Using ALE formulation") if relative else print("Using pressure BCs")

    # Initialize solver object
    solver = FlowSolver(mesh=mesh,
                        ft=ft,
                        direct=direct,
                        model_version=model,
                        mesh_version=mesh_version,
                        T=T,
                        dt=dt,
                        relative=relative,
                        write_output=write_output,
                        write_checkpoint=write_checkpoint)
    solver.setup() # Perform setup
    print("\n#-------------Solver setup complete--------------#\n")
    print("\n#-------------Solving linear system--------------#\n")
    solver.solve() # Perform solve
    print("\n#-------------Simulation complete----------------#\n")