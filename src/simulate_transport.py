from utilities.local_imports import *
from utilities.mesh          import create_ventricle_volumes_meshtags
from diffusion import diffusion_problem

import os
import time
import adios4dolfinx as a4d

from ufl       import avg, jump
from sys       import argv
from pathlib   import Path
from basix.ufl import element
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix, create_vector, set_bc, apply_lifting

""" Solve an advection-diffusion equation, using a discontinuous Galerkin finite 
element method, on a computational mesh of zebrafish brain ventricles.

Author: Halvor Herlyng, 2023-.
"""

print = PETSc.Sys.Print
# PETSc.Log.begin()

# Set compiler options for runtime optimization
cache_dir = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                   "cache_dir"               : cache_dir,
                   "cffi_libraries"          : ["m"]
}

# Mesh tags for flow
ANTERIOR_PRESSURE    = 2
POSTERIOR_PRESSURE   = 3
VOLUME               = 4
MIDDLE_VENTRAL_CILIA = 5
MIDDLE_DORSAL_CILIA  = 6
ANTERIOR_CILIA       = 7
ANTERIOR_CILIA2      = 8
SLIP                 = 9

# Diffusion coefficients
D_coeffs = {
    'D1' : 2.17e-6, # Extracellular vesicles (150 nm radius)
    'D2' : 57.5e-6, # STM-GFP 
    'D3' : 115e-6,  # Dendra2 fluorescent protein
}

class TransportSolver:
    """ Base class for simulating transport in zebrafish brain ventricles. """
    
    comm = MPI.COMM_WORLD # MPI communicator
    ghost_mode = dfx.mesh.GhostMode.shared_facet # Mesh partitioning method for MPI communication
    record_periods = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]) # Periods for which concentration snapshots will be written

    def __init__(self, data_fname: str,
                       model_version: str,
                       mesh_version: str,
                       T: float,
                       dt: float,
                       period: float,
                       molecule: str,
                       element_degree: int,
                       write_data: bool=False,
                       write_output_vtx: bool=False,
                       write_output_xdmf: bool=False,
                       write_checkpoint: bool=False,
                       write_snapshot_checkpoint: bool=False,
                       use_direct_solver: bool=True):
        """ Constructor.

        Parameters
        ----------
        data_fname : str
            Filename of velocity data.
        
        model_version : str
            The problem setup considered for the flow model, 
            either model A, B or C:
                - Model A: only cilia forces
                - Model B: only cardiac forces
                - Model C: cilia and cardiac forces, i.e. A+B
        
        mesh_version : str
            The mesh version used, as specified by the mesh filename prefix.
        
        T : float
            Simulation end time in seconds.
        
        dt : float
            Timestep size in seconds.
        
        period : float
            Cardiac cycle period in seconds.
        
        molecule : str
            Which molecule to simulate transport of. Options are 'D1', 'D2' and 'D3'.

        element_degree : int
            Polynomial degree of finite element basis functions.

        write_data : bool
            Write ROI mean concentrations to .npy data file if True.
        
        write_output_vtx : bool
            Write concentration to VTX file if True.
        
        write_output_xdmf : bool
            Write concentration to XDMF file if True.
        
        write_checkpoint : bool
            Write concentration to adios4dolfinx checkpoint file if True. 
        
        write_snapshot_checkpoint : bool
            Write concentration snapshots at times equal to record_periods if True. 

        use_direct_solver : bool
            Use a direct solver if True, else use an iterative solver.
        """

        self.model_version = model_version
        self.mesh_version = mesh_version
        self.molecule = molecule
        self.D_value = D_coeffs[molecule]
        self.write_data = write_data
        self.write_output_vtx = write_output_vtx
        self.write_output_xdmf = write_output_xdmf
        self.write_checkpoint = write_checkpoint
        self.write_snapshot_checkpoint = write_snapshot_checkpoint
        self.use_direct_solver = use_direct_solver
        self.data_fname = data_fname
        self.element_degree = element_degree
        
        # Read mesh and meshtags from velocity data file
        self.mesh = a4d.read_mesh(comm=self.comm, filename=data_fname, engine='BP4', ghost_mode=self.ghost_mode)
        self.ft   = a4d.read_meshtags(filename=data_fname, mesh=self.mesh, meshtag_name='ft')
        print(f"Total # of cells in mesh: {self.mesh.topology.index_map(self.mesh.topology.dim).size_global}")

        # Temporal parameters
        self.T  = T # Final simulation time
        self.dt = dt # Timestep size
        self.t  = 0 # Initial time
        self.num_timesteps = int(T / dt)
        self.period = period # Cardiac cycle period length
        self.write_time = 0 # For checkpointing

        # DG interior penalty parameter
        self.alpha_val = 25.0 * self.element_degree  

        # Create meshtags for the different ventricle ROIs
        self.ROI_ct, self.ROI_tags = create_ventricle_volumes_meshtags(mesh=self.mesh)      

        # Photoconversion curve logarithm growth factor
        self.a = 65

        # Output directory
        self.output_dir = f'../output/transport/mesh={self.mesh_version}_model={self.model_version}_molecule={self.molecule}_ciliaScenario={cilia_string}_dt={dt:.4g}/'
    
    def setup_variational_problem(self):
        """ Set up bilinear and linear form for the weak form of the advection-diffusion equation discretized 
            with Discontinuous Galerkin elements using a Symmetric Interior Penalty method,
            using an upwind scheme for the velocity.
            
            In time, the equations are discretized with the BDF2 scheme.
        """

        # Initialize
        # Define the cells in ROI 1
        ROI1_cells = self.ROI_ct.find(self.ROI_tags[0])
        self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)
        ROI1_dofs = dfx.fem.locate_dofs_topological(self.W, self.mesh.topology.dim, ROI1_cells)
        self.bc_func = dfx.fem.Function(self.W)
        self.bcs = [dfx.fem.dirichletbc(self.bc_func, ROI1_dofs)]

        dS = ufl.Measure('dS', domain=self.mesh) # Interior facet integral measure
        alpha  = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.alpha_val))  # DG interior penalty parameter
        un  = (dot(self.u, self.n) + abs(dot(self.u, self.n))) / 2.0 # Upwind velocity
        pres_tags = (ANTERIOR_PRESSURE, POSTERIOR_PRESSURE) # pressure BC tags

        # Time-discretization: BDF2 scheme
        a0 = 3/2*self.c*self.w / self.deltaT * self.dx # Time derivative

        a1 = dot(grad(self.w), self.D * grad(self.c) - self.u * self.c) * self.dx # Flux term integrated by parts

        # Diffusive terms with interior penalization
        a2  = self.D('+') * alpha('+') / avg(self.hf) * dot(jump(self.w, self.n), jump(self.c, self.n)) * dS
        a2 -= self.D('+') * dot(avg(grad(self.w)), jump(self.c, self.n)) * dS
        a2 -= self.D('+') * dot(jump(self.w, self.n), avg(grad(self.c))) * dS
        
        # Advective term
        a3  = dot(jump(self.w), un('+') * self.c('+') - un('-') * self.c('-')) * dS

        a = a0+a1+a2+a3
        
        # Linear form time derivative terms
        L0 = 2*self.c_ * self.w / self.deltaT * self.dx 
        L1 = -1/2*self.c__ * self.w / self.deltaT * self.dx
        L = L0+L1
        
        if (self.model_version=='B' or self.model_version=='C'):    
            # Impose consistent flux BC: influx(t_n) = outflux(t_{n-1})
            # No diffusive flux on outflux BC, advective outflux kept in variational form
            outflux  = self.c*dot(self.u, self.n) # Only advective flux on outflow boundary, diffusive flux is zero
            u_normal = dot(self.u, self.n) # The normal velocity

            # Create conditional expressions
            cond  = ufl.lt(u_normal, 0.0) # Condition: True if the normal velocity is less than zero, u.n < 0
            minus = ufl.conditional(cond, 1.0, 0.0) # Conditional that returns 1.0 if u.n <  0, else 0.0. Used to "activate" terms on the influx  boundary
            plus  = ufl.conditional(cond, 0.0, 1.0) # Conditional that returns 1.0 if u.n >= 0, else 0.0. Used to "activate" terms on the outflux boundary
            
            # Add outflux term to the weak form
            a += plus*outflux * self.w * self.ds(pres_tags)
                
            # Calculate influx as the outflux at the previous timestep
            plus_ = ufl.conditional(ufl.lt(dot(self.u_, self.n), 0.0), 0.0, 1.0) # Conditional that returns 0.0 if u.n < 0 at the previous timestep, else returns 1.0
            self.flux_ = plus_*dot((self.c_*self.u_ - self.D*grad(self.c_)), self.n)

            L += minus*self.flux_ * self.w * self.ds(pres_tags)

        # Compile RHS linear form
        self.a_cpp = dfx.fem.form(a, jit_options=jit_parameters)
        self.L_cpp = dfx.fem.form(L, jit_options=jit_parameters)

    def photoconversion_curve(self):
        return np.log(1+self.t/self.a)/np.log(1+self.T/self.a) # Parameter a determines growth speed

    def setup_preconditioner(self):
        """ Variational problem for the preconditioner matrix:
            velocity field u=0, such that the system is simply
            the diffusion equation:

            dc/dt = D*nabla^2(c).
        """

        # boundary tags
        dS = ufl.Measure('dS', domain=self.mesh)
        alpha = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.alpha_val))  # DG interior penalty parameter

        # Bilinear form
        a0 = 3/2*self.c * self.w / self.deltaT * self.dx
        a1 = dot(grad(self.w), self.D * grad(self.c)) * self.dx # Flux term integrated by parts

        # Diffusive terms with interior penalization
        a2  = self.D('+') * alpha('+') / avg(self.hf) * dot(jump(self.w, self.n), jump(self.c, self.n)) * dS
        a2 -= self.D('+') * dot(avg(grad(self.w)), jump(self.c, self.n)) * dS
        a2 -= self.D('+') * dot(jump(self.w, self.n), avg(grad(self.c))) * dS

        a = a0+a1+a2

        # Convert to C++ form and assemble the preconditioner matrix
        a_P_cpp = dfx.fem.form(a, jit_options=jit_parameters)
        self.P = assemble_matrix(a_P_cpp, bcs=self.bcs)
        self.P.assemble()

    def setup(self):
        """ Perform initialization and setup of the variational form. """

        # Diffusion coefficient
        self.D = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.D_value))

        # Facet normal vector and integral measures
        self.n  = ufl.FacetNormal(self.mesh)
        self.dx = ufl.Measure('dx', domain=self.mesh, subdomain_data=self.ROI_ct)
        self.ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.ft)

        # Velocity field finite elements
        DG1_vec = element('DG', self.mesh.basix_cell(), 1, shape=(3,)) # Piecewise linear vector elements for the velocity
        V = dfx.fem.functionspace(self.mesh, DG1_vec)
        self.u  = dfx.fem.Function(V) # Velocity
        self.u_ = dfx.fem.Function(V) # Velocity at previous timestep
        
        # Read velocity data
        a4d.read_function(u=self.u, filename=self.data_fname, engine="BP4")
        
        # Piecewise (discontinuous) linear lagrange elements for concentration
        DG = element('DG', self.mesh.basix_cell(), self.element_degree) 

        self.W = W = dfx.fem.functionspace(self.mesh, DG) # Concentration function space
        print("Number of dofs: ", W.dofmap.index_map.size_global) # Print the number of dofs
        
        # Trial and test functions
        self.c, self.w = ufl.TrialFunction(W), ufl.TestFunction(W)

        # Functions for storing the concentrations
        self.c_h  = dfx.fem.Function(W) # Concentration at current  timestep

        # Solve the diffusion equation for a smooth initial condition for the concentration
        # at two previous timesteps
        print('Solving diffusion equation for initial conditions ...')
        self.c__, self.c_ = diffusion_problem(self.mesh, k=self.element_degree, D_value=self.D_value)
        for conc in [self.c__, self.c_]:
            # Scale the concentration values so that they are on
            # the interval [0, 1]. This is so that they later can 
            # be multiplied by the bc_func values at the first
            # timestep, resulting in concentration values on
            # the interval [0, max(bc_func)].
            max_conc = self.comm.allreduce(conc.x.array.max(), op=MPI.MAX)
            conc.x.array[:] /= max_conc
        print('Initial condition set.')

        #------VARIATIONAL FORM------#
        self.deltaT = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.dt)) # Compiled timestep
        self.hf = ufl.CellDiameter(self.mesh) # Cell diameter

        # Set up the variational problem
        self.setup_variational_problem()

        # Create RHS vector
        self.A = create_matrix(self.a_cpp)
        self.b = create_vector(self.L_cpp)

        # Setup solver
        self.setup_preconditioner()
        self.setup_solver()

        # Prepare stuff to be calculated
        self.total_c = dfx.fem.form(self.c_h * self.dx, jit_options=jit_parameters)
        
        if self.write_output_vtx:
            # Intialize VTX output file for the concentration
            self.output_vtx_str = self.output_dir + 'concentration.bp'            
            self.vtx_c = dfx.io.VTXWriter(comm=self.comm, filename=self.output_vtx_str, output=[self.c_h], engine="BP4")
            print(f"Writing VTX output to: {self.output_vtx_str}\n")
        
        if self.write_output_xdmf:
            # Intialize XDMF output file for the concentration
            self.output_xdmf_str = self.output_dir + 'concentration.xdmf' 
            self.c_dg0 = dfx.fem.Function(dfx.fem.functionspace(self.mesh, element("DG", self.mesh.basix_cell(), 0)))           
            self.xdmf_c = dfx.io.XDMFFile(self.comm, self.output_xdmf_str, "w")
            self.xdmf_c.write_mesh(self.mesh)
            print(f"Writing XDMF output to: {self.output_xdmf_str}\n")
        
        if self.write_checkpoint:
            # Intialize output file for checkpoints of the concentration
            self.checkpoint_filename = self.output_dir + 'checkpoints/concentration/'
            a4d.write_mesh(mesh=self.mesh, filename=self.checkpoint_filename)
            print(f"Writing concentration checkpoints to: {self.checkpoint_filename}\n")

        if self.write_data:
            self.data_output_dir = self.output_dir + "data/"
            if self.comm.rank==0 and not os.path.isdir(self.data_output_dir): os.makedirs(self.data_output_dir)
            
            # Compile integration forms
            self.c_hat_forms = []
            for ROI_tag in self.ROI_tags:
                if ROI_tag==4:
                    # For ROI 4, we need to include the integrals of regions 1, 2 and 3
                    self.c_hat_forms.append(
                                        dfx.fem.form(
                                                self.c_h*self.dx((ROI_tag, ROI_tag-1, ROI_tag-2, ROI_tag-3)),
                                                jit_options=jit_parameters
                                                )
                                            )
                else:
                    self.c_hat_forms.append(
                                        dfx.fem.form(
                                                self.c_h*self.dx(ROI_tag),
                                                jit_options=jit_parameters
                                                )
                                            )

            # Pre-allocate concentration arrays
            self.c_hat_arrays = np.zeros((self.num_timesteps, len(self.ROI_tags)), dtype=np.float64)

            print(f"Writing concentration data arrays to: {self.data_output_dir}\n")

        if self.write_snapshot_checkpoint:
            self.snapshot_times = self.record_periods*self.period
            self.snapshot_filename = self.output_dir + 'checkpoints/concentration_snapshots/'
            a4d.write_mesh(mesh=self.mesh, filename=self.snapshot_filename)
            print(f"Writing concentration snapshots to: {self.snapshot_filename}")
            print(f"Writing the snapshots at times: {self.snapshot_times}\n")

        self.t_hat = dfx.fem.Function(W) # Finite element function for storing the times to threshold
        self.t_hat.x.array[:] = -1 # Initialize all t_hats as -1
        self.c_bar_threshold = 0.25 # Threshold concentration value        

    def setup_solver(self):
        """ Create and configure linear solver. """

        if self.use_direct_solver:
            # Configure direct solver
            self.solver = PETSc.KSP().create(self.comm)
            self.solver.setOperators(self.A)
            self.solver.setType("preonly")
            self.solver.getPC().setType("lu")
            self.solver.getPC().setFactorSolverType("mumps")
            self.solver.getPC().getFactorMatrix().setMumpsIcntl(icntl=58, ival=1) # activate symbolic factorization
        else:
            # Configure iterative solver
            self.solver = PETSc.KSP().create(self.comm)
            self.solver.setOperators(self.A, self.P)
            self.solver.setType("fgmres")
            self.solver.getPC().setType("bjacobi")
            self.solver.setTolerances(rtol=1e-10)
            opts = PETSc.Options()
            opts.setValue('sub_pc_type', 'ilu')
            self.solver.setFromOptions()
        
    def assemble_system_matrix(self):
        """ Assemble the system matrix of the variational problem. """

        self.A.zeroEntries()
        assemble_matrix(self.A, self.a_cpp, bcs=self.bcs)
        self.A.assemble()

    def assemble_RHS(self):
        """ Assemble the right-hand side vector of the variational problem. """
        
        with self.b.localForm() as b_loc: b_loc.set(0) # Avoid accumulation of values
        assemble_vector(self.b, self.L_cpp) # Assemble the vector
        if len(self.bcs)>0:
            # Enforce, Dirichlet conditions
            apply_lifting(self.b, [self.a_cpp], bcs=[self.bcs])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) # Perform ghost update: retrieve
            set_bc(self.b, bcs=self.bcs) # Set the boundary conditions
            self.b.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD) # Perform ghost update: send
        
    def run(self):
        """ Run transport simulations. """

        if self.model_version=='A':

            # Read velocity data from file
            # This is done before the solution loop
            # because the velocity field is a
            # steady-state solution for this model version
            a4d.read_function(u=self.u, filename=self.data_fname, engine="BP4", time=1)
            
            for i in range(self.num_timesteps):

                self.t += self.dt

                if i==0:
                    # Scale initial conditions by the photoconversion curve values at
                    # the respective time points
                    self.bc_func.x.array[:] = self.photoconversion_curve()  
                    self.c__.x.array[:] *= self.comm.allreduce(self.bc_func.x.array.max(), op=MPI.MAX)
                    self.t += self.dt

                    self.bc_func.x.array[:] = self.photoconversion_curve()
                    self.c_.x.array[:] *= self.comm.allreduce(self.bc_func.x.array.max(), op=MPI.MAX)
                    self.t += self.dt

                # Update photoconversion site (ROI 1) concentration
                self.bc_func.x.array[:] = self.photoconversion_curve()

                # Assemble the system matrix and the right-hand side vector and solve
                self.assemble_system_matrix()
                self.assemble_RHS()
                self.solver.solve(self.b, self.c_h.x.petsc_vec)
                self.c_h.x.scatter_forward()

                # Update previous timestep solution
                self.c__.x.array[:] = self.c_.x.array.copy()
                self.c_.x.array[:] = self.c_h.x.array.copy()

                # Print stuff
                print(f"Timestep t = {self.t}")
                print("Maximum concentration: ", self.comm.allreduce(self.c_h.x.array.max(), op=MPI.MAX))
                print("Minimum concentration: ", self.comm.allreduce(self.c_h.x.array.min(), op=MPI.MIN))

                total_c = dfx.fem.assemble_scalar(self.total_c)
                total_c = self.comm.allreduce(total_c, op=MPI.SUM)
                print(f"Total concentration: {total_c:.2e}")

                # Write output to file
                if self.write_output_vtx : self.vtx_c.write(self.t)
                if self.write_output_xdmf:
                    self.c_dg0.interpolate(self.c_h)
                    self.xdmf_c.write_function(u=self.c_dg0, t=self.t)
                
                # Write checkpoints
                if self.write_checkpoint: 
                    self.write_time += 1
                    a4d.write_function(u=self.c_h, filename=self.checkpoint_filename, time=self.write_time)
                
                if self.write_data:
                    # Calculate mean concentrations and store them in arrays
                    for j in range(len(self.ROI_tags)):
                        self.c_hat_arrays[i, j] = self.comm.allreduce(dfx.fem.assemble_scalar(self.c_hat_forms[j]), op=MPI.SUM)
                        if j==3:
                            # ROI 4 needs to add integrals from regions 1, 2 and 3
                            self.c_hat_arrays[i, j] += self.c_hat_arrays[i, j-1] + self.c_hat_arrays[i, j-2] + self.c_hat_arrays[i, j-3]
                    
                    idx_set = np.where(self.c_h.x.array > self.c_bar_threshold)[0] # Index to dofs of f_k that are above the threshold
                    below = np.where(self.t_hat.x.array == -1)[0] # Index to dofs of t_hat that have not yet been assigned a time
                    above_idx = np.intersect1d(idx_set, below) # The intersection of the above two, which are the dofs where t_hat should be assigned as current timestep
                    self.t_hat.x.array[above_idx] = self.t

        else: # Model B or C
            read_time = 0
            period_counter = 0

            tic = time.perf_counter()
        
            for i in range(self.num_timesteps):
                
                # Increment time
                self.t += self.dt

                # Read velocity data
                if read_time==period_partition:
                    # Previous timestep was at the end of the period of the 
                    # cardiac cycle -> reset the read_time variable to the first
                    # timestep of the cardiac cycle
                    read_time = 1
                    period_counter += 1
                else:
                    read_time += 1
                self.u_ = self.u
                a4d.read_function(u=self.u, filename=self.data_fname, engine="BP4", time=read_time) # velocity at this timestep
                
                if i==0:
                    # Scale initial conditions by the photoconversion curve values at
                    # the respective time points
                    self.bc_func.x.array[:] = self.photoconversion_curve()  
                    self.c__.x.array[:] *= self.comm.allreduce(self.bc_func.x.array.max(), op=MPI.MAX)
                    self.t += self.dt

                    self.bc_func.x.array[:] = self.photoconversion_curve()
                    self.c_.x.array[:] *= self.comm.allreduce(self.bc_func.x.array.max(), op=MPI.MAX)
                    self.t += self.dt

                # Update photoconversion site (ROI 1) concentration
                self.bc_func.x.array[:] = self.photoconversion_curve()

                # Assemble the system matrix and the right-hand side vector
                self.assemble_system_matrix()
                self.assemble_RHS()

                # Solve
                self.solver.solve(self.b, self.c_h.x.petsc_vec)
                self.c_h.x.scatter_forward()
                # PETSc.Log.view()

                # Update previous timestep solution
                self.c__.x.array[:] = self.c_.x.array.copy()
                self.c_.x.array[:] = self.c_h.x.array.copy()
            
                # Print stuff
                print(f"Timestep t = {self.t}")
                max_c = self.comm.allreduce(self.c_h.x.array.max(), op=MPI.MAX)
                min_c = self.comm.allreduce(self.c_h.x.array.min(), op=MPI.MIN)

                # Print stuff
                print("Maximum concentration: ", max_c)
                print("Minimum concentration: ", min_c)
                
                total_c = dfx.fem.assemble_scalar(self.total_c)
                total_c = self.comm.allreduce(total_c, op=MPI.SUM)
                print(f"Total concentration: {total_c:.2e}")

                if self.write_output_vtx: self.vtx_c.write(self.t) # Write to file

                if self.write_output_xdmf:
                    self.c_dg0.interpolate(self.c_h)
                    self.xdmf_c.write_function(u=self.c_dg0, t=self.t)

                if self.write_checkpoint:
                    self.write_time += 1
                    a4d.write_function(u=self.c_h, filename=self.checkpoint_filename, time=self.write_time)
                
                if self.write_snapshot_checkpoint and True in np.isclose(self.t, self.snapshot_times):
                    record_period_index = np.where(np.isclose(self.t, self.snapshot_times)==True)[0][0]
                    a4d.write_function(u=self.c_h, filename=self.snapshot_filename, time=self.record_periods[record_period_index])

                if self.write_data:
                    # Calculate mean concentrations and store them in arrays
                    for j in range(len(self.ROI_tags)):
                        self.c_hat_arrays[i, j] = self.comm.allreduce(dfx.fem.assemble_scalar(self.c_hat_forms[j]), op=MPI.SUM)
                        if j==3:
                            # ROI 4 needs to add integrals from regions 1, 2 and 3
                            self.c_hat_arrays[i, j] += self.c_hat_arrays[i, j-1] + self.c_hat_arrays[i, j-2] + self.c_hat_arrays[i, j-3]
                    
                    idx_set = np.where(self.c_h.x.array > self.c_bar_threshold)[0] # Index to dofs of f_k that are above the threshold
                    below = np.where(self.t_hat.x.array == -1)[0] # Index to dofs of t_hat that have not yet been assigned a time
                    above_idx = np.intersect1d(idx_set, below) # The intersection of the above two, which are the dofs where t_hat should be assigned as current timestep
                    self.t_hat.x.array[above_idx] = self.t

        print("\nFor-loop finished, closing ...\n")
        print(f'Solve loop time: {time.perf_counter() - tic:.4g} sec')

        if self.write_data:
            with open(file=self.data_output_dir+f"final_total_c.txt", mode="w+") as file: file.write(str(float(total_c)))

            # Save concentration arrays to file
            with open(self.data_output_dir + 'c_hats.npy', 'w+b') as c_means_file:
                np.save(c_means_file, self.c_hat_arrays)
            
            # Write output of t_hat (times to threshold)
            with dfx.io.VTXWriter(self.comm, self.output_dir+f"t_hat.bp", [self.t_hat], "BP4") as vtx: vtx.write(0) # VTX file (for visualization)
            a4d.write_function_on_input_mesh(filename=self.output_dir+f"t_hat", u=self.t_hat) # Checkpoint file (can be reloaded later)

        if self.write_output_xdmf: self.xdmf_c.close()
        if self.write_output_vtx: self.vtx_c.close()

if __name__ == '__main__':

    # Model versions:
    # A = cilia-driven/no-cardiac
    # B = cardiac-induced/no-cilia
    # C = cilia+cardiac (baseline)

    model_version = 'C' 
    write_data = True # Write ROI mean concentrations as numpy data
    write_output_vtx = False # Write VTX output file
    write_output_xdmf = False # Write XDMF output file
    write_checkpoint  = False # Write DOLFINx fem function concentration checkpoints
    write_snapshot_checkpoint = True # Write checkpoints at specific periods defined in class
    use_direct_solver = False # Use a direct or iterative solver
    k = 2 # DG element polynomial degree
    
    # Molecule options:
    # D1 = Extracellular vesicles (radius 150 nm)
    # D2 = Starmaker + Green Fluorescent Protein
    # D3 = Dendra2 Fluorescent Protein
    molecule_input = int(argv[1])
    if molecule_input==1:
        molecule = 'D1'
    elif molecule_input==2:
        molecule = 'D2'
    elif molecule_input==3:
        molecule = 'D3'
    else:
        raise ValueError('Error in molecule number input. Choose 1, 2, or 3.')

    # Set mesh version from input
    mesh_version_input = int(argv[2])
    if mesh_version_input==0:
        mesh_version = 'original'
    elif mesh_version_input==1:
        mesh_version = 'fore_shrunk'
    elif mesh_version_input==2:
        mesh_version = 'hind_shrunk'
    elif mesh_version_input==3:
        mesh_version = 'middle_shrunk'
    elif mesh_version_input==4:
        mesh_version = 'fore_middle_hind_shrunk'
    else:
        raise ValueError('Error in mesh version input. Choose an integer in the interval [0, 4].')

    # Parse cilia modification scenario
    cilia_scenario_input = int(argv[3])
    if cilia_scenario_input==0:
        cilia_string = 'all_cilia'
    elif cilia_scenario_input==1:
        cilia_string = 'rm_anterior'
    elif cilia_scenario_input==2:
        cilia_string = 'rm_dorsal'
    elif cilia_scenario_input==3:
        cilia_string = 'rm_ventral'
    else:
        raise ValueError('Error in cilia modification scenario input. Choose an integer in the interval [0, 3].')
    
    # Temporal parameters
    f = 2.22         # Cardiac frequency [Hz]
    period = 1 / f   # Cardiac period [s]
    T  = 1900*period # Simulation end time
    period_partition = 20
    dt = period / period_partition # Timestep size [s]
    print("Transport simulation information:")
    print("Model version: " + model_version)
    print("Mesh version: " + mesh_version)
    print("Molecule: " + molecule)
    print("Cilia scenario: " + cilia_string)
    print(f"Timestep: dt = {dt:.4g}")

    # Set velocity data filename
    data_fname = f'../output/flow/checkpoints/velocity_mesh={mesh_version}_model={model_version}_ciliaScenario={cilia_string}_dt={dt:.4g}'

    # Create and set up solver object
    transport_sim = TransportSolver(data_fname=data_fname, model_version=model_version,
                                    mesh_version=mesh_version,
                                    T=T, dt=dt, period=period,
                                    molecule=molecule,
                                    element_degree=k, 
                                    write_data=write_data,
                                    write_output_vtx=write_output_vtx,
                                    write_output_xdmf=write_output_xdmf,
                                    write_checkpoint=write_checkpoint,
                                    write_snapshot_checkpoint=write_snapshot_checkpoint,
                                    use_direct_solver=use_direct_solver)
    transport_sim.setup()

    # Run the transport simulation
    tic = time.perf_counter()
    transport_sim.run()
    toc = time.perf_counter()

    # Print information
    print("\n#-------------------------#")
    print("Transport simulation completed.")
    print("Model version: " + model_version)
    print("Mesh version: " + mesh_version)
    print("Molecule: " + molecule)
    print("Final time: ", T)
    print("Timestep size: ", dt)
    if write_output_vtx : print(f"Concentration VTX file written to {transport_sim.output_dir}")
    if write_output_xdmf: print(f"Concentration XDMF file written to {transport_sim.output_dir}")
    if write_checkpoint : print(f"Concentration checkpoints written to {transport_sim.output_dir}")
    if write_snapshot_checkpoint: print(f"Concentration snapshots written to {transport_sim.output_dir}")
    print(f"Solver time: {MPI.COMM_WORLD.allreduce(toc-tic, op=MPI.MAX):.2e}")
    print("#-------------------------#\n")