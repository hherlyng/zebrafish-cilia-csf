from utilities.local_imports import *
from utilities.mesh          import mark_boundaries_flow_and_transport, create_ventricle_volumes_meshtags

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
    'D1'  : 1.63e-6,
    'D2'  : 57.5e-6,
    'D3'  : 115e-6,
    'NPY' : 150e-6,
    'DFP' : 115e-6,
    'STM_GFP' : 57.5-6,
    'EXO' : 1.63e-6
}

class TransportSolver:
    """ Base class for simulating transport in zebrafish brain ventricles. """
    
    comm = MPI.COMM_WORLD # MPI communicator
    ghost_mode = dfx.mesh.GhostMode.shared_facet # Mesh partitioning method for MPI communication
    record_periods = np.array([10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750])

    def __init__(self, data_fname: str,
                       model_version: str,
                       mesh_version: int,
                       T: float,
                       dt: float,
                       period: float,
                       molecule: str,
                       element_degree: int,
                       relative: bool,
                       tau_version: str,
                       flux_BC: bool=True,
                       write_data: bool=False,
                       write_peclet_vtx: bool=False,
                       write_peclet_checkpoint: bool=False,
                       write_output_vtx: bool=False,
                       write_output_xdmf: bool=False,
                       write_checkpoint: bool=False,
                       write_snapshot_checkpoint: bool=False,
                       use_direct_solver: bool=True):
        """ Constructor for the transport model solver. """

        self.model_version = model_version
        self.mesh_version = mesh_version
        self.molecule = molecule
        self.D_value = D_coeffs[molecule]
        self.write_data = write_data
        self.write_peclet_vtx = write_peclet_vtx
        self.write_peclet_checkpoint = write_peclet_checkpoint
        self.write_output_vtx = write_output_vtx
        self.write_output_xdmf = write_output_xdmf
        self.write_checkpoint = write_checkpoint
        self.write_snapshot_checkpoint = write_snapshot_checkpoint
        self.use_direct_solver = use_direct_solver
        self.data_fname = data_fname
        self.element_degree = element_degree
        self.relative = relative
        self.flux_BC = flux_BC
        self.tau_version = tau_version
        
        self.mesh = a4d.read_mesh(comm=self.comm, filename=data_fname, engine='BP4', ghost_mode=self.ghost_mode)
        print(f"Total # of cells in mesh: {self.mesh.topology.index_map(3).size_global}")

        # Set mesh and facet tags
        io_boundaries = False if model_version=='A' else True
        self.ft = mark_boundaries_flow_and_transport(mesh=self.mesh, inflow_outflow=io_boundaries)

        # Temporal parameters
        self.T  = T # Final simulation time
        self.dt = dt # Timestep size
        self.t  = 0 # Initial time
        self.num_timesteps = int(T / dt)
        self.period = period # Cardiac cycle period length
        self.write_time = 0 # For checkpointing

        # Penalty parameters
        self.alpha_val = 25.0  # DG interior penalty parameter
        self.beta_val  = 500.0 # Nitsche penalty parameter, for weak imposition of Dirichlet BC

        # Create meshtags for the different ventricle ROIs
        self.ROI_ct, self.ROI_tags = create_ventricle_volumes_meshtags(mesh=self.mesh)      

        # Concentration Dirihclet condition value
        self.c_tilde = 1.0 # Max concentration in ROI 1 as time t->large
        self.a = 65 # Logarithm growth factor

        # Set output directory
        if self.flux_BC:
            #injection_site_{self.injection_site_str}/"
            if self.relative:
                self.output_dir = f"./output/transport/results/{self.tau_version}/{self.mesh_version}/log_model_{self.model_version}_{self.molecule}_DG{self.element_degree}_ALE/"
            else:
                self.output_dir = f"./output/transport/results/{self.tau_version}/{self.mesh_version}/log_model_{self.model_version}_{self.molecule}_DG{self.element_degree}_pressureBC/"
        else:
            if self.relative:
                self.output_dir = f"./output/transport/results/{self.tau_version}/{self.mesh_version}/log_model_{self.model_version}_{self.molecule}_DG{self.element_degree}_ALE_flux_BC_off/"
            else:
                self.output_dir = f"./output/transport/results/{self.tau_version}/{self.mesh_version}/log_model_{self.model_version}_{self.molecule}_DG{self.element_degree}_pressureBC/"

    def setup_variational_problem(self, c: ufl.TrialFunction, w: ufl.TestFunction):
        """ Set up bilinear and linear form for the weak form of the advection-diffusion equation discretized 
            with Discontinuous Galerkin elements.
        """

        # Initialize
        dS = ufl.Measure('dS', domain=self.mesh) # Interior facet integral measure
        alpha  = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.alpha_val))  # DG interior penalty parameter
        self.un = (dot(self.u, self.n) + abs(dot(self.u, self.n))) / 2.0 # Upwind velocity
        pres_tags = (ANTERIOR_PRESSURE, POSTERIOR_PRESSURE) # pressure BC tags

        # Bilinear form
        a0 = c * w / self.deltaT * self.dx # Time derivative
        a1 = dot(grad(w), self.D * grad(c) - self.u * c) * self.dx # Flux term integrated by parts

        # Diffusive terms with interior penalization
        a2  = self.D('+') * alpha('+') / avg(self.hf) * dot(jump(w, self.n), jump(c, self.n)) * dS
        a2 -= self.D('+') * dot(avg(grad(w)), jump(c, self.n)) * dS
        a2 -= self.D('+') * dot(jump(w, self.n), avg(grad(c))) * dS

        # Advective term
        a3  = dot(jump(w), self.un('+') * c('+') - self.un('-') * c('-')) * dS

        a = a0+a1+a2+a3

        # Linear form
        L  = self.c_ * w / self.deltaT * self.dx # Time derivative
        
        if (self.model_version=='B' or self.model_version=='C') and self.flux_BC:
            # Impose consistent flux BC: influx(t_n) = outflux(t_{n-1})
            # No diffusive flux on outflux BC, advective outflux kept in variational form
            outflux  = c*dot(self.u, self.n) # Only advective flux on outflow boundary, diffusive flux is zero
            u_normal = dot(self.u, self.n) # The normal velocity

            # Create conditional expressions
            cond  = ufl.lt(u_normal, 0.0) # Condition: True if the normal velocity is less than zero, u.n < 0
            minus = ufl.conditional(cond, 1.0, 0.0) # Conditional that returns 1.0 if u.n <  0, else 0.0. Used to "activate" terms on the influx  boundary
            plus  = ufl.conditional(cond, 0.0, 1.0) # Conditional that returns 1.0 if u.n >= 0, else 0.0. Used to "activate" terms on the outflux boundary
            
            # Add outflux term to the bilinear form
            a += plus*outflux * w * self.ds(pres_tags)
            
            # Calculate influx as the outflux at the previous timestep
            plus_ = ufl.conditional(ufl.lt(dot(self.u_, self.n), 0.0), 0.0, 1.0) # Conditional that returns 0.0 if u.n < 0 at the previous timestep, else returns 1.0
            self.flux_ = plus_*dot((self.c_*self.u_ - self.D*grad(self.c_)), self.n)

            L += minus*self.flux_ * w * self.ds(pres_tags)
            
        # Set strong BC in ROI 1
        ROI1_cells = self.ROI_ct.find(self.ROI_tags[0])
        self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)
        ROI1_dofs = dfx.fem.locate_dofs_topological(self.W, self.mesh.topology.dim, ROI1_cells)
        self.bc_func = dfx.fem.Function(self.W)
        self.bc_func.x.array[:] = self.experimental_concentration_curve()
        bc = dfx.fem.dirichletbc(self.bc_func, ROI1_dofs)
        self.bcs = [bc]

        # Compile C++ forms
        self.a_cpp = dfx.fem.form(a, jit_options=jit_parameters)
        self.L_cpp = dfx.fem.form(L, jit_options=jit_parameters)

    def experimental_concentration_curve(self): return self.c_tilde*np.log(1+self.t/self.a)/np.log(1+self.T/self.a) # a factor to determine growth speed

    def setup_preconditioner(self, c: ufl.TrialFunction, w: ufl.TestFunction):
        """ Variational problem for the preconditioner matrix:
            velocity field u=0, such that the system is simply

            dc/dt = D*Laplace(c).
        """

        # boundary tags
        dS = ufl.Measure('dS', domain=self.mesh)
        alpha = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.alpha_val))  # DG interior penalty parameter

        # Bilinear form
        a0 = c * w / self.deltaT * self.dx
        a1 = dot(grad(w), self.D * grad(c)) * self.dx # Flux term integrated by parts

        # Diffusive terms with interior penalization
        a2  = self.D('+') * alpha('+') / avg(self.hf) * dot(jump(w, self.n), jump(c, self.n)) * dS
        a2 -= self.D('+') * dot(avg(grad(w)), jump(c, self.n)) * dS
        a2 -= self.D('+') * dot(jump(w, self.n), avg(grad(c))) * dS

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
        self.dx = ufl.Measure('dx', domain=self.mesh)
        self.ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.ft)

        # Velocity field finite elements
        DG1_vec = element('DG', self.mesh.basix_cell(), 1, shape=(3,)) # Piecewise linear vector elements
        V = dfx.fem.functionspace(self.mesh, DG1_vec)
        self.u  = dfx.fem.Function(V) # velocity
        self.u_ = dfx.fem.Function(V) # velocity at previous timestep
        
        # Read velocity data
        a4d.read_function(u=self.u, filename=self.data_fname, engine="BP4")
        
        # Finite elements
        DG = element("DG", self.mesh.basix_cell(), self.element_degree) # Piecewise constant elements for computations

        self.W = W = dfx.fem.functionspace(self.mesh, DG) # Concentration function space
        print("Number of dofs: ", W.dofmap.index_map.size_global, flush=True) # Print the number of dofs

        # Trial and test functions
        c, w = ufl.TrialFunction(W), ufl.TestFunction(W)
        self.c = c; self.w = w
        # Functions for storing solution
        self.c_h  = dfx.fem.Function(W) # Concentration at current  timestep
        self.c_   = dfx.fem.Function(W) # Concentration at previous timestep

        #------VARIATIONAL FORM------#
        self.deltaT = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.dt)) # Form timestep
        self.hf = ufl.CellDiameter(self.mesh) # Cell diameter

        # Set up the variational problem
        self.setup_variational_problem(c, w)

        # Create linear system
        self.A = create_matrix(self.a_cpp) # Create system matrix
        self.b = create_vector(self.L_cpp) # Create RHS vector

        # Setup solver
        self.setup_preconditioner(c, w)
        self.setup_solver()

        # Prepare stuff to be calculated
        self.total_c = dfx.fem.form(self.c_h * self.dx)
        
        if self.write_output_vtx:
            # Intialize VTX output file for the concentration
            self.output_vtx_str = self.output_dir + 'concentration.bp'            
            self.vtx_c = dfx.io.VTXWriter(comm=self.comm, filename=self.output_vtx_str, output=[self.c_h], engine="BP4")
            print(f"Writing VTX output to: {self.output_vtx_str}\n")

        if self.write_peclet_vtx or self.write_peclet_checkpoint:
            # Intialize output file for the Peclet number
            self.peclet = dfx.fem.Function(self.W)
            if self.write_peclet_vtx:
                self.vtx_peclet = dfx.io.VTXWriter(comm=self.comm, filename=self.output_dir+"peclet_number.bp", output=[self.peclet], engine="BP4")   
            if self.write_peclet_checkpoint:
                self.peclet_cpoint_fname = self.output_dir + "checkpoints/peclet_number/"
                for record_period in self.record_periods:
                    a4d.write_mesh(mesh=self.mesh, filename=f'{self.peclet_cpoint_fname}_period{record_period}')
        
        if self.write_output_xdmf:
            # Intialize XDMF output file for the concentration
            self.output_xdmf_str = self.output_dir + 'concentration.xdmf' 
            self.c_dg0 = dfx.fem.Function(dfx.fem.functionspace(self.mesh, element("DG", self.mesh.basix_cell(), 0)))           
            self.xdmf_c = dfx.io.XDMFFile(self.comm, self.output_xdmf_str, "w")
            self.xdmf_c.write_mesh(self.mesh)
            print(f"Writing XDMF output to: {self.output_xdmf_str}\n")
        
        if self.write_checkpoint:
            # Intialize output file for checkpoints of the concentration
            self.checkpoint_filename = self.output_dir + "checkpoints/concentration/"
            a4d.write_mesh(mesh=self.mesh, filename=self.checkpoint_filename)
            print(f"Writing concentration checkpoints to: {self.checkpoint_filename}\n")

        if self.write_data:
            self.data_output_dir = self.output_dir + "data/"
            if self.comm.rank==0 and not os.path.isdir(self.data_output_dir): os.makedirs(self.data_output_dir)

            # Define integral measure for volume integrals in ROIs
            dx_roi = ufl.Measure('dx', domain=self.mesh, subdomain_data=self.ROI_ct)
            
            # Compile integration forms
            self.c_hat_forms = []
            for ROI_tag in self.ROI_tags:
                if ROI_tag==4:
                    # For ROI 4, we need to include the integrals of regions 1, 2 and 3
                    self.c_hat_forms.append(dfx.fem.form(self.c_h*dx_roi((ROI_tag, ROI_tag-1, ROI_tag-2, ROI_tag-3))))
                else:
                    self.c_hat_forms.append(dfx.fem.form(self.c_h*dx_roi(ROI_tag)))

            # Pre-allocate concentration arrays
            self.c_hat_arrays = np.zeros((self.num_timesteps, len(self.ROI_tags)), dtype=np.float64)

            print(f"Writing concentration data arrays to: {self.data_output_dir}\n")

        if self.write_snapshot_checkpoint:
            self.snapshot_times = self.record_periods*self.period
            self.snapshot_filename = self.output_dir + "checkpoints/concentration_snapshots/"
            a4d.write_mesh(mesh=self.mesh, filename=self.snapshot_filename)
            print(f"Writing concentration snapshots to: {self.snapshot_filename}")
            print(f"Writing the snapshots at times: {self.snapshot_times}\n")

        self.t_hat = dfx.fem.Function(W) # Finite element function for storing the times to threshold
        self.t_hat.x.array[:] = -1 # Initialize all t_hats as -1
        self.c_bar_threshold = 0.25 # Threshold concentration value

        

    def setup_solver(self):
        """ Create and configure PETSc Krylov solver. """

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
            self.solver.setType("gmres")
            self.solver.getPC().setType("bjacobi")
            self.solver.setTolerances(rtol=1e-10)

    def assemble_system_matrix(self):
        """ Assemble the system matrix of the variational problem. """

        self.A.zeroEntries() # Avoid accumulation of values
        assemble_matrix(self.A, self.a_cpp, bcs=self.bcs) # Assemble the system matrix
        self.A.assemble() # Assemble the PETSc matrix object

    def assemble_RHS(self):
        """ Assemble the right-hand side vector of the variational problem. """
    
        with self.b.localForm() as b_loc: b_loc.set(0) # Avoid accumulation of values
        assemble_vector(self.b, self.L_cpp) # Assemble the vector
        if len(self.bcs)>0:
            # Apply lifting of, and enforce, Dirichlet conditions
            apply_lifting(self.b, [self.a_cpp], bcs=[self.bcs]) # Apply lifting of the boundary conditions
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) # Perform ghost update: retrieve
            set_bc(self.b, bcs=self.bcs) # Set the boundary conditions
            self.b.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD) # Perform ghost update: send

    def calc_peclet_number(self):
        """ Calculate local Peclet number as the ratio between the norms
            of the advective flux and diffusive flux. """
        adv_flux_norm  = dot(self.c_h*self.u, self.c_h*self.u) # Advective flux norm (squared)
        diff_flux_norm = dot(self.D*grad(self.c_h), self.D*grad(self.c_h)) # Diffusive flux norm (squared)
        peclet_expr = dfx.fem.Expression(adv_flux_norm / (ufl.max_value(diff_flux_norm, 1e-14)), # Avoid division by zero
                                            self.W.element.interpolation_points(),
                                            comm=self.comm) 
        self.peclet.interpolate(peclet_expr)
        self.peclet.x.array[:] = np.sqrt(self.peclet.x.array[:])
        
    def run(self):
        """ Run transport simulations. """

        if self.model_version == 'A':
            a4d.read_function(u=self.u, filename=self.data_fname, engine="BP4", time=1)
            for i in range(self.num_timesteps):

                self.t += self.dt

                # Update Dirichlet condition
                self.bc_func.x.array[:] = self.experimental_concentration_curve()

                self.assemble_system_matrix()
                self.assemble_RHS()

                # Compute solution to the Advection-Diffusion equation and perform ghost update
                self.solver.solve(self.b, self.c_h.x.petsc_vec)
                self.c_h.x.scatter_forward()
                self.c_.interpolate(self.c_h)

                # Update solution at previous time step
                with self.c_h.x.petsc_vec.localForm() as c_h_loc, \
                     self.c_.x.petsc_vec.localForm()  as c_loc:
                    
                    idx = np.where(c_h_loc.array < 0)[0].astype(np.int32)
                    c_h_loc[idx] = np.zeros(idx.shape[0])
                    
                    c_loc[:] = c_h_loc[:]

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
                #--------  Post-process data  ---------#
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

        else:
            read_time = 0
            period_counter = 0
        
            for i in range(self.num_timesteps):
                
                # Increment time
                self.t += self.dt

                # Read velocity data
                if read_time==20:
                    # Previous timestep was at the end of the period of the 
                    # cardiac cycle -> reset the read_time variable to the first
                    # timestep of the cardiac cycle
                    read_time = 1
                    period_counter += 1
                else:
                    read_time += 1
                self.u_ = self.u
                a4d.read_function(u=self.u, filename=self.data_fname, engine="BP4", time=read_time) # velocity at this timestep

                # Update Dirichlet condition
                self.bc_func.x.array[:] = self.experimental_concentration_curve()

                # Assemble the system matrix and the right-hand side vector
                self.assemble_system_matrix()
                self.assemble_RHS()
                
                # Compute solution to the Advection-Diffusion equation and perform ghost update
                self.solver.solve(self.b, self.c_h.x.petsc_vec)
                self.c_h.x.scatter_forward()

                # Update solution at previous time step. Force negative concentrations
                # to being zero (as they are in the order of -1e-15) to avoid numerical issues.
                with self.c_h.x.petsc_vec.localForm() as c_h_loc, \
                     self.c_.x.petsc_vec.localForm()  as c_loc:
                    
                    idx = np.where(c_h_loc.array < 0)[0].astype(np.int32)
                    c_h_loc[idx] = np.zeros(idx.shape[0])
                    
                    c_loc[:] = c_h_loc[:] 
            
                # Print stuff
                print(f"Timestep t = {self.t}")
                print("Maximum concentration: ", self.comm.allreduce(self.c_h.x.array.max(), op=MPI.MAX))
                print("Minimum concentration: ", self.comm.allreduce(self.c_h.x.array.min(), op=MPI.MIN))

                
                total_c = dfx.fem.assemble_scalar(self.total_c)
                total_c = self.comm.allreduce(total_c, op=MPI.SUM)
                print(f"Total concentration: {total_c:.2e}")
                
                # Peclet number checkpointing
                if (self.write_peclet_checkpoint and period_counter in self.record_periods) or self.write_peclet_vtx:
                    self.calc_peclet_number()
                    if self.write_peclet_vtx: self.vtx_peclet.write(self.t)
                    if self.write_peclet_checkpoint: a4d.write_function(filename=f'{self.peclet_cpoint_fname}_period{period_counter}', u=self.peclet, time=read_time)

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

                #--------  Post-process data  ---------#
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
        if self.write_peclet_vtx: self.vtx_peclet.close()

if __name__ == '__main__':

    model_version = 'C'
    write_data = False
    write_output_vtx = False
    write_output_xdmf = False
    write_checkpoint  = False
    write_peclet_vtx = False
    write_peclet_checkpoint = False
    write_snapshot_checkpoint = False
    use_direct_solver = False
    relative = False
    flux_BC = True # if True, periodic in-/outflow BCs. Else, J.n=0 everywhere
    k = 1 # Concentration DG element degree
    
    #-----Create transport solver object-----#
    # Molecule options:
    # 'NPY' , 'DFP', 'STM_GFP', 'EXO'
    # Neuropeptide Y, Dendra Fluorescent Protein, Starmaker + Green Fluorescent Protein, Exosomes
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
        raise ValueError('Error in mesh version input. Choose 0, 1, 2 or 3.')
    
    # Temporal parameters
    f = 2.22 # Cardiac frequency [Hz]
    period = 1 / f # Cardiac period [s]
    T  = 1900*period # Simulation end time
    dt = period / 20 # Timestep size [s]
    print(f"Timestep: dt = {dt:.4g}")

    # Set velocity data filename
    tau_version = 'variable_tau'
    if relative:
        data_fname = f'../output/flow/checkpoints/{tau_version}/relative+{mesh_version}/model_{model_version}/velocity_data_dt=0.02252'
    else:
        data_fname = f'../output/flow/checkpoints/{tau_version}/pressure+{mesh_version}/model_{model_version}/velocity_data_dt=0.02252'

    # Create and set up solver object
    transport_sim = TransportSolver(data_fname=data_fname, model_version=model_version,
                                    mesh_version=mesh_version,
                                    T=T, dt=dt, period=period,
                                    molecule=molecule,
                                    element_degree=k, 
                                    relative=relative,
                                    tau_version=tau_version,
                                    flux_BC=flux_BC,
                                    write_data=write_data,
                                    write_peclet_vtx=write_peclet_vtx,
                                    write_peclet_checkpoint=write_peclet_checkpoint,
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
    print("Transport simulation information:")
    print("Model version: " + model_version)
    print("Mesh version: " + mesh_version)
    print("Molecule: " + molecule)
    print("Injection site: " + transport_sim.injection_site_str)
    print("Final time: ", T)
    print("Timestep size: ", dt)
    if write_output_vtx : print(f"Concentration VTX file written to {transport_sim.output_dir}")
    if write_output_xdmf: print(f"Concentration XDMF file written to {transport_sim.output_dir}")
    if write_checkpoint : print(f"Concentration checkpoints written to {transport_sim.output_dir}")
    if write_snapshot_checkpoint: print(f"Concentration snapshots written to {transport_sim.output_dir}")
    print(f"Solver time: {MPI.COMM_WORLD.allreduce(toc-tic, op=MPI.MAX):.2e}")
    print("#-------------------------#\n")