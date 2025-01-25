import multiphenicsx.fem
import multiphenicsx.fem.petsc
from imports.mesh                import mark_boundaries_flow
from imports.local_imports       import *
from imports.utilities           import tangential_projection
from imports.forcing_expressions import OscillatoryPressure

import adios2
import adios4dolfinx as a4d

print = PETSc.Sys.Print

# Mesh tags for flow
ANTERIOR_PRESSURE    = 2
POSTERIOR_PRESSURE   = 3
VOLUME               = 4
MIDDLE_VENTRAL_CILIA = 5
MIDDLE_DORSAL_CILIA  = 6
ANTERIOR_CILIA       = 7
SLIP                 = 9

class FlowSolver:
    """ Solver for the Stokes equations using a Lagrange multiplier for the normal stresses.

    """

    f = 2.22 # Cardiac frequency [Hz]

    def __init__(self, mesh: dfx.mesh.Mesh,
                       model_version: str,
                       write_output: bool = False,
                       pp: bool = False) -> None:
        """ Constructor.

        Parameters
        ----------
        mesh : dfx.mesh.Mesh
            Computational mesh.

        """

        self.mesh = mesh

        io_boundaries = False if model_version == 'A' else True
        self.ft   = mark_boundaries_flow(mesh=mesh, inflow_outflow=io_boundaries)
        self.model_version = model_version
        self.write_output = write_output
        self.pp  = pp
        

    def SetBilinearForm(self, u    : ufl.TrialFunction, v   : ufl.TestFunction,
                              p    : ufl.TrialFunction, q   : ufl.TestFunction,
                              zeta : ufl.TrialFunction, eta : ufl.TestFunction):
        """ Create and set the bilinear form for the Stokes equations variational problem. """
        eps   = lambda u: ufl.sym(grad(u)) # Symmetric gradient
        dx = self.dx
        ds = self.ds
        n  = self.n

        # Initialize form blocks
        a00 = a01 = a02 = 0
        a10 = 0
        a20 = 0

        if self.model_version == 'A':
            # whole boundary is slip BC
            a00 = 2*self.mu * inner(eps(u), eps(v)) * dx # Diffusive term
            a01 = - p * div(v) * dx       # Pressure term
            a02 = - zeta * dot(v, n) * ds   # Multiplier trial function term

            a10 = - q * div(u) * dx # Continuity equation
            
            a20 = - eta * dot(u, n) * ds # Multiplier test function term

        elif self.model_version == 'B' or self.model_version == 'C':
            # pressure boundary conditions at inlet/outlet
            ds_slip = ds(SLIP) + ds(ANTERIOR_CILIA) + ds(MIDDLE_VENTRAL_CILIA) + ds(MIDDLE_DORSAL_CILIA) # slip boundary integral

            a00 = 2*self.mu * inner(eps(u), eps(v)) * dx # Diffusive term
            a01 = - p * div(v) * dx       # Pressure term
            a10 = - q * div(u) * dx # Continuity equation

            # Multiplier only on slip boundary
            a02 = - zeta * dot(v, n) * ds_slip   # Multiplier trial function term
            a20 = - eta * dot(u, n) * ds_slip # Multiplier test function term

            # pressure boundary integrals
            a00 += - self.mu * inner(dot(grad(u).T, n), v) * (ds(ANTERIOR_PRESSURE) + ds(POSTERIOR_PRESSURE))        

        a = [[a00, a01, a02],
            [a10, None, None],
            [a20, None, None]]
        
        self.a_cpp = dfx.fem.form(a)


    def SetLinearForm(self, v: ufl.TestFunction, q: ufl.TestFunction, eta: ufl.TestFunction):
        """ Create and set the linear form for the Stokes equations variational problem. """

        dx = self.dx
        ds = self.ds

        # Define stuff used in the variational form
        zero = dfx.fem.Constant(self.mesh, PETSc.ScalarType(0)) # Zero value function

        f_body = dfx.fem.Constant(self.mesh, (0.0, ) * self.mesh.geometry.dim) # Zero body forces
        tau_0 = .000525*ufl.as_vector((1, 0, 1)) # Tangential stress on curved boundaries
        tau_0 = tangential_projection(tau_0, self.n)

        self.p_bc_expr = OscillatoryPressure(A=0.0005, f=self.f)
        self.p_bc   = dfx.fem.Function(self.Q)
        self.p_bc.interpolate(self.p_bc_expr)

        L0 = inner(f_body, v) * dx # Source term with force f = 0
        
        if self.model_version == 'A':
            # stress boundary condition
            L0 += inner( tau_0, v) * ds(MIDDLE_VENTRAL_CILIA)
            L0 += inner(-tau_0, v) * (ds(ANTERIOR_CILIA) + ds(MIDDLE_DORSAL_CILIA))

        elif self.model_version == 'B':
            # pressure boundary condition
            L0 += dot(self.p_bc*self.n, v) * ds(ANTERIOR_PRESSURE)

        elif self.model_version == 'C':
            # stress and pressure boundary conditions
            L0 += inner( tau_0, v) * ds(MIDDLE_VENTRAL_CILIA)
            L0 += inner(-tau_0, v) * (ds(ANTERIOR_CILIA) + ds(MIDDLE_DORSAL_CILIA))
            L0 += dot(self.p_bc*self.n, v) * ds(ANTERIOR_PRESSURE)
        
        L1 = inner(zero, q) * dx # Zero RHS for pressure test eq.

        L2 = inner(zero, eta) * ds # Zero RHS for multiplier test eq.

        L = [L0, L1, L2]

        self.L_cpp = dfx.fem.form(L)

    def SetNullspace(self):
        """ Create a PETSc.NullSpace for the underdetermined pressure and Lagrange multiplier. """

        # Create pressure nullspace vector
        pc1 = dfx.fem.Function(self.Q)
        pc1.interpolate(lambda x: -1*np.ones(x.shape[1]))
        PC1 = pc1.vector

        # Create Lagrange multiplier nullspace vector
        pc2 = dfx.fem.Function(self.Z)
        pc2.interpolate(lambda x: np.ones(x.shape[1]))
        PC2 = pc2.vector

        ns_vec = multiphenicsx.fem.petsc.create_vector_block(self.L_cpp, restriction=self.restriction)
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
            ns_vec, [self.V.dofmap, self.Q.dofmap, self.Z.dofmap], self.restriction) as ns_vec_wrapper:
            for ns_vec_wrapper_component_local, data_vector in zip(ns_vec_wrapper, (None, PC1, PC2)):
                if data_vector is not None:  # skip first block, since velocity is determined uniquely by the BCs
                    with data_vector.localForm() as data_vector_local:
                        ns_vec_wrapper_component_local[:] = data_vector_local

        # Normalize the nullspace vector
        ns_vec.scale(1 / ns_vec.norm())
        assert np.isclose(ns_vec.norm(), 1.0)

        self.nullspace = PETSc.NullSpace().create(vectors=[ns_vec], comm=self.mesh.comm)

    def solve_system(self):
        """ Solve the Stokes equations

                -div(sigma) = 0,
                 div(u)     = 0,

            where sigma is the stress tensor and u is the velocity. """

        # Solve and ghost update solution vector
        self.solver.solve(self.b, self.x)
        self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # Update previous solution
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(self.x, [self.V.dofmap, self.Q.dofmap, self.Z.dofmap], self.restriction) as u_p_wrapper:
            for u_p_wrapper_local, component in zip(u_p_wrapper, (self.u_h, self.p_h, self.z_h)):
                with component.vector.localForm() as component_local:
                    component_local[:] = u_p_wrapper_local
        
        if self.write_output:
            # Update output functions
            self.u_out.interpolate(self.u_h)
            self.u_out.x.array[:] *= 1e3 # scale to um/s
            self.p_out.interpolate(self.p_h)
            self.z_out.interpolate(self.z_h)
            

    def solve_timeloop(self):
        
        t = 0
        T = 1 / self.f
        dt = T / 20
        num_timesteps = int(T / dt) + 1

        # Initialize max values
        self.u_h_mag_max = 0
        self.p_h_max = 0
        self.u_L2_norm_max = 0
        
        write_time = 0
        filename = './output/checkpoints/TH/model_' + self.model_version + '/velocity_data_dt=' + f'{dt:.4g}'
        a4d.write_mesh(mesh=mesh, filename=filename, engine="BP4")
        a4d.write_function(u=self.u_h, filename=filename, engine="BP4", mode=adios2.Mode.Append, time=write_time)

        for _ in range(num_timesteps):
            # Increment time
            t += dt
            print(f"Time {t}")
            
            # Update time-dependent pressure BC
            self.p_bc_expr.t = t
            self.p_bc.interpolate(self.p_bc_expr)

            with self.b.localForm() as loc_b: loc_b.set(0)
            multiphenicsx.fem.petsc.assemble_vector_block(self.b, self.L_cpp, self.a_cpp, bcs=[], restriction=self.restriction)
            
            self.solve_system()

            # Calculate maximum velocity norm and max pressure
            u1 = self.u_h.sub(0).collapse().x.array
            u2 = self.u_h.sub(1).collapse().x.array
            u3 = self.u_h.sub(2).collapse().x.array
            u_h_mag = np.sqrt(u1**2 + u2**2 + u3**2)
            u_h_mag_max = u_h_mag.max()
            u_h_mag_max = max(self.u_h_mag_max, u_h_mag_max)
            self.u_h_mag_max = self.mesh.comm.allreduce(u_h_mag_max, op=MPI.MAX)
            p_h_max = max(np.abs(self.p_h_max), np.abs(self.p_h.vector.norm(norm_type=PETSc.NormType.NORM_INFINITY)))
            self.p_h_max = self.mesh.comm.allreduce(p_h_max, op=MPI.MAX)

            # Calculate velocity L2 norm
            self.u_L2_norm_max = max(self.u_L2_norm_max, self.calc_L2_norm(u=self.u_h))

            # Calculate divergence
            div_u_expr = dfx.fem.Expression(ufl.div(self.u_h), self.p_h.function_space.element.interpolation_points())
            div_u = dfx.fem.Function(self.p_h.function_space)
            div_u.interpolate(div_u_expr)
            print(f"Divergence max: {np.max(div_u.x.array[:])}")
            div_L2 = dfx.fem.assemble_scalar(dfx.fem.form(ufl.div(self.u_h)**2 * self.dx))
            print(f"Divergence L2 norm: {div_L2}")
            
            # Write to file
            write_time += 1
            a4d.write_function(u=self.u_h, filename=filename, engine="BP4", mode=adios2.Mode.Append, time=write_time)
            
            if write_output:
                self.vtx_u.write(t)
                self.vtx_p.write(t)
                self.vtx_z.write(t)
    
    def setup(self):
        self.n  = ufl.FacetNormal(self.mesh)
        self.dx = ufl.Measure("dx", domain=self.mesh)
        self.ds = ufl.Measure("ds", self.mesh, subdomain_data=self.ft) # Facet integral measure

        # Fluid parameters
        nu  = dfx.fem.Constant(self.mesh, PETSc.ScalarType(0.697)) # Kinematic viscosity [mm^2 / s]
        rho = dfx.fem.Constant(self.mesh, PETSc.ScalarType(1e-3)) # Density [g / mm^3]
        self.mu = dfx.fem.Constant(self.mesh, PETSc.ScalarType(nu.value * rho.value)) # Dynamic viscosity [g / (mm * s)]
        
        # Finite elements
        P2_vec = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), 2) # Quadratic Lagrange vector elements
        P2     = ufl.FiniteElement("Lagrange", self.mesh.ufl_cell(), 2) # Quadratic Lagrange elements
        P1     = ufl.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1) # Linear Lagrange elements

        # Function spaces
        self.V = V = dfx.fem.FunctionSpace(self.mesh, P2_vec)  # Velocity function space 
        self.Q = Q = dfx.fem.FunctionSpace(self.mesh, P1)  # Pressure function space 
        self.Z = Z = dfx.fem.FunctionSpace(self.mesh, P2)  # Multiplier space 
        
        # Set up restrictions
        gdim = self.mesh.topology.dim # Geometry dimension

        # Interior dofs for velocity and pressure
        dofs_V = np.arange(0, V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)
        dofs_Q = np.arange(0, Q.dofmap.index_map.size_local + Q.dofmap.index_map.num_ghosts)
        
        gamma_facets = np.concatenate((self.ft.find(SLIP), self.ft.find(ANTERIOR_CILIA), self.ft.find(MIDDLE_DORSAL_CILIA), self.ft.find(MIDDLE_VENTRAL_CILIA)))
        dofs_gamma   = dfx.fem.locate_dofs_topological(Z, gdim-1, gamma_facets)

        V_restr = multiphenicsx.fem.DofMapRestriction(V.dofmap, dofs_V)
        Q_restr = multiphenicsx.fem.DofMapRestriction(Q.dofmap, dofs_Q)
        Z_restr = multiphenicsx.fem.DofMapRestriction(Z.dofmap, dofs_gamma)

        self.restriction = [V_restr, Q_restr, Z_restr]

        # Print stuff
        size_local  = V.dofmap.index_map.size_local  + Q.dofmap.index_map.size_local  + Z.dofmap.index_map.size_local
        size_global = V.dofmap.index_map.size_global + Q.dofmap.index_map.size_global + Z.dofmap.index_map.size_global
        num_ghosts  = V.dofmap.index_map.num_ghosts  + Q.dofmap.index_map.num_ghosts  + Z.dofmap.index_map.num_ghosts
        
        print(f"MPI rank: {self.mesh.comm.rank}")
        print(f"Size of local index map: {size_local}")
        print(f"Size of global index map: {size_global}")
        print(f"Number of ghost nodes: {num_ghosts}")

        # Trial and test functions
        (u, p, zeta) = (ufl.TrialFunction(V), ufl.TrialFunction(Q), ufl.TrialFunction(Z))
        (v, q, eta ) = (ufl.TestFunction (V), ufl.TestFunction (Q), ufl.TestFunction (Z))

        # zeta  = sigma_hat . n    is the Lagrange multiplier 
        # sigma_hat = sigma . n    are the stresses

        # Get variational form
        self.SetBilinearForm(u, v, p, q, zeta, eta)
        self.SetLinearForm(v, q, eta)

        # Assemble system
        self.A = multiphenicsx.fem.petsc.assemble_matrix_block(self.a_cpp, bcs=[], restriction=(self.restriction, self.restriction))
        self.A.assemble()

        self.b = multiphenicsx.fem.petsc.assemble_vector_block(self.L_cpp, self.a_cpp, bcs=[], restriction=self.restriction)
        
        if self.model_version == 'A':
            # Provide A with the nullspaces of the pressure and the
            # Lagrange multiplier
            self.SetNullspace()
            assert self.nullspace.test(self.A) # Check that A @ nullspace = 0

            # Direct solver approach: set nullspace and orthogonalize
            # right-hand side vector w.r.t. the nullspace
            self.A.setNullSpace(self.nullspace)
            self.nullspace.remove(self.b)        

        # Set up solver
        self.solver = PETSc.KSP().create(self.mesh.comm)
        self.solver.setOperators(self.A)

        # Direct solver
        self.solver.setType("preonly")
        self.solver.getPC().setType("lu")
        self.solver.getPC().setFactorSolverType("mumps")

        # Create output files
        self.u_h   = dfx.fem.Function(V)
        self.u_out = dfx.fem.Function(dfx.fem.FunctionSpace(self.mesh, P2_vec))
        self.p_h   = dfx.fem.Function(Q)
        self.p_out = dfx.fem.Function(dfx.fem.FunctionSpace(self.mesh, P1))
        self.z_h   = dfx.fem.Function(Z)
        self.z_out = dfx.fem.Function(dfx.fem.FunctionSpace(self.mesh, P2))

        self.u_out.name = "vel"

        self.out_dir = './output/flow/Taylor_Hood/model_' + self.model_version
        u_out_str = self.out_dir + 'velocity.bp'
        p_out_str = self.out_dir + 'pressure.bp'
        z_out_str = self.out_dir + 'zeta.bp'

        if self.write_output:
            self.vtx_u = dfx.io.VTXWriter(self.mesh.comm, u_out_str, [self.u_out], engine="BP4")
            self.vtx_p = dfx.io.VTXWriter(self.mesh.comm, p_out_str, [self.p_out], engine="BP4")
            self.vtx_z = dfx.io.VTXWriter(self.mesh.comm, z_out_str, [self.z_out], engine="BP4")
            
            self.vtx_u.write(0)
            self.vtx_p.write(0)
            self.vtx_z.write(0)

        # Create solution vector
        self.x = multiphenicsx.fem.petsc.create_vector_block(self.L_cpp, restriction=self.restriction)

    def timestep(self, t: float):

        # Update time-dependent pressure BC
        self.p_bc_expr.t = t
        self.p_bc.interpolate(self.p_bc_expr)

        # Assemble right-hand side
        with self.b.localForm() as loc_b: loc_b.set(0)
        multiphenicsx.fem.petsc.assemble_vector_block(self.b, self.L_cpp, self.a_cpp, bcs=[], restriction=self.restriction)

        # Solve system
        self.solve_system()

    def run(self):
        if self.model_version == 'A':
            self.solve_system()
            write_adios = True
            if write_adios:
                a4d.write_mesh(mesh=mesh, filename='./output/checkpoints/TH/model_A/velocity_data', engine="BP4")
                a4d.write_function(u=self.u_h, filename='./output/checkpoints/TH/model_A/velocity_data', engine="BP4")

        elif self.model_version == 'B' or self.model_version == 'C':
            self.solve_timeloop()

        if self.pp: self.post_process()
        
    def post_process(self):
        comm = self.mesh.comm # MPI communicator

        if self.model_version == 'A':
            # Calculate maximum velocity norm
            u1 = self.u_h.sub(0).collapse().x.array
            u2 = self.u_h.sub(1).collapse().x.array
            u3 = self.u_h.sub(2).collapse().x.array
            u_h_mag = np.sqrt(u1**2 + u2**2 + u3**2)
            u_h_mag_max = u_h_mag.max()

            cilia_area = dfx.fem.assemble_scalar(dfx.fem.form(1 * (self.ds(MIDDLE_DORSAL_CILIA) + self.ds(MIDDLE_VENTRAL_CILIA) + self.ds(ANTERIOR_CILIA))))
            cilia_area = comm.allreduce(cilia_area, op=MPI.SUM)
            self.u_h_mag_max = comm.allreduce(u_h_mag_max, op=MPI.MAX)
            self.u_h_mag_max_scaled = comm.allreduce(u_h_mag_max/cilia_area, op=MPI.MAX)

            # Calculate mean pressure and subtract it from the calculated pressure
            vol = comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1 * self.dx)), op=MPI.SUM)
            if comm.rank==0:
                print(f"Volume vol = {vol}")
            mean_p_h = comm.allreduce(1/vol * dfx.fem.assemble_scalar(dfx.fem.form(self.p_h * self.dx)), op=MPI.SUM)

            self.p_h.x.array[:] -= mean_p_h
            
            # Get maximum pressure
            p_h_max = self.p_h.vector.norm(norm_type=PETSc.NormType.NORM_INFINITY)
            self.p_h_max = comm.allreduce(p_h_max, op=MPI.MAX)

            # Calculate pressure L2 norm
            self.p_L2_norm_max = self.calc_L2_norm(u=self.p_h)

            # Calculate velocity L2 norm        
            self.u_L2_norm_max = self.calc_L2_norm(u=self.u_h)

            # Calculate divergence
            div_u_ufl = ufl.div(self.u_h)
            div_u_expr = dfx.fem.Expression(div_u_ufl, self.p_h.function_space.element.interpolation_points())
            div_u = dfx.fem.Function(self.p_h.function_space)
            div_u.interpolate(div_u_expr)
            print(f"Divergence max: {np.max(div_u.x.array[:])}")
            div_L2 = dfx.fem.assemble_scalar(dfx.fem.form(ufl.div(self.u_h)**2 * self.dx))
            print(f"Divergence L2 norm: {div_L2}")
            
            if self.write_output:
                # Write output
                self.vtx_u.write(0)
                self.vtx_p.write(0)
                self.vtx_z.write(0)

        # else: for model B and C values have been calculated in the timeloop

        if comm.rank == 0:
            # Print stuff
            print("Printing")
            print(f"Max velocity: {self.u_h_mag_max}")
            if self.model_version=='A': print(f"Max velocity scaled: {self.u_h_mag_max_scaled}")
            print(f"Max pressure: {self.p_h_max}")
            print(f"L2 norm velocity: {self.u_L2_norm_max:.2e}")
    

    def calc_L2_norm(self, u):
        vol = dfx.fem.assemble_scalar(dfx.fem.form(1 * self.dx))
        vol = self.mesh.comm.allreduce(vol, op=MPI.SUM)
        u_L2 = dfx.fem.form(inner(u, u) * self.dx)
        u_L2_norm_local = 1/vol * dfx.fem.assemble_scalar(u_L2)
        u_L2_norm_global = self.mesh.comm.allreduce(u_L2_norm_local, op=MPI.SUM)

        return np.sqrt(u_L2_norm_global)


if __name__ == '__main__':

    comm = MPI.COMM_WORLD # MPI communicator

    # Model options
    model_version = 'C'
    write_output = False
    post_process = True
    mesh_filename = "./geometries/ventricles_4.xdmf"
    
    with dfx.io.XDMFFile(comm, mesh_filename, "r") as xdmf:
        mesh = xdmf.read_mesh()

    print(f"Total # of cells in mesh: {mesh.topology.index_map(3).size_global}")
    
    #-----Create flow solver object-----#
    sim = FlowSolver(mesh=mesh, model_version=model_version,
                     write_output=write_output, pp=post_process)
                     
    #-----Perform setup and run flow simulation-----#
    sim.setup()
    sim.run()