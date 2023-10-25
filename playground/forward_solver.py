import dolfinx as dlx
from mpi4py import MPI
import matplotlib.pyplot as plt
import pyvista
import ufl
import numpy as np
import petsc4py

STATE= 0
PARAMETER = 1
ADJOINT = 2
NVAR = 3

def vector2Function(x,Vh, **kwargs):
    """
    Wrap a finite element vector :code:`x` into a finite element function in the space :code:`Vh`.
    :code:`kwargs` is optional keywords arguments to be passed to the construction of a dolfin :code:`Function`.
    """
    fun = dlx.fem.Function(Vh,**kwargs)
    fun.interpolate(lambda x: np.full((x.shape[1],),0.))
    fun.vector.axpy(1., x)
    
    return fun

def Transpose(A):
    # Amat = dlx.fem.as_backend_type(A).mat()
    # AT = PETSc.Mat()
    # Amat.transpose(AT)
    # rmap, cmap = Amat.getLGMap()
    # AT.setLGMap(cmap, rmap)
    # return dl.Matrix( dl.PETScMatrix(AT) )

    #A -> petsc4py.PETSc.Mat object obtained from a ufl.form expression,
    #example:
    #A = dlx.fem.petsc.create_matrix(dlx.fem.form(a_goal)) #petsc4py.PETSc.Mat

    A.assemble()
    # AT = petsc4py.PETSc.Mat()
    AT = petsc4py.PETSc.Mat()
    AT = A.transpose()
    rmap,cmap = A.getLGMap()
    AT.setLGMap(cmap,rmap)
    return AT 

class DiffusionApproximation:
    def __init__(self, D, u0, ds):
        """
        Define the forward model for the diffusion approximation to radiative transfer equations
        
        D: diffusion coefficient 1/mu_eff with mu_eff = sqrt(3 mu_a (mu_a + mu_ps) ), where mu_a
           is the unknown absorption coefficient, and mu_ps is the reduced scattering coefficient
           
        u0: Incident fluence (Robin condition)
        
        ds: boundary integrator for Robin condition
        """
        
        self.D = D
        self.u0 = u0
        self.ds = ds
        
    def __call__(self, u, m, p):
        # return ufl.inner(self.D*ufl.grad(u), ufl.grad(p))*ufl.dx + \
        #        ufl.exp(m)*ufl.inner(u,p)*ufl.dx + \
        #        dl.Constant(.5)*ufl.inner(u-self.u0,p)*self.ds 
        
        return ufl.inner(self.D*ufl.grad(u), ufl.grad(p))*ufl.dx + \
            ufl.exp(m)*ufl.inner(u,p)*ufl.dx + \
            .5*ufl.inner(u-self.u0,p)*self.ds 
    

# pde = PDEVariationalProblem(Vh, pde_handler, [], [],  is_fwd_linear=True)
class PDEVariationalProblem:
    def __init__(self, Vh, varf_handler, bc, bc0, is_fwd_linear=False):
        self.Vh = Vh
        self.varf_handler = varf_handler
        
        # if isinstance(bc, dlx.fem.dirichletbc):
        #     self.bc = [bc]  
        # else:
        #     self.bc = bc

        if(str(type(bc)) == "<class 'dolfinx.fem.bcs.DirichletBC'>" ):
            self.bc = [bc]
        else:
            self.bc = bc

        if(str(type(bc0)) == "<class 'dolfinx.fem.bcs.DirichletBC'>" ):
            self.bc0 = [bc0]
        else:
            self.bc0 = bc0

        self.A = None
        self.At = None
        self.C = None
        self.Wmu = None
        self.Wmm = None
        self.Wuu = None

        self.solver = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None

        self.is_fwd_linear = is_fwd_linear
        self.n_calls = {"forward": 0, "adjoint": 0, "incremental_forward": 0, "incremental_adjoint": 0}

    def generate_state(self):
        """ Return a vector in the shape of the state. """
        return dlx.fem.Function(self.Vh[STATE]).vector

    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        return dlx.fem.Function(self.Vh[PARAMETER]).vector()

    def init_parameter(self, m):
        """ Initialize the parameter."""
        dummy = self.generate_parameter()
        m.init(dummy.mpi_comm(), dummy.dofmap.index_map)

    # pde.solveFwd(u_true, x_true)
    def solveFwd(self, state, x):
        # print('inside at start',id(state))

        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that
            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""
        self.n_calls["forward"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()
            # print(type(self.solver))
        if self.is_fwd_linear:
            # print(self.bc) #[]
            u = ufl.TrialFunction(self.Vh[STATE])
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = ufl.TestFunction(self.Vh[ADJOINT])
            res_form = self.varf_handler(u, m, p)   
            A_form = ufl.lhs(res_form)
            b_form = ufl.rhs(res_form)
            
            A = dlx.fem.petsc.assemble_matrix(dlx.fem.form(A_form),bcs=self.bc)
            A.assemble() #petsc4py.PETSc.Mat
            # A.view()
            self.solver.setOperators(A)
            b = dlx.fem.petsc.assemble_vector(dlx.fem.form(b_form))
            dlx.fem.petsc.apply_lifting(b,[dlx.fem.form(A_form)],[self.bc])
            b.ghostUpdate(petsc4py.PETSc.InsertMode.ADD_VALUES,petsc4py.PETSc.ScatterMode.REVERSE)
            dlx.fem.petsc.set_bc(b,self.bc)
            b.assemble() #petsc4py.PETSc.Vec
            # b.view()
            self.solver.solve(b,state)
            print('inside at end',id(state[0]))

            for i in range(10):
                print(state[i],id(state[i]))
    
            # return state
            
            # A  = dlx.fem.petsc.assemble_matrix(dlx.fem.form(A_form),self.bc)
            # A.assemble()
            # # A.view()
            # # print(type(A))
            # self.solver.setOperators(A)
            # b = dlx.fem.petsc.create_vector(dlx.fem.form(b_form))
            # b.assemble()
            # # b.view()
            # self.solver.solve(b,state) #KSP requires solve(b,x)
        # else:
        #     u = vector2Function(x[STATE], self.Vh[STATE])
        #     m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        #     p = TestFunction(self.Vh[ADJOINT])
        #     res_form = self.varf_handler(u, m, p)
        #     solve(res_form == 0, u, bcs=self.bc)
        #     state.zero()
        #     state.axpy(1.0, u.vector())

    def _createLUSolver(self):   
        # return PETScLUSolver(self.Vh[STATE].mesh().mpi_comm() )
        ksp = petsc4py.PETSc.KSP().create()
        # ksp.setType("preonly")
        pc = ksp.getPC()
        # pc.setType("ilu")
        ksp.setFromOptions()
        return ksp

    
def master_print(comm, *args, **kwargs):
    if comm.rank == 0:
        print(*args, **kwargs)


sep = "\n"+"#"*80+"\n"

comm = MPI.COMM_WORLD
rank  = comm.rank
nproc = comm.size

msh = dlx.cpp.mesh.Mesh
fname = 'meshes/circle.xdmf'
fid = dlx.io.XDMFFile(comm,fname,"r")
msh = fid.read_mesh(name='mesh')

#Question: Is the mesh read in correctly to define the function spaces over it??
#Ans: Yes
# f = dlx.fem.Constant(msh,petsc4py.PETSc.ScalarType(1.0))

Vh_phi = dlx.fem.FunctionSpace(msh, ("CG", 1)) 
Vh_m = dlx.fem.FunctionSpace(msh, ("CG", 1))

Vh = [Vh_phi, Vh_m, Vh_phi]

# master_print(comm, sep, "Set up the mesh and finite element spaces", sep)
#need to find substitute of dolfin.function.constant.Constant(1.) in dolfinx

# u0 = dlx.fem.Constant(1.)
# u0 = dlx.fem.Constant(msh,petsc4py.PETSc.ScalarType(1.))
# D = dlx.fem.Constant(msh,petsc4py.PETSc.ScalarType(1./24.))

u0 = 1.
D = 1./24.


# GROUND TRUTH
# m_true_expr = dlx.Expression("std::log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )", degree=1)
# m_true = dl.interpolate(m_true_expr, Vh_m).vector()
m_true = dlx.fem.Function(Vh_m)
m_true.interpolate(lambda x: np.log(0.01) + 3.*( ( ( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) < 1.) )  )
m_true = m_true.vector
# print(type(m_true))

pde_handler = DiffusionApproximation(D, u0, ufl.ds) #returns a ufl form
pde = PDEVariationalProblem(Vh, pde_handler, [], [],  is_fwd_linear=True)

u_true = pde.generate_state()
# print(type(u_true))
x_true = [u_true, m_true, None]

# print('before fwd solve call',id(u_true))

pde.solveFwd(u_true, x_true)
print('\n')
print('after fwd solve call',id(u_true[0]))

# print(type(u_true)) #petsc4py.PETSc.Vec

# print("u_true_results")

for i in range(10):
    print(u_true[i],id(u_true[i]))

# for i in range(10):
#     print(np.array(u_true)[i])

# for i in range(10):
#     print(u_true[i])
#utrue contains results of A.u_true = b ; A,b -> LHS, RHS of the weak form
# print(type(x_true))
# x = dlx.fem.dirichletbc(4.,)
