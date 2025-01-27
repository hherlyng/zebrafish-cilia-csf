import ufl
import typing
import numpy.typing

import numpy     as np
import dolfinx   as dfx

from ufl      import inner, grad, div, dot, dx
from mpi4py   import MPI
from petsc4py import PETSc