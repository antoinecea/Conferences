
import numpy as np
import os
import ufl
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, geometry
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import create_matrix, create_vector
from dolfinx.cpp.nls.petsc import NewtonSolver
from dolfinx_materials.solvers import CustomNewtonProblem
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.material.mfront import MFrontMaterial
from dolfinx_materials.jax_materials import LinearElasticAnisotropic
from dolfinx.mesh import create_box, CellType, meshtags
from dolfinx_materials.utils import (
    symmetric_tensor_to_vector,
)
import tfel.math as tmath
from tfel.material import KGModuli
import tfel.material.homogenization as homo

import meshio

comm = MPI.COMM_WORLD
rank = comm.rank
current_path = os.getcwd()

from ufl import (
    dot,
    grad,
    TestFunction,
    TrialFunction,
    sym,
    inner,
    Measure,
    dx,
)

L = 6.
e=10.

domain=meshio.read("plate_hole.msh")

for cell in domain.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "tetra":
        tetra_cells = cell.data
for key in domain.cell_data_dict["gmsh:physical"].keys():
    if key == "triangle":
        triangle_data = domain.cell_data_dict["gmsh:physical"][key]
    elif key == "tetra":
        tetra_data = domain.cell_data_dict["gmsh:physical"][key]


tetra_mesh =meshio.Mesh(points=domain.points, cells={"tetra": tetra_cells}, cell_data={"data": [tetra_data]})
meshio.write("plate_hole.xdmf", tetra_mesh)


with XDMFFile(MPI.COMM_WORLD, "plate_hole.xdmf", "r") as file:
    domain=file.read_mesh(name="Grid")

dim = domain.topology.dim
#print(f"Mesh topology dimension d={dim}.")

degree = 2
shape = (dim,)

V = fem.functionspace(domain, ("Lagrange", degree, shape))

scheme_name="Mori_Tanaka"
E0=1e8
Ei=1e9
nu0=0.2
nui=0.3
#frac=0.1
na=tmath.TVector3D()
na[0]=1.
na[1]=0.
na[2]=0.
nb=tmath.TVector3D()
nb[0]=0.
nb[1]=1.
nb[2]=0.

material = MFrontMaterial(
    os.path.join(current_path, "mfront_laws/src/libBehaviour.so"),
   "MoriTanaka",
    material_properties={'E0': E0,'nu0': nu0,'Ei': Ei,'nui': nui,'a': 20,'b': 1.,'c': 1.}
)       

#Ce=np.array(homo.computeOrientedMoriTanakaScheme(E0,nu0,frac,Ei,nui,na,20.,nb,1.,1.))
#material = LinearElasticAnisotropic(Ce)

def eps(v):
    return symmetric_tensor_to_vector(ufl.grad(v))

def left(x):
    return np.isclose(x[0], 0.)

def right(x):
    return np.isclose(x[0], L)
    

def point0(x):
    return np.isclose(x[0],0.) & np.isclose(x[1],0.5*L) & np.isclose(x[2],0.)
    
def point1(x):
    return np.isclose(x[0],0.) & np.isclose(x[1],0.5*L) & np.isclose(x[2],L/e)
    
def fraction(x):
    return 0.1*(0*x[1]+1.)#np.heaviside(x[1],0.)
    
def normal_a(x):
    theta=(2*abs(x[1]/L-0.5))*np.pi/2
    phi=0.
    return np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)
    
        
right_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, right)
left_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, left)

p0_dofs = fem.locate_dofs_geometrical(V, point0)
p1_dofs = fem.locate_dofs_geometrical(V, point1)
V0=V.sub(0)
V_x, _ = V0.collapse()
right_comb_dofs = fem.locate_dofs_topological((V0, V_x), domain.topology.dim - 1, right_facets)
left_comb_dofs = fem.locate_dofs_topological((V0, V_x), domain.topology.dim - 1, left_facets)
uD = fem.Function(V_x)
uL = fem.Function(V_x)
uL.x.array[:]=0.
bcs = [
    fem.dirichletbc(uL, left_comb_dofs, V0),
    #fem.dirichletbc(uD, right_comb_dofs, V0),
    fem.dirichletbc(np.zeros((dim,)), p0_dofs, V),
    #fem.dirichletbc(np.zeros((dim,)), p1_dofs, V)
]


du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u = fem.Function(V, name="Displacement")

qmap = QuadratureMap(domain, 2, material)

W=fem.functionspace(domain, ("Lagrange", degree))
frac = fem.Function(W)
frac.interpolate(fraction)
na = fem.Function(V)
na.interpolate(normal_a)

qmap.register_external_state_variable("frac", frac)
qmap.register_external_state_variable("na", na)
qmap.register_gradient("Strain", eps(u))
sigma = qmap.fluxes['Stress']

indices=[]
markers=[]
all_boundary_facets = mesh.exterior_facet_indices(domain.topology)
remaining_facets = np.setdiff1d(all_boundary_facets, right_facets)
indices.append(remaining_facets)
markers.append(np.full_like(remaining_facets,1))
indices.append(right_facets)
markers.append(np.full_like(right_facets,0))
 
indices=np.hstack(indices).astype(np.int32)
markers=np.hstack(markers).astype(np.int32)
sorted_f=np.argsort(indices)
facet_tag=meshtags(domain,domain.topology.dim-1,indices[sorted_f],markers[sorted_f])
ds=Measure("ds",domain=domain,subdomain_data=facet_tag)

force=-1e6
#Res = (ufl.dot(sigma, eps(v))) * qmap.dx
Res = (ufl.dot(sigma, eps(v))) * qmap.dx + force*v[0]*ds(0) 
Jac = qmap.derivative(Res, u, du)

problem = CustomNewtonProblem(qmap, Res, Jac, u, bcs)
newton = NewtonSolver(comm)
newton.rtol = 1e-7
newton.atol = 1e-7
newton.convergence_criterion = "residual"
newton.report = True
# Set solver options
ksp = newton.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

vtk = io.VTKFile(domain.comm, f"results/fields/linear_{scheme_name}.pvd", "w")

#file=open(f"results/linear_{scheme_name}.txt",'a')

domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, boundary_facets)
dof_coordinates = V.tabulate_dof_coordinates()[boundary_dofs]
points = domain.geometry.x
points = np.array([[0., 0.,L],[0.,0.,0.]], dtype=points.dtype)
points_ = np.array([[0., 0.,L]], dtype=points.dtype)

#uD.x.array[:]=0.012*L

converged, it = problem.solve(ksp,print_solution=False)

sig = qmap.project_on("Stress", ("Lagrange", 1))

vtk.write_function(sig)

bb_tree = geometry.bb_tree(domain,3)
cells = []
points_on_proc = []
# Find cells whose bounding-box collide with the the points
cell_candidates = geometry.compute_collisions_points(bb_tree, points)
# Choose one of the cells that contains the point
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
for ii, point in enumerate(points):
    if len(colliding_cells.links(ii))>0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(ii)[0])       
points_on_proc = np.array(points_on_proc, dtype=np.float64)
expr1 = fem.Expression(eps(u), points_)
eps_values = expr1.eval(domain, cells)
print(eps_values)

sig_values = sig.eval(points_on_proc, cells)
print(sig_values)

    
vtk.close()














