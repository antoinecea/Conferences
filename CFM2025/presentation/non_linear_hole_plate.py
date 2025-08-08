## type: ignore
import numpy as np
import os
import time
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, geometry
from dolfinx.io import XDMFFile
from dolfinx.cpp.nls.petsc import NewtonSolver
from ufl import dot, grad, TestFunction, TrialFunction, Measure, dx
from dolfinx_materials.solvers import SNESNonlinearMaterialProblem
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.material.mfront import MFrontMaterial
from dolfinx.mesh import meshtags
from dolfinx_materials.utils import symmetric_tensor_to_vector
import meshio

comm = MPI.COMM_WORLD
rank = comm.rank
size=comm.size
current_path = os.getcwd()

L = 6.
e=10.


with XDMFFile(MPI.COMM_WORLD, "plate_hole_20k.xdmf", "r") as file:
    domain=file.read_mesh(name="Grid")

dim = domain.topology.dim

degree = 2
shape = (dim,)

V = fem.functionspace(domain, ("P", degree, shape))

# Loading the behaviour 
material = MFrontMaterial(
    os.path.join(current_path, "mfront_laws/src/libBehaviour.so"),
    #"Idiart_elastic_inclusions",
    #material_properties={'ka0': 2.e9,'mu0': 1.e9,'kav0': 0.4e9,'muv0': 0.2e9,'kar': 100.e9,'mur': 50.e9,'tau0': 0.2,'fr': 0.1,'e': 1.},
    #"Idiart_random_elastic_inclusions",
    #material_properties={'ka0': 2.e9,'mu0': 1.e9,'kav0': 0.4e9,'muv0': 0.2e9,'kar': 100.e9,'mur': 50.e9,'e': 10.,'fr': 0.1},
   "Molinari_Explicit",
    material_properties={'E_0': 8.181818e9,'nu_0': 0.363636,'sigy0': 100.e6,'kk0': 0.,'fi': 0.17,'E_i': 16.363636e9,'nu_i': 0.363636,'sigyi': 100.e12,'kki': 0.},
)       

if rank == 0:
    print(material.internal_state_variable_names)
    print(material.gradient_names, material.gradient_sizes)
    print(material.flux_names, material.flux_sizes)


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
    

right_facets = mesh.locate_entities_boundary(domain, dim - 1, right)
left_facets = mesh.locate_entities_boundary(domain, dim - 1, left)

p0_dofs = fem.locate_dofs_geometrical(V, point0)
p1_dofs = fem.locate_dofs_geometrical(V, point1)
V0=V.sub(0)
V_x, _ = V0.collapse()
right_comb_dofs = fem.locate_dofs_topological((V0, V_x), dim - 1, right_facets)
left_comb_dofs = fem.locate_dofs_topological((V0, V_x), dim - 1, left_facets)
uD = fem.Function(V_x)
uL = fem.Function(V_x)
uL.x.array[:]=0.
bcs = [
    fem.dirichletbc(uL, left_comb_dofs, V0),
    fem.dirichletbc(uD, right_comb_dofs, V0),
    fem.dirichletbc(np.zeros((dim,)), p0_dofs, V),
]

du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u = fem.Function(V, name="Displacement")

qmap = QuadratureMap(domain, 2, material)
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
facet_tag=meshtags(domain,dim-1,indices[sorted_f],markers[sorted_f])
ds=Measure("ds",domain=domain,subdomain_data=facet_tag)

#force=-1e6
Res = (ufl.dot(sigma, eps(v))) * qmap.dx #+ force*v[0]*ds(0) 
Jac = qmap.derivative(Res, u, du)

num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
if rank == 0:
    print(num_dofs_global)


problem = SNESNonlinearMaterialProblem(qmap, Res, Jac, u, bcs)

snes = PETSc.SNES().create()
snes.setType('newtonls')
snes.getKSP().setType("preonly")
snes.getKSP().getPC().setType("lu")
snes.getKSP().getPC().setFactorSolverType("mumps")
snes.getKSP().setTolerances(atol=1e-6, rtol=1e-6)
PETSc.Options()['snes_monitor'] = None
PETSc.Options()['snes_linesearch_monitor'] = None
snes.setFromOptions()

#Molinari Explicit
dt_=[0.05,0.04,0.01]
dt_+=100*[0.005]

#visco linear
#dt_=100*[0.02]
Nincr=len(dt_)

t=0.
t_=[]
for dt in dt_:
    t+=dt
    t_.append(t)
if rank==0:
    print(t_)

#vtk = io.VTKFile(domain.comm, f"results/fields/{material.name}.pvd", "w")
#file=open(f"results/{material.name}_40k.txt",'a')
#zero=np.zeros((13,))
#if rank==0:
#    file.write(str(zero[0]))
#    for jj in range(6):
#        file.write(" "+str(zero[jj]))
#    for jj in range(6):
#        file.write(" "+str(zero[jj]))
#    file.write("\n")
#    file.flush()

domain.topology.create_connectivity(dim-1, dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, dim-1, boundary_facets)
dof_coordinates = V.tabulate_dof_coordinates()[boundary_dofs]
points = domain.geometry.x
points = np.array([[0., 0.,L],[0.,0.,0.]], dtype=points.dtype)
points_ = np.array([[0., 0.,L]], dtype=points.dtype)

for i, t in enumerate(t_[:]):
    material.dt=dt_[i]
    if rank==0:
        print(t)
        print(dt_[i])
    uD.x.array[:]=t*0.12*L
    if MPI.COMM_WORLD.rank==0:
    	t0=time.time()
    converged, it =problem.solve(snes, print_solution=False)
    if MPI.COMM_WORLD.rank==0:
    	print("temps total : ",time.time()-t0)
    if rank == 0:
        print(f"Increment {i+1} converged in {it} iterations.")
    
    sig = qmap.project_on("Stress", ("Lagrange", 1))
    
    bb_tree = geometry.bb_tree(domain,3)
    cells = []
    points_on_proc = []
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    for ii, point in enumerate(points):
        if len(colliding_cells.links(ii))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(ii)[0])       
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    expr1 = fem.Expression(eps(u), points_)
    eps_values = expr1.eval(domain, cells)
    sig_values = sig.eval(points_on_proc, cells)
    #if rank == size - 3:
    #    print(rank,eps_values)
    #    print(rank,sig_values)
    #    file.write(str(t))
    #    for jj in range(6):
    #        file.write(" "+str(eps_values[0][jj]))
    #    for jj in range(6):
    #        file.write(" "+str(sig_values[jj]))
    #    file.write("\n")
    #    file.flush()
#vtk.close()



    
    













