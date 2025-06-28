import numpy as np
from numba import jit,njit,prange
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as vnp

# This function just takes the center pc of the spheroid and a 
# point pi, and checks if pi is inside a spheroid with normal n, length l and aspect ratio e. 
@jit(nopython=True)
def inside(pc,pi,n,l,e,D):
    value=False
    j1=-1
    while j1<2:
        j2=-1
        while j2<2:
            j3=-1
            while j3<2:
                vec=np.array([j1*D,j2*D,j3*D])
                pi_=pi+vec
                u=pi_-pc
                x=np.dot(u,n)
                if x<l/2:
                    u_n=u-x*n
                    a=l/2
                    b=l/2/e
                    y=np.linalg.norm(u_n)
                    if (x/a)**2+(y/b)**2<=1:
                        value=True
                        j1=2
                        j2=2
                        j3=2
                    else:
                        j3+=1
                else:
                    j3+=1
            j2+=1
        j1+=1
    return value
    
# This function defines (im,jm,km) (maybe outside the cell) and
# (iM,jM,kM) (maybe outside also) that defines like a parallelepipedic box outside of which there cannot be any voxel of the spheroid. It permits to fasten the voxellization.
# Then, it uses the function "inside" as follows: a voxel is inside the spheroid if the point situated at the smallest x,y,z of the voxel is inside the spheroid.
@jit(nopython=True)
def fill(micro_v,c,n,l,e,i,D):
    N0,N1,N2=micro_v.shape
    xc,yc,zc=c
    xc=xc+D/2
    yc=yc+D/2
    zc=zc+D/2
    p1=c-l/2*n+D/2*np.array([1,1,1])
    p2=c+l/2*n+D/2*np.array([1,1,1])
    x1,y1,z1=p1
    x2,y2,z2=p2
    r=l/e/2
    i1=int(x1*N0)
    j1=int(y1*N1)
    k1=int(z1*N2)
    i2=int(x2*N0)
    j2=int(y2*N1)
    k2=int(z2*N2)
    im=min(i1,i2)-int(r*N0)-1
    iM=max(i1,i2)+int(r*N0)+1
    jm=min(j1,j2)-int(r*N1)-1
    jM=max(j1,j2)+int(r*N1)+1
    km=min(k1,k2)-int(r*N2)-1
    kM=max(k1,k2)+int(r*N2)+1
    for ii in range(im,iM):
        for jj in range(jm,jM):
            for kk in range(km,kM):
                if ii>=N0:
                    ii-=N0
                if jj>=N1:
                    jj-=N1
                if kk>=N2:
                    kk-=N2
                xi=ii/N0
                yi=jj/N1
                zi=kk/N2
                pi=np.array([xi,yi,zi])
                pc=np.array([xc,yc,zc])
                value=inside(pc,pi,n,l,e,D)
                if value:
                    micro_v[ii,jj,kk]=i

#This function gives number 0 for matrix and 1 for all spheroids
def voxelize_ell(micro,N0,l,e,D):
    micro_v=np.zeros((N0,N0,N0))
    for i in range(len(micro)):
        print(i)
        c,n,ang=micro[i]
        fill(micro_v,c,n,l,e,1,D)
    return micro_v

#This function allows to give a number to each spheroid
def voxelize_ell_n(micro,N0,l,e,D):
    micro_v=np.zeros((N0,N0,N0))
    for i in range(len(micro)):
        print(i)
        c,n,ang=micro[i]
        fill(micro_v,c,n,l,e,i+1,D)
    return micro_v

#just for using prange (parallel loop)
@jit
def compute_n(i,N,dim):
    n=[]
    for p in range(dim):
        n.append(0)
    i_=i
    for ind in range(1,dim+1):
        q=i_
        for j in range(ind,dim):
            q/=N[j]
        q=int(q)
        n[ind-1]=q
        for j in range(ind,dim):
            q*=N[j]
        i_-=q
    return n

#each voxel is divided in 8 voxels with the same value
@njit(parallel=True)
def raffine(micro_v,N,dim,NN):  
    micro_v2=np.zeros((2*N[0],2*N[1],2*N[2]))
    for i in prange(NN):
        n=compute_n(i,N,dim)
        ph=micro_v[n[0],n[1],n[2]]
        if ph!=0:
            micro_v2[2*n[0]:2*n[0]+2,2*n[1]:2*n[1]+2,2*n[2]:2*n[2]+2]=ph
    return micro_v2,np.array([2*N[0],2*N[1],2*N[2]]),8*NN

#just for reading a .vtk file
def read_vtk(file_name, field_name):
    reader = vtkStructuredPointsReader()
    reader.SetFileName(file_name)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    data = reader.GetOutput()
    dim = data.GetDimensions()
    vec = [i - 1 for i in dim]
    e0 = vnp.vtk_to_numpy(data.GetCellData().GetArray(champ_name))
    return e0.reshape(vec, order="F")

#just for writing a .vtk file
def write_vtk(mic,name,N):
	VTK_HEADER = f"""# vtk DataFile Version 4.5
Materiau
BINARY
DATASET STRUCTURED_POINTS
DIMENSIONS    {N+1}   {N+1}   {N+1}
ORIGIN    0.000   0.000   0.000
SPACING   {1.:.7e} {1.:.7e} {1.:.7e}
CELL_DATA   {N*N*N}
SCALARS MaterialId unsigned_short
LOOKUP_TABLE default
"""
	data = np.zeros((N, N, N), dtype=">u2")
	for i in np.ndindex((N,N,N)):
		ii,jj,kk=i
		data[ii,jj,kk]=mic[kk,jj,ii]
	with open(name+".vtk", "wb") as file:
		file.write(VTK_HEADER.encode())
		data.tofile(file)
