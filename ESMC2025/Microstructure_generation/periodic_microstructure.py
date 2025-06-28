import numpy as np
from random import uniform
from numba import jit

@jit(nopython=True)
def dist(p1,p2):
    return np.linalg.norm(p1-p2)
    
@jit(nopython=True)
def dist_point_segment(a,a2,b2):
    l=np.linalg.norm(b2-a2)
    t2=-np.dot(b2-a2,a2-a)/l**2
    if t2<0:
        return dist(a,a2)
    if t2>1:
        return dist(a,b2)
    else:
        p1=a
        p2=a2+t2*(b2-a2)
        return dist(p1,p2)
        
@jit(nopython=True)
def dist_segment(a1,b1,a2,b2):
    l1=np.linalg.norm(b1-a1)
    l2=np.linalg.norm(b2-a2)
    if np.dot(b2-a2,b1-a1)**2==l1**2*l2**2:
        if -np.dot(b2-a2,a2-a1)/l2**2<0 and (np.dot(b2-a2,b1-a1)-np.dot(b2-a2,a2-a1))/l2**2<-np.dot(b2-a2,a2-a1)/l2**2:
            p1=a1
            p2=a2
            return dist(p1,p2)
        if (np.dot(b2-a2,b1-a1)-np.dot(b2-a2,a2-a1))/l2**2<0 and (np.dot(b2-a2,b1-a1)-np.dot(b2-a2,a2-a1))/l2**2>-np.dot(b2-a2,a2-a1)/l2**2:
            p1=b1
            p2=a2
            return dist(p1,p2)
        if -np.dot(b2-a2,a2-a1)/l2**2>1 and (np.dot(b2-a2,b1-a1)-np.dot(b2-a2,a2-a1))/l2**2>-np.dot(b2-a2,a2-a1)/l2**2:
            p1=a1
            p2=b2
            return dist(p1,p2)
        if (np.dot(b2-a2,b1-a1)-np.dot(b2-a2,a2-a1))/l2**2>1 and (np.dot(b2-a2,b1-a1)-np.dot(b2-a2,a2-a1))/l2**2<-np.dot(b2-a2,a2-a1)/l2**2:
            p1=b1
            p2=b2
            return dist(p1,p2)
        else:
            t1=0
            t2=-np.dot(b2-a2,a2-a1)/l2**2
            p1=a1+t1*(b1-a1)
            p2=a2+t2*(b2-a2)
            return dist(p1,p2)   
    else:    
        t1=(l2**2*np.dot(b1-a1,a2-a1)-np.dot(b2-a2,a2-a1)*np.dot(b2-a2,b1-a1))/(l1**2*l2**2-(np.dot(b2-a2,b1-a1))**2)
        t2=(t1*np.dot(b2-a2,b1-a1)-np.dot(b2-a2,a2-a1))/l2**2
        if t1>1 or t1<0 or t2>1 or t2<0:
            d=dist_point_segment(a1,a2,b2)
            d2=dist_point_segment(b1,a2,b2)
            if d2<d:
                d=d2
            d3=dist_point_segment(a2,a1,b1)
            if d3<d:
                d=d3
            d4=dist_point_segment(b2,a1,b1)
            if d4<d:
                d=d4
            return d
        else:
            p1=a1+t1*(b1-a1)
            p2=a2+t2*(b2-a2)
            return dist(p1,p2)



@jit(nopython=True)
def test_overlap(c1,c2,l,r,R):
    bool_overlap=False
    j1=-1
    while j1<2:
        j2=-1
        while j2<2:
            j3=-1
            while j3<2:
                k1=-1
                while k1<2:
                    k2=-1
                    while k2<2:
                        k3=-1
                        while k3<2:
                            if not((c1==c2).all()) or j1!=k1 or j2!=k2 or j3!=k3:
                                vect1=np.array([j1*2*R,j2*2*R,j3*2*R])
                                c10_per=c1[0]+vect1
                                vect2=np.array([k1*2*R,k2*2*R,k3*2*R])
                                c20_per=c2[0]+vect2
                                a1=c10_per-c1[1]*l/2
                                b1=c10_per+c1[1]*l/2
                                a2=c20_per-c2[1]*l/2
                                b2=c20_per+c2[1]*l/2
                                d=dist_segment(a1,b1,a2,b2)
                                if d<=2*r:
                                    bool_overlap=True
                                    j1=2
                                    j2=2
                                    j3=2
                                    k1=2
                                    k2=2
                                    k3=2
                                else:
                                    k3+=1
                            else:
                                k3+=1
                        k2+=1
                    k1+=1
                j3+=1
            j2+=1
        j1+=1
    return bool_overlap

@jit(nopython=True)
def is_near(c1,c2,R,l,r):
    bool_near=False    
    j1=-1
    while j1<2:
        j2=-1
        while j2<2:
            j3=-1
            while j3<2:
                k1=-1
                while k1<2:
                    k2=-1
                    while k2<2:
                        k3=-1
                        while k3<2:
                            vect1=np.array([j1*2*R,j2*2*R,j3*2*R])
                            vect2=np.array([k1*2*R,k2*2*R,k3*2*R])
                            if dist(c1+vect1,c2+vect2)<=l+2*r:
                                bool_near=True
                                j1=2
                                j2=2
                                j3=2
                                k1=2
                                k2=2
                                k3=2
                            else:
                                k3+=1
                        k2+=1
                    k1+=1
                j3+=1
            j2+=1
        j1+=1
    return bool_near

@jit(nopython=True)
def appendjit_m(micro,cyl,index):
    if index==0:
        n=0
    else:
        n=len(micro)
    micro2=np.zeros((n+1,3,3))
    for i in range(n):
        micro2[i]=micro[i]
    micro2[n]=cyl
    return micro2
    
@jit(nopython=True)
def add(m,c,R,l,r,index):
    length=len(m)
    bool_test=True
    i=0
    while i<length:
        cyl=m[i]
        if is_near(cyl[0],c[0],R,l,r):
            if test_overlap(cyl,c,l,r,R):
                bool_test=False
                i=length
                #print('overlap')
            else:
                i+=1
        else:
            i+=1
    if test_overlap(c,c,l,r,R):
        bool_test=False
        #print('overlap')
    if bool_test:
        m=appendjit_m(m,c,index)

    return bool_test,m
                
        
#warning R must be > l/2
@jit(nopython=True)
def generate_micro(R,l,r,f):
    Vol=np.pi*r**2*l
    Voltot=8*R**3
    N=int(f*Voltot/Vol)
    print(N)
    m=np.zeros((1,3,3))
    i=0
    while i<N:
        center=np.array([uniform(-R,R),uniform(-R,R),uniform(-R,R)])
        theta=np.arccos(1-2*uniform(0,1))
        phi=uniform(0,2*np.pi)
        normal=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        angle=np.array([theta*180/np.pi,phi*180/np.pi,0])
        c=np.array([[center[0],center[1],center[2]],[normal[0],normal[1],normal[2]],[angle[0],angle[1],angle[2]]])
        bool1,m=add(m,c,R,l,r,i)
        if bool1:
            i+=1
            print("put inclusion ", i)
    return m


