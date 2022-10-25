import numpy as np
from numba import njit, f8, f4, i4, b1
from numba.types import Tuple

CACHE = True
PARALLEL = False
float_datatype = np.float64 # Choose from np.float64 and np.float32
if float_datatype == np.float64:
	fp = f8
elif float_datatype == np.float32:
	fp = f4

bb1 = np.zeros((6,6),dtype=np.float64) # for linear stiffness matrix
bb1[0,0] = bb1[3,3] = 1
bb1[0,3] = bb1[3,0] = -1

@njit(Tuple((fp[:,:,:],fp[:]))(fp[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL)
def TransformationMatrices(node,member):
	'''
	(input)
	node[nn,3]<float> : nodal locations (x,y coordinates) [mm]
	member[nm,3]<int> : member connectivity

	(output)
	tt[nm,6,6]<float> : transformation matrices
	length[nm]<float> : member lengths [mm]
	'''
	nm = np.shape(member)[0]
	dxyz = np.zeros((nm,3),dtype=float_datatype)
	length = np.zeros(nm,dtype=float_datatype)
	for i in range(nm):
		dxyz[i] = node[member[i,1],:] - node[member[i,0],:]
		length[i] = np.linalg.norm(dxyz[i])
	tt = np.zeros((nm,6,6),dtype=float_datatype)
	for i in range(nm):
		tt[i,0:3,0] = dxyz[i]/length[i]
	flag = np.abs(tt[:,0,0]) >= 0.9
	tt[flag,1,1] = 1.0
	tt[~flag,0,1] = 1.0
	for i in range(nm):
		for j in range(3):
			tt[i,j,2] = tt[i,(j+1)%3,0] * tt[i,(j+2)%3,1] - tt[i,(j+2)%3,0] * tt[i,(j+1)%3,1]
		tt[i,:,2] /= np.linalg.norm(tt[i,:,2])
		for j in range(3):
			tt[i,j,1] = tt[i,(j+1)%3,2] * tt[i,(j+2)%3,0] - tt[i,(j+2)%3,2] * tt[i,(j+1)%3,0]
	tt[:,3:,3:] = tt[:,:3,:3]
 
	return tt, length

@njit(Tuple((fp[:],fp[:,:,:],fp))(fp[:,:],i4[:,:],b1[:,:],fp[:]),cache=CACHE,parallel=PARALLEL)
def StiffnessMatrixEig(node0,member,support,A):
	### node[nk,3]: 節点位置
	### connectivity[nm,2]: 部材接続関係
	### support[nk,3]: True if supported, else False # ただし、Isolated nodeはsupportをTrueにすることで剛性行列の構成時に無視できる
	### A[nm]: Cross-sectional area. # 剛性行列のランクを正しく計算するために除去したとみなす部材は微小値ではなく必ず0の断面積を入力すること

	### Organize input model
	nn = node0.shape[0] # number of nodes
	nm = member.shape[0] # number of members
	free = np.logical_not(support.flatten()) # DOFs are True, otherwise False

	### linear stiffness matrix
	tt,ll0 = TransformationMatrices(node0,member)
	tt = np.ascontiguousarray(tt)

	kel = np.zeros((nm,6,6),dtype=float_datatype)
	for i in range(nm):
		kel[i] = np.dot(tt[i],A[i]/ll0[i]*bb1)
		kel[i] = np.dot(kel[i],tt[i].transpose())

	Ka = np.zeros((3*nn,3*nn),float_datatype) # linear stiffness matrix
	for i in range(nm): # assemble element stiffness matries into one matrix
		Ka[3*member[i,0]:3*member[i,0]+3,3*member[i,0]:3*member[i,0]+3] += kel[i,0:3,0:3]
		Ka[3*member[i,0]:3*member[i,0]+3,3*member[i,1]:3*member[i,1]+3] += kel[i,0:3,3:6]
		Ka[3*member[i,1]:3*member[i,1]+3,3*member[i,0]:3*member[i,0]+3] += kel[i,3:6,0:3]
		Ka[3*member[i,1]:3*member[i,1]+3,3*member[i,1]:3*member[i,1]+3] += kel[i,3:6,3:6]

	K = Ka[free][:,free]
		
	eig_val,eig_vec = np.linalg.eigh(K)

	u = np.zeros((len(eig_val),nn*3),dtype=float_datatype)
	for i in range(len(eig_val)):
		uu = u[i] # This is shallow copy, and u also changes in the next line
		uu[free] = eig_vec[:,i]
	eig_mode = u.reshape((len(eig_val),nn,3))

	return eig_val,eig_mode,len(eig_val)-np.linalg.matrix_rank(K)

@njit(Tuple((fp[:,:],fp[:],fp[:,:]))(fp[:,:],i4[:,:],b1[:,:],fp[:,:],fp[:],fp[:]),cache=CACHE,parallel=PARALLEL)
def StructuralAnalysis(node0,member,support,load,A,E):

	### node[nk,3]: 節点位置
	### member[nm,2]: 部材接続関係
	### support[nk,3]: True if supported, else False
	### load[nk,3]: Load magnitude. 0 if no load is applied.
	### A[nm]: Cross-sectional area.
	### E[nm]: Young's modulus.

	### Organize input model
	nn = node0.shape[0] # number of nodes
	nm = member.shape[0] # number of members
	free = np.logical_not(support.flatten()) # DOFs are True, otherwise False
	pp = load.flatten()[free]

	### linear stiffness matrix
	tt,ll0 = TransformationMatrices(node0,member)
	tt = np.ascontiguousarray(tt)

	kel = np.zeros((nm,6,6),dtype=float_datatype)
	for i in range(nm):
		kel[i] = np.dot(tt[i],E[i]*A[i]/ll0[i]*bb1)
		kel[i] = np.dot(kel[i],tt[i].transpose())

	Ka = np.zeros((3*nn,3*nn),float_datatype) # linear stiffness matrix
	for i in range(nm): # assemble element stiffness matries into one matrix
		Ka[3*member[i,0]:3*member[i,0]+3,3*member[i,0]:3*member[i,0]+3] += kel[i,0:3,0:3]
		Ka[3*member[i,0]:3*member[i,0]+3,3*member[i,1]:3*member[i,1]+3] += kel[i,0:3,3:6]
		Ka[3*member[i,1]:3*member[i,1]+3,3*member[i,0]:3*member[i,0]+3] += kel[i,3:6,0:3]
		Ka[3*member[i,1]:3*member[i,1]+3,3*member[i,1]:3*member[i,1]+3] += kel[i,3:6,3:6]

	K = Ka[free][:,free]
	Up = np.linalg.solve(K,pp) # Compute displacement
	U = np.zeros(nn*3,dtype=float_datatype)
	U[free] = Up
	deformation = U.reshape((nn,3))

	node = node0 + deformation
	tt,ll = TransformationMatrices(node,member)

	strain = (ll-ll0)/ll0
	stress = (strain * E) # 軸方向応力=ひずみ×ヤング係数(引張正)

	K2 = Ka[~free][:,free]
	Rp = np.dot(K2,Up)
	R = np.zeros(nn*3,dtype=float_datatype)
	R[~free] = Rp
	R[~free] -= load.flatten()[~free]
	reaction = R.reshape((nn,3))

	return deformation, stress, reaction

# node = np.array([[0,0,0],[1,0,0]],dtype=float_datatype)
# connectivity = np.array([[0,1]],dtype=np.int32)
# support = np.array([[1,1,1],[1,1,1]],dtype=bool)
# load = np.array([[1,0,0],[0,0,0]],dtype=float_datatype)
# A = np.array([1.0],dtype=float_datatype)
# E = np.array([1.0],dtype=float_datatype)

# node = np.array([[0,0,0],[8,0,0],[4,4,0]],dtype=float_datatype)
# connectivity = np.array([[0,1],[0,2],[1,2]],dtype=np.int32)
# support = np.array([[1,1,1],[0,1,1],[0,0,1]],dtype=bool)
# load = np.array([[0,0,0],[0,0,0],[0,-1,0]],dtype=float_datatype)
# A = np.array([1.0,1.0,1.0],dtype=float_datatype)
# E = np.array([1.0,1.0,1.0],dtype=float_datatype)

# d,s,r = StructuralAnalysis(node,connectivity,support,load,A,E)

# import time
# t1 = time.perf_counter()
# for i in range(100):
# 	d,s,c = StructuralAnalysis(node,connectivity,support,load,A,E)
# t2 = time.perf_counter()
# print("d={0}".format(d))
# print("s={0}".format(s))
# print("time={0}".format(t2-t1))

# node = np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0],[2,1,0]],dtype=float_datatype)
# connectivity = np.array([[0,1],[1,2],[3,4],[4,5],[1,4],[2,5],[0,4],[1,3],[1,5],[2,4]],dtype=np.int32)
# support = np.array([[1,1,1],[0,0,1],[0,0,1],[1,1,1],[0,0,1],[0,0,1]],dtype=bool)
# A = np.array([1,0,1,1,1,1,1,1,0,1],dtype=float_datatype)

# eig_val, eig_mode = StiffnessMatrixEig(node,connectivity,support,A)

# import plotter
# node2 = np.concatenate([node,node+eig_mode[np.where(eig_val<1.0e-5)[0][0]]*0.5])
# connectivity2 = np.concatenate([connectivity,connectivity+node.shape[0]])
# plotter.Draw(node2[:,0:2],connectivity2,np.tile(A,2),node_color=[(0.8,0.8,0.8)]*node.shape[0]+[(0.0,0.0,0.0)]*node.shape[0],edge_color=[(0.8,0.8,0.8)]*connectivity.shape[0]+[(0.4,0.4,0.4)]*connectivity.shape[0],show=True)