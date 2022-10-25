from cmath import sqrt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from numba import f8, i4, b1
import numba as nb

CACHE = False

@nb.jit(nb.types.Tuple((i4,i4,f8[:,:],i4[:,:],i4[:]))(i4,i4,b1),parallel=True,cache=CACHE)
def InitializeGeometry2D_grid_singlebrace(nx,ny,test):
	
	# node
	nk = (nx+1)*(ny+1)
	node = np.zeros((nk,3),dtype=np.float64)
	for i in range(nk):
		iy, ix = np.divmod(i,nx+1)
		node[i,1] = iy
		node[i,0] = ix

	# member
	nm = (1+3*ny)*nx+ny
	connectivity = np.zeros((nm,2),dtype=np.int32)

	count = 0
	# horizontal member
	for i in range(ny+1):
		for j in range(nx):
			connectivity[count,0] = i*(nx+1)+j
			connectivity[count,1] = i*(nx+1)+j+1
			count += 1
	# vertical member
	for i in range(ny):
		for j in range(nx+1):
			connectivity[count,0] = i*(nx+1)+j
			connectivity[count,1] = (i+1)*(nx+1)+j
			count += 1
	# bracing member
	for i in range(ny):
		for j in range(nx):
			connectivity[count,0] = i*(nx+1)+j
			connectivity[count,1] = (i+1)*(nx+1)+j+1
			count += 1

	if test:
		pin_nodes = np.array([0,nx],dtype=np.int32)
	else:
		pin_nodes = np.random.choice(nx+1,np.random.randint(2,nx+1),replace=False).astype(np.int32).flatten()

	return nk,nm,node,connectivity,pin_nodes

@nb.jit(nb.types.Tuple((i4,i4,f8[:,:],i4[:,:],i4[:]))(i4,i4,b1),parallel=True,cache=CACHE)
def InitializeGeometry2D_grid_doublebrace(nx,ny,test):
	
	# node
	nk = (nx+1)*(ny+1)
	node = np.zeros((nk,3),dtype=np.float64)
	for i in range(nk):
		iy, ix = np.divmod(i,nx+1)
		node[i,1] = iy
		node[i,0] = ix

	# member
	nm = (1+4*ny)*nx+ny
	connectivity = np.zeros((nm,2),dtype=np.int32)

	count = 0
	# horizontal member
	for i in range(ny+1):
		for j in range(nx):
			connectivity[count,0] = i*(nx+1)+j
			connectivity[count,1] = i*(nx+1)+j+1
			count += 1
	# vertical member
	for i in range(ny):
		for j in range(nx+1):
			connectivity[count,0] = i*(nx+1)+j
			connectivity[count,1] = (i+1)*(nx+1)+j
			count += 1
	# bracing member
	for i in range(ny):
		for j in range(nx):
			connectivity[count,0] = i*(nx+1)+j
			connectivity[count,1] = (i+1)*(nx+1)+j+1
			count += 1
			connectivity[count,0] = i*(nx+1)+j+1
			connectivity[count,1] = (i+1)*(nx+1)+j
			count += 1

	if test:
		pin_nodes = np.array([0,nx],dtype=np.int32)
	else:
		pin_nodes = np.random.choice(nx+1,np.random.randint(2,nx+1),replace=False).astype(np.int32).flatten()

	return nk,nm,node,connectivity,pin_nodes


@nb.jit(nb.types.Tuple((i4,i4,f8[:,:],i4[:,:],i4[:]))(),parallel=True,cache=CACHE)
def InitializeGeometry2D_arch():

	divi = 8
	R = 3.0
	r = 2.0

	nk = (divi+1)*2
	nm = (divi*4)+1

	node = np.zeros((nk,3),dtype=np.float64)
	for i in range(divi+1):
		node[2*i,0] = -R*np.cos(np.pi/divi*i) # x座標
		node[2*i,1] = R*np.sin(np.pi/divi*i) # y座標
		node[2*i+1,0] = -r*np.cos(np.pi/divi*i) # x座標
		node[2*i+1,1] = r*np.sin(np.pi/divi*i) # y座標
		
	# member
	connectivity = np.zeros((nm,2),dtype=np.int32)
	count = 0
	# pillar member
	for i in range(divi+1):
		connectivity[count] = [2*i,2*i+1]
		count += 1
	# beam member
	for i in range(divi):
		connectivity[count] = ([2*i,2*i+2])
		count += 1
		connectivity[count] = [2*i+1,2*i+3]
		count += 1
	# bracing member
	for i in range(int(divi/2)):
		connectivity[count] = [4*i,4*i+3]
		count += 1
		connectivity[count] = [4*i+3,4*i+4]
		count += 1

	pin_nodes = np.argwhere(node[:,1] < 0.0001).astype(np.int32).flatten()

	return nk,nm,node,connectivity,pin_nodes

@nb.jit(nb.types.Tuple((i4,i4,f8[:,:],i4[:,:],i4[:]))(),parallel=True,cache=CACHE)
def InitializeGeometry2D_tower():

	# number of nodes and members
	nk = 44 # 節点数
	nm = 88 # 部材数

	# node
	node = np.zeros((nk,3),dtype=np.float64) # 節点位置

	for i in range(2):
		for j in range(9):
			for k in range(2):
				num = i*18+j*2+k
				width = j
				width2 = 3*j
				if j >= 5:
					width -= (j-4)*0.5
					#width2 -= (j-4)
				node[num,0] = (-11 + 3*k + width)*((-1)**(i+2))
				node[num,1] = width2

	node[36,0] = -2.5
	node[36,1] = 9
	node[37,0] = -2
	node[37,1] = 12
	node[38,0] = 0
	node[38,1] = 9
	node[39,0] = 0
	node[39,1] = 12
	node[40,0] = 2.5
	node[40,1] = 9
	node[41,0] = 2
	node[41,1] = 12

	node[42,0] = 0
	node[42,1] = 21
	node[43,0] = 0
	node[43,1] = 24

	# member
	connectivity_array = []

	# horizontal member
	for i in range(18):
		connectivity_array.append((2*i,2*i+1))
	for i in range(4):
		connectivity_array.append((36+i,36+i+2))
	connectivity_array.append((7,36))
	connectivity_array.append((9,37))
	connectivity_array.append((24,40))
	connectivity_array.append((26,41))
	connectivity_array.append((15,42))
	connectivity_array.append((17,43))
	connectivity_array.append((32,42))
	connectivity_array.append((35,43))

	# vertical member
	for i in range(2):
		for j in range(8):
			for k in range(2):
				connectivity_array.append((i*18+2*j+k,i*18+2*j+k+2))
	for i in range(4):
		connectivity_array.append((36+2*i,36+2*i+1))

	# bracing member
	for i in range(8):
		connectivity_array.append((2*i,2*i+3)) # connectivity_array.append((2*i+1,2*i+2))
	for i in range(8):
		connectivity_array.append((18+2*i,18+2*i+3))
	connectivity_array.append((7,37))
	connectivity_array.append((36,39))
	connectivity_array.append((39,40))
	connectivity_array.append((25,41))
	connectivity_array.append((15,43))
	connectivity_array.append((33,43))
	connectivity = np.asarray(connectivity_array,dtype=np.int32) # 部材がどの2節点に接続しているか

	pin_nodes = np.argwhere(node[:,1] < 0.0001).astype(np.int32).flatten()

	return nk,nm,node,connectivity,pin_nodes


@nb.jit(nb.types.Tuple((i4,i4,f8[:,:],i4[:,:],i4[:]))(i4,i4,b1),parallel=True,cache=CACHE)
def InitializeGeometry3D_grid(nx,ny,test):

	'''
	nx,ny: numbers of grids in the upper layer
	'''
	
	# node
	nk = nx*ny+(nx+1)*(ny+1)
	node = np.zeros((nk,3),dtype=np.float64)
	for i in range(nx*ny): # bottom layer
		iy, ix = np.divmod(i,nx)
		node[i,1] = iy
		node[i,0] = ix
	for i in range((nx+1)*(ny+1)): # upper layer
		iy, ix = np.divmod(i,(nx+1))
		node[nx*ny+i,1] = iy
		node[nx*ny+i,0] = ix
	node[nx*ny:,0:2] -= 0.5
	node[nx*ny:,2] += 0.5

	# member
	'''
	nm = nx*(ny-1)+(nx-1)*ny # bottom members
	+ (nx+1)*ny+nx*(ny+1) # upper members
	+ nx*ny*4 # brace members
	= nx*ny*8
	'''
	nm = nx*ny*8

	connectivity = np.zeros((nm,2),dtype=np.int32)

	count = 0
	# x-axis member in the bottom layer
	for i in range(ny):
		for j in range(nx-1):
			connectivity[count,0] = i*nx+j
			connectivity[count,1] = i*nx+j+1
			count += 1
	# y-axis member in the bottom layer
	for i in range(ny-1):
		for j in range(nx):
			connectivity[count,0] = i*nx+j
			connectivity[count,1] = (i+1)*nx+j
			count += 1
	# x-axis member in the upper layer
	for i in range(ny+1):
		for j in range(nx):
			connectivity[count,0] = nx*ny+i*(nx+1)+j
			connectivity[count,1] = nx*ny+i*(nx+1)+j+1
			count += 1
	# y-axis member in the upper layer
	for i in range(ny):
		for j in range(nx+1):
			connectivity[count,0] = nx*ny+i*(nx+1)+j
			connectivity[count,1] = nx*ny+(i+1)*(nx+1)+j
			count += 1
	# bracing member
	for i in range(ny):
		for j in range(nx):
			connectivity[count,0] = i*nx+j
			connectivity[count,1] = nx*ny+i*(nx+1)+j
			count += 1
			connectivity[count,0] = i*nx+j
			connectivity[count,1] = nx*ny+i*(nx+1)+j+1
			count += 1
			connectivity[count,0] = i*nx+j
			connectivity[count,1] = nx*ny+(i+1)*(nx+1)+j
			count += 1
			connectivity[count,0] = i*nx+j
			connectivity[count,1] = nx*ny+(i+1)*(nx+1)+j+1
			count += 1

	if test:
		pin_nodes = np.array([0,nx-1,(ny-1)*nx,nx*ny-1],dtype=np.int32)
	else:
		ixs = np.random.choice(nx,2,replace=False)
		iys = np.random.choice(ny,2,replace=False)
		pin_nodes = np.array([nx*iy+ix for iy in iys for ix in ixs],dtype=np.int32)

	return nk,nm,node,connectivity,pin_nodes


@nb.jit(nb.types.Tuple((i4,i4,f8[:,:],i4[:,:],i4[:]))(i4,i4),parallel=False,cache=CACHE)
def InitializeGeometry3D_dome(nr,ng):

	'''
	nr: number of Ngons in the radius direction
	ng: N of Ngon
	'''
	
	# node
	nk = ng*nr+1
	node = np.zeros((nk,3),dtype=np.float64)
	rise = 0.5*nr
	for i in range(nr):
		z = np.sqrt((nr-i-1)/nr)*rise
		for j in range(ng):
			r = 1.0*(i+1)
			x = r*np.cos(2*np.pi*(j+0.5*(i%2))/ng)
			y = r*np.sin(2*np.pi*(j+0.5*(i%2))/ng)
			node[1+ng*i+j,0] = x
			node[1+ng*i+j,1] = y
			node[1+ng*i+j,2] = z
	node[0,2] = rise

	# member
	'''
	nm = ng # bars from the center
	+ ng*(nr-1)*2 # 2 bars from interior nodes (except the center)
	+ ng*(nr-1) # bars in the circumferential direction
	= ng*(3*nr-2)
	'''
	nm = ng*(3*nr-2)
	connectivity = np.zeros((nm,2),dtype=np.int32)

	count = 0
	# bars from the center
	for i in range(ng):
		connectivity[count,0] = 0
		connectivity[count,1] = i+1
		count += 1
	# 2 bars from interior nodes (except the center)
	for i in range(nr-1):
		for j in range(ng):
			connectivity[count,0] = 1+ng*i+j
			connectivity[count,1] = 1+ng*(i+1)+j
			count += 1
			connectivity[count,0] = 1+ng*i+j
			if i%2 == 0:
				connectivity[count,1] = 1+ng*(i+1)+(j-1)%ng
			else:
				connectivity[count,1] = 1+ng*(i+1)+(j+1)%ng
			count += 1
	# ng*(nr-1) # bars in the circumferential direction
	for i in range(nr-1):
		for j in range(ng):
			connectivity[count,0] = 1+ng*i+j
			connectivity[count,1] = 1+ng*i+(j+1)%ng
			count += 1

	pin_nodes = np.arange(nk-ng,nk,dtype=np.int32)

	return nk,nm,node,connectivity,pin_nodes

# def set_load(self):

# 	# loading condition
# 	self.load = np.zeros((self.nk,3),dtype=np.float64)
# 	for i in range(self.nm):
# 		self.load[self.connectivity[i],1] += self.lengths[i]*self.material[i]/2
# 	return