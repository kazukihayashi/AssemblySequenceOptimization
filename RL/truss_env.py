import numpy as np
np.random.seed(0)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import plotter
import truss_analysis
from numba import i4, b1, f8
import numba as nb
import init_geom

CACHE = False

@nb.njit((b1[:])(i4,i4,i4[:,:],b1[:]),parallel=True,cache=CACHE)
def NodeExist(nk,nm,connectivity,existence):
	node_existence = np.zeros(nk,dtype=np.bool_)
	for i in range(nm):
		if existence[i]:
			node_existence[connectivity[i,:]] = True
	return node_existence

def NodalStability(nk,node,connectivity,connected_members,dimension):
	node_stable = np.ones(nk,dtype=bool)
	for i in range(nk):
		if len(connected_members[i]) < dimension:
			node_stable[i] = False
		else:
			index = connectivity[connected_members[i]].flatten()
			index = np.delete(index, np.argwhere(index == i))
			matrix = node[index] - node[i]
			if np.linalg.matrix_rank(matrix) < dimension:
				node_stable[i] = False
	return node_stable


class Truss():

	def __init__(self):
		self.reset()

	def reset(self,test=0):

		### 2D ###
		self.dim = 2
		if test==1:
			self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry2D_grid_singlebrace(1,3,test=True)
			self.nsize = 10
			self.lsize = 4
			self.spsize = 17
		elif test==2:
			self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry2D_arch()
			self.nsize = 10
			self.lsize = 4
			self.spsize = 17
		elif test==3:
			self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry2D_tower()
		else:
			self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry2D_grid_doublebrace(np.random.randint(3,6),np.random.randint(3,6),test=False)
		self.support = np.zeros((self.nk,3),dtype=bool)
		self.support[:,2] = True # !!! only 2D
		self.support[self.pin_nodes] = True

		if test == 0:
			self.node = np.copy(self.init_node)
			for i in range(self.nk):
				if np.random.rand() < 0.05:
					self.node[i,0:2] += np.random.rand(2)*0.2-0.1
		else:
			self.node = np.copy(self.init_node)
		##########

		# ### 3D ###
		# self.dim = 3
		# if test==1:
		# 	self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry3D_grid(5,2,test=True)
		# 	self.nsize = 10
		# 	self.lsize = 4
		# 	self.spsize = 17
		# elif test==2:
		# 	self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry3D_dome(3,8)
		# 	self.nsize = 18
		# 	self.lsize = 6
		# 	self.spsize = 30
		# elif test==3:
		# 	self.init_node = np.loadtxt("input/node.txt",dtype=float)
		# 	self.connectivity = np.loadtxt("input/member.txt",dtype=int)
		# 	self.pin_nodes = np.loadtxt("input/pin_node.txt",dtype=int)
		# 	self.nk = self.init_node.shape[0]
		# 	self.nm = self.connectivity.shape[0]
		# 	self.nsize = 1
		# 	self.lsize = 1.5
		# 	self.spsize = 10
		# else:
		# 	if np.random.rand() < 0.5:
		# 		self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry3D_grid(np.random.randint(2,5),np.random.randint(2,5),test=False)
		# 	else:
		# 		self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry3D_dome(np.random.randint(2,5),np.random.randint(5,8))
		# self.support = np.zeros((self.nk,3),dtype=bool)
		# self.support[self.pin_nodes] = True

		# if test == 0:
		# 	self.node = np.copy(self.init_node)
		# 	for i in range(self.nk):
		# 		if np.random.rand() < 0.05:
		# 			self.node[i] += np.random.rand(3)*0.2-0.1
		# else:
		# 	self.node = np.copy(self.init_node)
		# #########
		
		# material(Young's modulus)
		self.material = np.ones(self.nm,dtype=np.float64)

		# initialize RL episode
		self.done = False
		self.steps = 0
		self.total_reward = 0

		# initialize edge existence and selection
		self.existence = np.ones(self.nm,dtype=bool)
		self.exist_member_i = np.arange(self.nm)
		
		# compute allowable disp by using full Level-1 GS
		self.section = np.ones(self.nm,dtype=np.float64)

		# connected members
		self.connected_members = [np.empty(0,dtype=int) for i in range(self.nk)]
		for i in range(self.nm):
			if self.existence[i]:
				for j in range(2):
					self.connected_members[self.connectivity[i,j]] = np.append(self.connected_members[self.connectivity[i,j]],i)

		# initialize edge inputs
		self.tsp_pos_before = []
		self.v, self.tsp_pos, _ = self.update_node_v()
		self.w = self.update_edge_w(np.copy(self.existence))

		self.infeasible_action = np.copy(~self.existence)
		
		return np.copy(self.connectivity),np.copy(self.v),np.copy(self.w),np.copy(self.infeasible_action)

	def update_node_v(self,v=None):
		'''
		0: binary feature if the node is pin-supported
		1: nodal stability
		'''
		v=np.zeros((self.nk,2),dtype=np.float64)
		v[self.pin_nodes,0] = 1.0

		sp = np.copy(self.support)
		node_existence = NodeExist(self.nk,self.nm,self.connectivity,self.existence)
		sp[~node_existence] = True

		if np.any(~sp):

			eig_val, eig_mode, instability_index = truss_analysis.StiffnessMatrixEig(self.node,self.connectivity,sp,self.section)
			tsp_pos = []
			
			while instability_index > 0:

				# node2 = np.concatenate([self.node,self.node+eig_mode[np.where(eig_val<self.tol)[0][0]]*0.5])
				# connectivity2 = np.concatenate([self.connectivity,self.connectivity+self.nk])
				# plotter.Draw(node2,connectivity2,np.tile(self.section,2),node_color=[(0.8,0.8,0.8)]*self.nk+[(0.0,0.0,0.0)]*self.nk,edge_color=[(0.8,0.8,0.8)]*self.nm+[(0.4,0.4,0.4)]*self.nm,save=False,show=True)
				# plotter.Draw(node2[:,0:2],connectivity2,np.tile(self.section,2),node_color=[(0.8,0.8,0.8)]*self.nk+[(0.0,0.0,0.0)]*self.nk,edge_color=[(0.8,0.8,0.8)]*self.nm+[(0.4,0.4,0.4)]*self.nm,save=False,show=True)

				em = eig_mode[0]
				order = np.concatenate([self.tsp_pos_before,np.argsort(np.linalg.norm(em,axis=1))[::-1]]).astype(int) # 既存の支保工を優先的に使用、その後は固有モードの変形量が大きい順
				j = 0
				instability_index_temp = instability_index
				while instability_index_temp >= instability_index:
					if node_existence[order[j]]:
						sp_temp = np.copy(sp)
						tsp_pos_temp = order[j]
						sp_temp[tsp_pos_temp] = True
						if np.any(~sp_temp):
							eig_val, eig_mode, instability_index_temp = truss_analysis.StiffnessMatrixEig(self.node,self.connectivity,sp_temp,self.section)
						else:
							instability_index_temp = 0
							break
					j += 1
				tsp_pos.append(tsp_pos_temp)
				sp[tsp_pos_temp] = True
				instability_index = instability_index_temp
			
			node_existence = NodeExist(self.nk,self.nm,self.connectivity,self.existence)
			node_stable = NodalStability(self.nk,self.node,self.connectivity,self.connected_members,self.dim)
			v[~node_stable,1] = 1.0
			v[self.pin_nodes,1] = 0.0

		else:
			tsp_pos = []

		return v[node_existence], tsp_pos, node_existence

	def update_edge_w(self,existence,w=None):

		'''
		0: 1 if exist, else 0
		'''
		w=np.zeros((self.nm,0),dtype=np.float64)

		return w[existence]

	def step(self, action):
		'''
		action(int):
		eliminate the corresponding edge
		'''
		a_to_m = self.exist_member_i[action]
		# print(a_to_m)
		self.exist_member_i = np.delete(self.exist_member_i,action)
		self.tsp_pos_before = self.tsp_pos.copy()

		assert self.existence[a_to_m] == True

		self.existence[a_to_m] = False
		self.section[a_to_m] = 0.0
		for i in range(2):
			self.connected_members[self.connectivity[a_to_m,i]] = np.setdiff1d(self.connected_members[self.connectivity[a_to_m,i]],a_to_m)

		self.v, self.tsp_pos, node_existence = self.update_node_v()
		c_temp = self.connectivity[self.existence]
		for i in np.where(node_existence==False)[0][::-1]:
			c_temp[c_temp>i] -= 1

		self.w = self.update_edge_w(np.copy(self.existence))

		reward = -len(self.tsp_pos)

		self.infeasible_action = np.copy(~self.existence)

		self.total_reward += reward
		self.steps += 1

		if np.all(self.infeasible_action):
			self.done=True

		return c_temp,np.copy(self.v),np.copy(self.w), reward, self.done, self.infeasible_action[self.existence], {}

	def render(self, q=None, close=False):
		nsize = [self.nsize for i in range(self.nk)]
		nshape = ['o' for i in range(self.nk)]
		ncolor = [(0.0,0.0,0.0) for i in range(self.nk)]
		for sn in self.pin_nodes:
			nshape[sn] = '^'
			nsize[sn] = self.spsize
		for tsp in self.tsp_pos:
			nshape[tsp] = '^'
			nsize[tsp] = self.spsize
			ncolor[tsp] = (1.0,0.0,0.0)

		outfile = plotter.Draw(self.node,self.connectivity,self.section,node_color=ncolor,node_size=nsize,node_shape=nshape,front_node_index=None,name=self.steps,scale=self.lsize)
		# outfile = plotter.Draw(self.node[:,0:2],self.connectivity,self.section,node_color=ncolor,node_size=nsize,node_shape=nshape,front_node_index=None,name=self.steps)

		return outfile
