import numpy as np
np.random.seed(0)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import plotter
import truss_analysis
from numba import i4, b1
import numba as nb
import init_geom

CACHE = False

@nb.jit((b1[:])(i4,i4,i4[:,:],b1[:]),parallel=True,cache=CACHE)
def NodeExist(nk,nm,connectivity,existence):
	node_existence = np.zeros(nk,dtype=np.bool_)
	for i in range(nm):
		if existence[i]:
			node_existence[connectivity[i,:]] = True
	return node_existence

class Truss():

	def __init__(self,test=0):
		self.test = test
		self.reset()

	def reset(self):

		# ### 2D ###
		# if self.test==1:
		# 	self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry2D_grid_doublebrace(1,3,test=True)
		# elif self.test==2:
		# 	self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry2D_arch()
		# elif self.test==3:
		# 	self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry2D_tower()
		# else:
		# 	self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry2D_grid_doublebrace(np.random.randint(3,6),np.random.randint(3,6),test=False)
		# self.support = np.zeros((self.nk,3),dtype=bool)
		# self.support[:,2] = True # !!! only 3D
		# self.support[self.pin_nodes] = True

		# if self.test == 0:
		# 	self.node = np.copy(self.init_node) # self.init_node+np.hstack((np.random.rand(self.nk,2)*0.2-0.1,np.zeros((self.nk,1)))).astype(np.float64)
		# else:
		# 	self.node = np.copy(self.init_node)
		# ########

		## 3D ###
		if self.test==1:
			self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry3D_grid(6,2,test=True)
		elif self.test==2:
			self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry3D_dome(3,8)
		elif self.test==3:
			self.init_node = np.loadtxt("input/node.txt",dtype=float)
			self.connectivity = np.loadtxt("input/member.txt",dtype=int)
			self.pin_nodes = np.loadtxt("input/pin_node.txt",dtype=int)
			self.nk = self.init_node.shape[0]
			self.nm = self.connectivity.shape[0]
		else:
			self.nk,self.nm,self.init_node,self.connectivity,self.pin_nodes = init_geom.InitializeGeometry3D_grid(np.random.randint(2,5),np.random.randint(2,5),test=False)
		self.support = np.zeros((self.nk,3),dtype=bool)
		self.support[self.pin_nodes] = True

		if self.test == 0:
			self.node = np.copy(self.init_node)#self.init_node+np.random.rand(self.nk,3).astype(np.float64)*0.2-0.1
		else:
			self.node = np.copy(self.init_node)
		# ##########
		
		# material(Young's modulus)
		self.material = np.ones(self.nm,dtype=np.float64)

		# initialize RL episode
		self.done = False
		self.steps = 0
		self.total_reward = 0

		# initialize edge existence and selection
		self.existence = np.ones(self.nm,dtype=bool)
		
		# compute allowable disp by using full Level-1 GS
		self.section = np.ones(self.nm,dtype=np.float64)

		# randomize connectivity
		self.connected_members = [np.empty(0,dtype=int) for i in range(self.nk)]
		for i in range(self.nm):
			if self.existence[i]:
				for j in range(2):
					self.connected_members[self.connectivity[i,j]] = np.append(self.connected_members[self.connectivity[i,j]],i)

		# initialize edge inputs
		self.tsp_pos_before = []
		self.tsp_pos = self.update_node_v()

		self.infeasible_action = np.copy(~self.existence)
		
		return

	def update_node_v(self):

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
				# order = np.argsort(np.linalg.norm(em,axis=1))[::-1] # 固有モードの変形量が大きい順にソート
				order = np.concatenate([self.tsp_pos_before,np.argsort(np.linalg.norm(em,axis=1))[::-1]]).astype(int) # 既存の支保工を優先的に使用
				j = 0
				instability_index_temp = instability_index
				while instability_index_temp >= instability_index: # 仮設支持を付加してinstability_indexが増えることは理論上はないが、念のため>=としている
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

		else:
			tsp_pos = []

		return tsp_pos

	def step(self, action):
		'''
		action(int):
		eliminate the corresponding edge
		'''
		assert self.existence[action] == True
		self.tsp_pos_before = self.tsp_pos.copy()

		self.existence[action] = False
		self.section[action] = 0.0
		for i in range(2):
			self.connected_members[self.connectivity[action,i]] = np.setdiff1d(self.connected_members[self.connectivity[action,i]],action)

		self.tsp_pos = self.update_node_v()

		# tsp_pos_kept = set(self.tsp_pos_before) & set(self.tsp_pos)
		reward = -len(self.tsp_pos) # -(len(tsp_pos_kept)*0.25+(len(self.tsp_pos)-len(tsp_pos_kept))*0.5)  # -(len(tsp_pos_kept)*0.25+(len(self.tsp_pos)-len(tsp_pos_kept))*0.5)  #  - 0.25 * kept constuction support -0.5 * addition of construction support

		self.infeasible_action = np.copy(~self.existence)

		self.total_reward += reward
		self.steps += 1

		if np.all(self.infeasible_action):
			self.done=True

		return

	def render(self, q=None, close=False):
		nsize = [3 for i in range(self.nk)]
		nshape = ['o' for i in range(self.nk)]
		ncolor = [(0.0,0.0,0.0) for i in range(self.nk)]
		for sn in self.pin_nodes:
			nshape[sn] = '^'
			nsize[sn] = 15
		for tsp in self.tsp_pos:
			nshape[tsp] = '^'
			nsize[tsp] = 15
			ncolor[tsp] = (1.0,0.0,0.0)

		outfile = plotter.Draw(self.node,self.connectivity,self.section,node_color=ncolor,node_size=nsize,node_shape=nshape,front_node_index=None,name=self.steps)
		outfile = plotter.Draw(self.node[:,0:2],self.connectivity,self.section,node_color=ncolor,node_size=nsize,node_shape=nshape,front_node_index=None,name=self.steps)

		return outfile

	def func(self,x,functype):
		'''
		functype: 'minfun' or 'maxfun'
		'''
		self.reset()
		for i in range(self.nm):
			self.step(x[i])
		if functype == 'minfun':
			objfun = -self.total_reward # CMA-ES, minfun
		elif functype == 'maxfun':
			objfun = 1/-self.total_reward # GA, maxfun
		return objfun, True

	def func_render(self,x):
		self.reset()
		self.render()
		for i in range(self.nm):
			self.step(x[i])
			self.render()
