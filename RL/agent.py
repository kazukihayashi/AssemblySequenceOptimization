import os
import numpy as np
np.random.seed(0)
import random
random.seed(0)
import copy
import torch
torch.manual_seed(0)
from collections import deque
from dataclasses import dataclass
import pickle
import zlib

### User specified parameters ###
INIT_MEAN = 0.0 ## mean of initial training parameters
INIT_STD = 0.05 ## standard deviation of initial training parameters
TARGET_UPDATE_FREQ = 100
USE_BIAS = False
#################################

@dataclass
class Experience:
	c: np.ndarray # connectivity
	v: torch.Tensor
	w: torch.Tensor
	action: np.int32
	reward: np.float32
	c_next: np.ndarray
	v_next: torch.Tensor
	w_next: torch.Tensor
	done: bool
	infeasible_action: np.ndarray

@dataclass
class Temp_Experience:
	c: np.ndarray # connectivity
	v: torch.Tensor
	w: torch.Tensor
	action: np.int32
	reward: np.float32

class NN(torch.nn.Module):
	def __init__(self,n_node_inputs,n_edge_inputs,n_feature_outputs,n_action_types,batch_size,use_gpu):
		super(NN,self).__init__()
		self.l1_1 = torch.nn.Linear(n_edge_inputs,n_feature_outputs,False)
		self.l1_2 = torch.nn.Linear(n_feature_outputs,n_feature_outputs)
		self.l1_3 = torch.nn.Linear(n_node_inputs,n_feature_outputs)
		self.l1_4 = torch.nn.Linear(n_feature_outputs,n_feature_outputs)
		self.l1_5 = torch.nn.Linear(n_feature_outputs,n_feature_outputs)
		self.l1_6 = torch.nn.Linear(n_feature_outputs,n_feature_outputs)

		self.l2_1 = torch.nn.Linear(n_feature_outputs*2,n_action_types,bias=USE_BIAS)
		# self.l2_2 = torch.nn.Linear(n_feature_outputs,n_feature_outputs,bias=USE_BIAS)
		# self.l2_3 = torch.nn.Linear(n_feature_outputs,n_feature_outputs,bias=USE_BIAS)

		self.batch_size = batch_size
		self.ActivationF = torch.nn.LeakyReLU(0.2)

		self.Initialize_weight()

		self.n_feature_outputs = n_feature_outputs
		if use_gpu:
			self.to('cuda')
			self.device = torch.device('cuda')
		else:
			self.to('cpu')
			self.device = torch.device('cpu')

	def Connectivity(self,connectivity,n_nodes):
		'''
		connectivity[n_edges,2]
		'''
		n_edges = connectivity.shape[0]
		order = np.arange(n_edges)
		adjacency = torch.zeros(n_nodes,n_nodes,dtype=torch.float32,device=self.device,requires_grad=False)
		incidence = torch.zeros(n_nodes,n_edges,dtype=torch.float32,device=self.device,requires_grad=False)

		for i in range(2):
			adjacency[connectivity[:,i],connectivity[:,(i+1)%2]] = 1
		incidence[connectivity[:,0],order] = -1
		incidence[connectivity[:,1],order] = 1

		incidence_A = torch.abs(incidence)#.to_sparse()
		incidence_1 = (incidence==-1).type(torch.float32)
		incidence_2 = (incidence==1).type(torch.float32)

		return incidence_A,incidence_1,incidence_2,adjacency

	def Initialize_weight(self):
		for m in self._modules.values():
			if isinstance(m,torch.nn.Linear):
				torch.nn.init.normal_(m.weight,mean=0,std=INIT_STD)

	def Output_params(self):
		for name,m in self.named_modules():
			if isinstance(m,torch.nn.Linear):
				print(name)
				np.savetxt(f"agent_params/{name}_w.npy",m.weight.detach().to('cpu').numpy())
				if m.bias != None:
					np.savetxt(f"agent_params/{name}_b.npy",m.bias.detach().to('cpu').numpy())

	def mu(self,v,mu,w,incidence_A,incidence_1,incidence_2,adjacency,mu_iter):
		'''
		v (array[n_nodes,n_node_features])
		mu(array[n_edges,n_edge_out_features])
		w (array[n_edges,n_edge_in_features])
		'''
		if mu_iter == 0:
			h1 = self.l1_1.forward(w)
			h2_0 = self.ActivationF(self.l1_3.forward(v))
			h2 = self.l1_2.forward(torch.mm(incidence_A.T,h2_0))
			mu = self.ActivationF(h1+h2)

		else:
			h3 = self.l1_6.forward(mu)
			h4_0 = torch.mm(incidence_A,mu)
			n_connect_edges_1 = torch.clip(torch.sum(torch.mm(adjacency.T,incidence_1),axis=0).repeat(self.n_feature_outputs,1).T-1,1)
			n_connect_edges_2 = torch.clip(torch.sum(torch.mm(adjacency.T,incidence_2),axis=0).repeat(self.n_feature_outputs,1).T-1,1)
			h4_1 = self.l1_4.forward(torch.mm(incidence_1.T,h4_0)-mu)/n_connect_edges_1
			h4_2 = self.l1_4.forward(torch.mm(incidence_2.T,h4_0)-mu)/n_connect_edges_2
			h4 = self.l1_5.forward(self.ActivationF(h4_1)+self.ActivationF(h4_2))
			mu = self.ActivationF(h3+h4)
		return mu
		
	def Q(self,mu,n_edges):
		
		if type(n_edges) is int: # normal operation
			mu_sum = torch.sum(mu,axis=0)
			mu_sum = mu_sum.repeat(n_edges,1)
		else: # for mini-batch training
			mu_sum = torch.zeros((n_edges[-1],self.n_feature_outputs),dtype=torch.float32,device=self.device)
			for i in range(self.batch_size):
				mu_sum[n_edges[i]:n_edges[i+1],:] = torch.sum(mu[n_edges[i]:n_edges[i+1],:],axis=0)

		Q = self.l2_1(torch.cat((mu_sum,mu),1))
		return Q

	def Forward(self,v,w,connectivity,n_mu_iter=3,nm_batch=None):
	   
		'''
		v[n_nodes,n_node_in_features]
		w[n_edges,n_edge_in_features]
		connectivity[n_edges,2]
		nm_batch[BATCH_SIZE] : int
		'''
		IA,I1,I2,D = self.Connectivity(connectivity,v.shape[0])

		if type(v) is np.ndarray: 
			v = torch.tensor(v,dtype=torch.float32,device=self.device,requires_grad=False)
		if type(w) is np.ndarray:
			w = torch.tensor(w,dtype=torch.float32,device=self.device,requires_grad=False)
		mu = torch.zeros((connectivity.shape[0],self.n_feature_outputs),device=self.device)

		for i in range(n_mu_iter):
			mu = self.mu(v,mu,w,IA,I1,I2,D,mu_iter=i)
			# print("iter {0}: {1}".format(i,mu.norm(p=2)))
		if nm_batch is None:
			Q = self.Q(mu,w.shape[0])
		else:
			Q = self.Q(mu,nm_batch)

		Q = Q.flatten()

		return Q

	def Save(self,filename,directory=""):
		torch.save(self.to('cpu').state_dict(),os.path.join(directory,filename))
	
	def Load(self,filename,directory=""):
		self.load_state_dict(torch.load(os.path.join(directory,filename)))

class Brain():
	def __init__(self,n_node_inputs,n_edge_inputs,n_feature_outputs,n_action_types,use_gpu):
		if use_gpu:
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')
		self.n_node_inputs = n_node_inputs
		self.n_edge_inputs = n_edge_inputs
		self.batch_size = 32
		self.model = NN(n_node_inputs,n_edge_inputs,n_feature_outputs,n_action_types,self.batch_size,use_gpu)
		self.target_model = copy.deepcopy(self.model)
		self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1.0e-4) # RMSprop(self.model.parameters(),lr=1.0e-5)

		self.n_edge_action_type = n_action_types
		self.n_step = 3
		self.gamma = 1.0
		self.nstep_gamma = np.power(self.gamma,self.n_step)
		self.temp_buffer = deque(maxlen=self.n_step)
		self.capacity = int(1E4)
		self.buffer = deque([None for _ in range(self.capacity)],maxlen=self.capacity)
		self.tdfunc = torch.nn.L1Loss(reduction='none')
		self.priority = np.zeros(self.capacity,dtype=np.float32)
		self.max_priority = 1.0
		self.beta_scheduler = lambda progress: 0.4 + 0.6*progress
		self.store_count = 0

	def store_experience(self,c,v,w,action,reward,c_next,v_next,w_next,done,infeasible_action,stop):

		v = torch.tensor(v,dtype=torch.float32,device=self.device,requires_grad=False)
		w = torch.tensor(w,dtype=torch.float32,device=self.device,requires_grad=False)
		v_next = torch.tensor(v_next,dtype=torch.float32,device=self.device,requires_grad=False)
		w_next = torch.tensor(w_next,dtype=torch.float32,device=self.device,requires_grad=False)
		self.temp_buffer.append(Temp_Experience(c,v,w,action,reward))

		### Using Multistep learning ###

		if done or stop:
			for j in range(len(self.temp_buffer)):
				nstep_return = np.sum([self.gamma ** (i-j) * self.temp_buffer[i].reward for i in range(j,len(self.temp_buffer))])
				self.priority[0:-1], self.priority[-1] = self.priority[1:], self.max_priority
				# self.buffer.append(zlib.compress(pickle.dumps(Experience(self.temp_buffer[j].c,self.temp_buffer[j].v,self.temp_buffer[j].w,self.temp_buffer[j].action,nstep_return,c_next,v_next,w_next,done,infeasible_action))))
				self.buffer.append(Experience(self.temp_buffer[j].c,self.temp_buffer[j].v,self.temp_buffer[j].w,self.temp_buffer[j].action,nstep_return,c_next,v_next,w_next,done,infeasible_action))
				self.store_count += 1
			self.temp_buffer.clear()

		elif len(self.temp_buffer) == self.n_step:
			nstep_return = np.sum([self.gamma ** i * self.temp_buffer[i].reward for i in range(len(self.temp_buffer))])
			self.priority[0:-1], self.priority[-1] = self.priority[1:], self.max_priority
			# self.buffer.append(zlib.compress(pickle.dumps(Experience(self.temp_buffer[0].c,self.temp_buffer[0].v,self.temp_buffer[0].w,self.temp_buffer[0].action,nstep_return,c_next,v_next,w_next,done,infeasible_action))))
			self.buffer.append(Experience(self.temp_buffer[0].c,self.temp_buffer[0].v,self.temp_buffer[0].w,self.temp_buffer[0].action,nstep_return,c_next,v_next,w_next,done,infeasible_action))			
			self.store_count += 1


	def sample_batch(self,progress):
		p = self.priority/self.priority.sum()
		indices = np.random.choice(self.capacity,p=p,replace=False,size=self.batch_size)

		weight = np.power(p[indices]*self.capacity,-self.beta_scheduler(progress))
		weight /= np.max(weight)

		# batch = [pickle.loads(zlib.decompress(self.buffer[i])) for i in indices]
		batch = [self.buffer[i] for i in indices]

		c_batch = np.zeros((0,2),dtype=int)
		v_batch = torch.cat([dat.v for dat in batch],dim=0)
		w_batch = torch.cat([dat.w for dat in batch],dim=0)
		a_batch = np.array([dat.action for dat in batch])
		r_batch = torch.tensor([dat.reward for dat in batch],dtype=torch.float32,device=self.device,requires_grad=False)
		c2_batch = np.zeros((0,2),dtype=int)
		v2_batch = torch.cat([dat.v_next for dat in batch],dim=0)
		w2_batch = torch.cat([dat.w_next for dat in batch],dim=0)
		done_batch= torch.tensor([dat.done for dat in batch],dtype=bool,device=self.device,requires_grad=False)
		infeasible_a_batch = np.concatenate([dat.infeasible_action for dat in batch],axis=0)
		nm_batch = np.zeros(self.batch_size+1,dtype=int)
		nm2_batch = np.zeros(self.batch_size+1,dtype=int)

		nn = 0
		nm = 0
		for i in range(self.batch_size):
			c_batch = np.concatenate((c_batch,batch[i].c+nn),axis=0)
			nn += batch[i].v.shape[0]
			nm += batch[i].w.shape[0]
			nm_batch[i+1] = nm
		a_batch += nm_batch[:-1]*self.n_edge_action_type

		nn2 = 0
		nm2 = 0
		for i in range(self.batch_size):
			c2_batch = np.concatenate((c2_batch,batch[i].c_next+nn2),axis=0)
			nn2 += batch[i].v_next.shape[0]
			nm2 += batch[i].w_next.shape[0]
			nm2_batch[i+1] = nm2

		return weight,indices,c_batch,v_batch,w_batch,a_batch,r_batch,c2_batch,v2_batch,w2_batch,done_batch,infeasible_a_batch,nm_batch,nm2_batch

	def update_priority(self,indices,td_errors):
		pri = np.power((np.abs(td_errors.detach().to('cpu').numpy()) + 1e-3),0.6)
		self.priority[indices] = pri
		self.max_priority = max(self.max_priority,np.max(pri))
		return

	def experience_replay(self,progress):
		if self.store_count < self.batch_size:
			return float('nan')
		
		weight,indices,c,v,w,a,r,c_next,v_next,w_next,done,infeasible_action,nm,nm_next = self.sample_batch(progress)
		self.optimizer.zero_grad()
		td_errors = self.calc_td_error(c,v,w,a,r,c_next,v_next,w_next,done,infeasible_action,nm,nm_next)
		loss = torch.mean(torch.pow(td_errors,2)*torch.from_numpy(weight).clone().to(self.device))
		loss.backward()
		self.optimizer.step()
		self.update_priority(indices,td_errors)
		return loss.item()

	def calc_td_error(self,c,v,w,action,r,c_next,v_next,w_next,done,infeasible_action,nm,nm_next):

		current_Q = self.model.Forward(v,w,c,nm_batch=nm)
		next_QT = self.target_model.Forward(v_next,w_next,c_next,nm_batch=nm_next).detach()
		next_Q = self.model.Forward(v_next,w_next,c_next,nm_batch=nm_next).detach()

		### Without Double DQN ###
		next_QT[infeasible_action] = -1.0e20
		Q_max_next = torch.tensor([next_QT[nm_next[i]*self.n_edge_action_type:nm_next[i+1]*self.n_edge_action_type].max() if nm_next[i] != nm_next[i+1] else 0.0 for i in range(self.batch_size)],dtype=torch.float32,device=self.device,requires_grad=False)

		# ### Using Double DQN ###
		# next_Q[infeasible_action] = -1.0e20
		# action_next = [nm_next[i]*self.n_edge_action_type + next_Q[nm_next[i]*self.n_edge_action_type:nm_next[i+1]*self.n_edge_action_type].argmax().item() for i in range(self.batch_size)]
		# Q_max_next = next_QT[action_next]

		Q_target = r+(self.nstep_gamma*Q_max_next)*~done
		td_errors = self.tdfunc(current_Q[action],Q_target) # In rainbow, not td_error but KL divergence loss is used to 

		return td_errors

	def decide_action(self,v,w,c,eps,infeasible_actions):

		Q = self.model.Forward(v,w,c).detach().to('cpu').numpy()
		
		if np.random.rand() > eps:
			a = np.ma.masked_where(infeasible_actions,Q).argmax()
		else:
			a = np.random.choice(np.argwhere(~infeasible_actions)[:,0])

		return a, Q[a]

class Agent():

	def __init__(self,n_node_inputs,n_edge_inputs,n_feature_outputs,n_action_types,use_gpu):
		self.brain = Brain(n_node_inputs,n_edge_inputs,n_feature_outputs,n_action_types,use_gpu)     
		self.step = 0
		self.n_update = 0
		self.target_update_freq = TARGET_UPDATE_FREQ

	def update_q_function(self,progress):
		'''
		progress<float> : The overall progress of the training. Take a value within [0,1].
		'''
		loss = self.brain.experience_replay(progress)
		if self.n_update % self.target_update_freq == 0:
			self.brain.target_model.load_state_dict(self.brain.model.state_dict())

		self.n_update += 1
		return loss
		
	def get_action(self,v,w,c,eps,infeasible_actions):
		action, q = self.brain.decide_action(v,w,c,eps=eps,infeasible_actions=infeasible_actions)
		return action, q
	
	def memorize(self,c,v,w,action,reward,c_next,v_next,w_next,ep_end,infeasible_actions,stop):
		self.brain.store_experience(c,v,w,action,reward,c_next,v_next,w_next,ep_end,infeasible_actions,stop)



