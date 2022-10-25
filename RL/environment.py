import numpy as np
np.random.seed(1000)
from copy import deepcopy
import plotter
import agent

### User specified parameters ###
import truss_env as env
N_EDGE_FEATURE = 100
RECORD_INTERVAL = 10
#################################

class Environment():
	def __init__(self,gpu):
		self.env = env.Truss()
		_,v,w,_ = self.env.reset()
		self.n_edge_action = 1
		self.n_whole_action = 0
		self.agent = agent.Agent(v.shape[1],w.shape[1],N_EDGE_FEATURE,self.n_edge_action,gpu)
		if gpu:
			self.agent.brain.model = self.agent.brain.model.to("cuda")
		pass

	def Train(self,n_episode):

		history = np.zeros((3,n_episode//RECORD_INTERVAL),dtype=float)
		best_score = np.inf
		best_scored_iteration = -1
		best_model = deepcopy(self.agent.brain.model)
		n_analysis_until_best = 0
		n_analysis = 0
		eps0 = 0.2

		for episode in range(n_episode):

			c,v,w, infeasible_a = self.env.reset()
			total_reward = 0.0
			aveQ = 0.0
			aveloss = 0.0
			for t in range(self.env.nm):
				action,q = self.agent.get_action(v,w,c,eps0*(n_episode-episode)/n_episode, infeasible_a)
				aveQ += q
				c_next, v_next, w_next, reward, ep_end, infeasible_a, _ = self.env.step(action)
				self.agent.memorize(c,v,w,action,reward,c_next,v_next,w_next,ep_end,infeasible_a,t==self.env.nm-1)
				c = np.copy(c_next)
				v = np.copy(v_next)
				w = np.copy(w_next)
				aveloss += self.agent.update_q_function(episode/n_episode)
				total_reward += reward
				if ep_end:
					break

			print("episode {0:<4}: step={1:<3} reward={2:<+5.1f} aveQ={3:<+7.2f} loss={4:<7.2f}".format(episode,t+1,total_reward,aveQ/(t+1),aveloss/(t+1)))
			n_analysis += ((self.env.steps+1))
			if episode % RECORD_INTERVAL == RECORD_INTERVAL-1:
				score = 1.0
				for i in [1,2]:
					c,v,w, infeasible_a = self.env.reset(test=i)
					total_reward = 0.0
					for t in range(self.env.nm):
						action, _ = self.agent.get_action(v,w,c,0.0, infeasible_a)
						c, v, w, reward, ep_end, infeasible_a, _ = self.env.step(action)
						total_reward += reward
						if ep_end:
							break
					score *= total_reward # !!! Be careful to the sign of value
					history[i,episode//RECORD_INTERVAL] = total_reward
				if(score <= best_score):
					best_score = score
					best_scored_iteration = episode
					best_model = deepcopy(self.agent.brain.model)
					n_analysis_until_best = n_analysis
					
				history[0,episode//RECORD_INTERVAL] = score


		with open("result/info.txt", 'w') as f:
			f.write(str.format("total number of analysis:{0} \n",n_analysis))
			f.write(str.format("total number of analysis until best:{0} \n",n_analysis_until_best))
			f.write(str.format("top-scored iteration: {0} \n",best_scored_iteration+1))

		plotter.graph(history)

		np.savetxt('result/score.csv',history,fmt='%.4f',delimiter=',')

		best_model.Save(filename="trained_model_{0}".format(env.__name__))

	def Test(self):
		
		c,v,w, infeasible_a = self.env.reset(test=2)
		self.agent = agent.Agent(v.shape[1],w.shape[1],N_EDGE_FEATURE,self.n_edge_action,False)
		self.agent.brain.model.Load(filename="trained_model_{0}".format(env.__name__))

		# self.env.render()
		total_reward = 0.0

		for i in range(self.env.nm):
			# print('steps:'+str(i))
			# self.env.render()
			action, _ = self.agent.get_action(v,w,c,0.0, infeasible_a)
			c, v, w, reward ,ep_end, infeasible_a, _ = self.env.step(action)
			# print(action)
			total_reward += reward
			# print("Volume: {:}".format(self.env.volume.value))
			# print("Number of existing members {0}".format(np.sum(~self.env.infeasible_action)))
			# print("Number of existing nodes {0}".format(np.sum(self.env.node_existence)))
			if ep_end:
				# print("Volume: {:}".format(self.env.volume.value))
				# self.env.render()
				# print("Number of existing members {0}".format(np.sum(~self.env.infeasible_action)))
				# print("Number of existing nodes {0}".format(np.sum(self.env.node_existence)))
				break
			
		print(str.format("total rewards:{0}",total_reward))
