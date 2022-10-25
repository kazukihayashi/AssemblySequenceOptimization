import time
import environment

### User specified parameters ###
USE_GPU = True
N_EPISODE = 5000
#################################

## Train ##

t0 = time.time()

env = environment.Environment(gpu=USE_GPU)
env.Train(N_EPISODE)

t1 = time.time()

print("time: {:.3f} seconds".format(t1-t0))


env.Test()
# env.agent.brain.model.Output_params()
# for i in range(5):
#     env.Test()

t2 = time.time()

print("time: {:.3f} seconds".format(t2-t1))
# print(env.env.total_reward)
