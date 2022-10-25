import numpy as np
np.random.seed(0)

from deap import base
from deap import cma
from deap import creator
from deap import tools
import time

'''
CMA-ES
'''

import truss_env
env = truss_env.Truss(test=3)
env.reset()

def func(x):
    x2 = np.argsort(x)
    f,c = env.func(x2,functype='minfun')
    return f,

N = env.nm  # 問題の次元
NGEN = 20  # 総ステップ数

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("evaluate", func)

def main():

    # The CMA-ES algorithm 
    strategy = cma.Strategy(centroid=[0.0]*N, sigma=1.0, lambda_=50)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1)

    halloffame_array = []
    C_array = []
    centroid_array = []
    for gen in range(NGEN):
        # 新たな世代の個体群を生成
        population = toolbox.generate()
        # 個体群の評価
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 個体群の評価から次世代の計算のためのパラメタ更新
        toolbox.update(population)

        # hall-of-fameの更新
        halloffame.update(population)

        halloffame_array.append(halloffame[0])
        C_array.append(strategy.C)
        centroid_array.append(strategy.centroid)

    x2 = np.argsort(halloffame_array[-1])
    f2,_ = env.func(x2,functype='minfun')
    print(f2)
    # env.func_render(x2)
    pass

t1 = time.time()
for i in range(5):
    main()
t2 = time.time()
print("time: {:.3f} seconds".format(t2-t1))